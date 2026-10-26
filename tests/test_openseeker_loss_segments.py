"""OpenSeeker segmentation and weighted causal CE, with offline tokenizers/models."""
import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from search_comp.data.openseeker import encode, normalize, OpenSeekerDataset
from search_comp.data.searchr1_dataset import SearchR1Collator
from search_comp.loss_segments import (LossSegmentConfig, SEGMENT_IDS, SEGMENT_NAMES,
    SEGMENT_IGNORE, classify_assistant_text, segment_ids_to_weights)
from search_comp.trainer.plain_sft_trainer import PlainSFTDataset, collate_sft_batch
from search_comp.trainer.segmented_sft_trainer import segmented_causal_loss, SegmentedSFTTrainer
from test_openseeker import polar, Tokenizer


class OffsetTokenizer(Tokenizer):
    def __call__(self,text,**kwargs):
        return {'input_ids':self.encode(text),
                'offset_mapping':[(i,i+1) for i in range(len(text))]}


def test_classifies_final_answer_and_nested_tool_calls():
    text='<think>reason</think><tool_calls_begin>\n<tool_call>{"name":"search"}</tool_call>\n<tool_call>{"name":"visit"}</tool_call></tool_calls_end><answer>explain</answer><final_answer>Alice</final_answer>'
    kinds=classify_assistant_text(text,OffsetTokenizer())
    def kind_at(fragment):
        return SEGMENT_NAMES[kinds[text.index(fragment)]]
    assert kind_at('reason')=='think_content'
    assert kind_at('<think>')=='think_tag'
    assert kind_at('explain')=='answer_content'
    assert kind_at('Alice')=='final_answer_content'
    assert kind_at('<final_answer>')=='final_answer_tag'
    assert kind_at('{"name":"search"}')=='tool_call_content'
    assert kind_at('{"name":"visit"}')=='tool_call_content'
    assert kind_at('</tool_calls_end>')=='tool_call_tag'
    # Whitespace between nested calls stays in the outer envelope category.
    assert SEGMENT_NAMES[kinds[text.index('</tool_call>')+len('</tool_call>')]]=='tool_call_content'


def test_old_category_ids_and_config_roundtrip():
    assert SEGMENT_NAMES[:7]==('think_tag','think_content','search_tag','search_content',
                               'answer_tag','answer_content','other')
    config=LossSegmentConfig.from_dict({'final_answer_content':10,'answer_content':False})
    restored=LossSegmentConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    assert restored.to_dict()==config.to_dict()
    assert restored.answer_content.effective_weight==0
    assert restored.final_answer_content.effective_weight==10


def test_dataset_weights_preserve_tokens_masks_and_compression_regions(tmp_path):
    row=normalize(polar());tok=OffsetTokenizer()
    path=tmp_path/'train.jsonl';path.write_text(json.dumps(row)+'\n')
    loss=LossSegmentConfig.from_dict({'answer_content':0.2,'final_answer_content':10})
    ordinary=encode(row['messages'],tok)
    weighted=OpenSeekerDataset(path,tok,loss_segments=loss)[0]
    for key in ['input_ids','labels','regions']:
        assert weighted[key]==ordinary[key]
    kinds=weighted['loss_segment_ids']
    assert len(kinds)==len(weighted['input_ids'])
    for label,kind in zip(weighted['labels'],kinds):
        assert (kind==SEGMENT_IGNORE)==(label==-100)
    weights=segment_ids_to_weights(torch.tensor([kinds]),loss)[0]
    assert weights[kinds.index(SEGMENT_IDS['final_answer_content'])]==10
    assert weights[kinds.index(SEGMENT_IDS['answer_content'])]==pytest.approx(0.2)
    batch=SearchR1Collator(tok)([weighted])
    assert batch['loss_segment_ids'].shape==batch['labels'].shape
    plain=collate_sft_batch([PlainSFTDataset(path,tok,data_mode='openseeker',loss_segments=loss)[0]])
    torch.testing.assert_close(plain['loss_segment_ids'],batch['loss_segment_ids'])
    torch.testing.assert_close(plain['labels'],batch['labels'])
    assert encode(row['messages'],tok,loss_segments=LossSegmentConfig())['loss_segment_ids'] is None


def test_weighted_encoding_requires_aligned_offsets():
    row=normalize(polar())
    class NoOffsets(Tokenizer):
        def __call__(self,text,**kwargs):return {'input_ids':self.encode(text)}
    with pytest.raises(ValueError,match='offset_mapping'):
        encode(row['messages'],NoOffsets(),loss_segments={'final_answer_content':10})
    class WrongIDs(OffsetTokenizer):
        def __call__(self,text,**kwargs):
            value=super().__call__(text,**kwargs);value['input_ids']=[0]*len(text);return value
    with pytest.raises(ValueError,match='tokenizer IDs'):
        encode(row['messages'],WrongIDs(),loss_segments={'final_answer_content':10})


def targets():
    labels=torch.tensor([[-100,2,3,4,-100]])
    kinds=torch.tensor([[SEGMENT_IGNORE,SEGMENT_IDS['think_content'],
        SEGMENT_IDS['answer_content'],SEGMENT_IDS['final_answer_content'],SEGMENT_IGNORE]])
    return labels,kinds


def test_plain_weighted_loss_shifts_categories_with_targets_and_zeroes_gradients():
    torch.manual_seed(7)
    logits=torch.randn(1,5,8,requires_grad=True)
    labels,kinds=targets()
    config=LossSegmentConfig.from_dict({'think_content':0,'answer_content':0.2,'final_answer_content':10})
    loss=segmented_causal_loss(logits,labels,kinds,config)
    expected=(F.cross_entropy(logits[0,1:2],labels[0,2:3])*0.2+
              F.cross_entropy(logits[0,2:3],labels[0,3:4])*10)/10.2
    torch.testing.assert_close(loss,expected)
    loss.backward()
    assert torch.count_nonzero(logits.grad[0,0])==0
    assert torch.count_nonzero(logits.grad[0,3:])==0
    assert torch.count_nonzero(logits.grad[0,1:3])>0


def test_plain_defaults_match_causal_ce_and_all_disabled_is_finite_zero():
    logits=torch.randn(1,5,8,requires_grad=True)
    labels,kinds=targets()
    expected=F.cross_entropy(logits[:,:-1].reshape(-1,8),labels[:,1:].reshape(-1))
    torch.testing.assert_close(segmented_causal_loss(logits,labels,kinds,LossSegmentConfig()),expected)
    config=LossSegmentConfig.from_dict({name:False for name in SEGMENT_NAMES})
    loss=segmented_causal_loss(logits,labels,kinds,config)
    assert loss.item()==0 and torch.isfinite(loss)
    loss.backward()
    assert torch.count_nonzero(logits.grad)==0


def test_plain_trainer_passes_only_model_inputs():
    labels,kinds=targets();seen={}
    logits=torch.randn(1,5,8,requires_grad=True)
    def model(**kwargs):
        seen.update(kwargs);return SimpleNamespace(logits=logits)
    trainer=object.__new__(SegmentedSFTTrainer)
    trainer.loss_segments=LossSegmentConfig.from_dict({'final_answer_content':10})
    inputs={'input_ids':torch.ones_like(labels),'attention_mask':torch.ones_like(labels),
            'labels':labels,'loss_segment_ids':kinds}
    loss,outputs=trainer.compute_loss(model,inputs,return_outputs=True)
    assert set(seen)=={'input_ids','attention_mask'}
    assert set(inputs)=={'input_ids','attention_mask','labels','loss_segment_ids'}
    torch.testing.assert_close(loss,segmented_causal_loss(outputs.logits,labels,kinds,trainer.loss_segments))


@pytest.mark.parametrize('family',['qwen3','qwen35'])
def test_beacon_full_forward_matches_plain_manual_weighting(family):
    from test_qwen3_whole_information import tiny_qwen3, whole_config
    from test_beacon_qwen35_linear_memory import _tiny_model
    model=tiny_qwen3() if family=='qwen3' else _tiny_model()
    config=whole_config(model,'append',False)
    config.loss_segments=LossSegmentConfig.from_dict({'think_content':0,'answer_content':0.2,
                                                       'final_answer_content':10})
    model.set_beacon_config(config)
    ids=torch.tensor([[3,4,5,6,7,8,9,10]])
    labels=torch.full_like(ids,-100);labels[0,5:]=ids[0,5:]
    kinds=torch.full_like(ids,SEGMENT_IGNORE)
    kinds[0,5:]=torch.tensor([SEGMENT_IDS['think_content'],SEGMENT_IDS['answer_content'],
                             SEGMENT_IDS['final_answer_content']])
    model.eval()
    parts=[]
    original_loss=model._sparse_window_loss
    def record(hidden,targets,valid_num,weights=None):
        actual,mass=original_loss(hidden,targets,valid_num,weights)
        if weights is not None:
            ce=F.cross_entropy(model.lm_head(hidden)[0].float(),targets[0],reduction='none')
            expected=(ce*weights[0]).sum()/weights[0].sum().clamp(min=1e-6)
            torch.testing.assert_close(actual[0],expected)
            parts.append((actual[0],mass[0]))
        return actual,mass
    model._sparse_window_loss=record
    output=model(input_ids=ids,labels=labels,regions=[(1,4)],loss_segment_ids=kinds)
    assert torch.isfinite(output.loss) and output.loss>0
    assert parts
    expected=sum(value*mass for value,mass in parts)/sum(mass for _,mass in parts)
    torch.testing.assert_close(output.loss,expected)
    model._sparse_window_loss=original_loss
    # Same hidden states, manually read out only the supervised prediction rows.
    # _sparse_window_loss uses already shifted targets/categories.
    hidden=torch.randn(1,4,model.config.hidden_size,requires_grad=True)
    shifted=torch.tensor([[7,8,9,-100]])
    weights=torch.tensor([[0.,0.2,10.,0.]])
    actual,_=model._sparse_window_loss(hidden,shifted,(shifted!=-100).sum(-1),weights)
    per_token=F.cross_entropy(model.lm_head(hidden)[0].float(),shifted[0],reduction='none')
    expected=(per_token*weights[0]).sum()/10.2
    torch.testing.assert_close(actual[0],expected)
    grad=torch.autograd.grad(actual.sum(),hidden)[0]
    assert torch.count_nonzero(grad[0,0])==0 and torch.count_nonzero(grad[0,3])==0


def test_stratified_distillation_selects_final_answer_and_respects_disabled_segments():
    from search_comp.trainer.beacon_distillation_trainer import select_distillation_positions
    labels=torch.tensor([[-100,-100,-100,-100,4,5,6,7]])
    kinds=torch.tensor([[SEGMENT_IGNORE]*4+[SEGMENT_IDS[name] for name in
        ('think_content','tool_call_content','final_answer_content','answer_content')]])
    selected=select_distillation_positions(labels,[(1,3)],tokens=2,strategy='stratified',
        loss_segment_ids=kinds,loss_segments=LossSegmentConfig())
    assert 5 in selected.tolist()  # row 5 predicts the final_answer token at position 6
    disabled=select_distillation_positions(labels,[(1,3)],tokens=2,strategy='stratified',
        loss_segment_ids=kinds,loss_segments=LossSegmentConfig.from_dict({'final_answer_content':False}))
    assert 5 not in disabled.tolist()
