"""Protocol integrity, loss boundaries, train/inference parity and tool routing."""
import copy
import json
import pytest
from search_comp.data.openseeker import normalize, encode, parse_calls, validate, OpenSeekerDataset
from search_comp.evaluation.tool_agent import Registry, ReplayTools, ToolError, run_agent


class Tokenizer:
    pad_token_id=0
    def encode(self,text,add_special_tokens=False):return [ord(c) for c in text]


def polar():
    return dict(question='Who?',answer='Alice',trajectory=[
        dict(role='system',content='Use search and visit.'),dict(role='user',content='Who?'),
        dict(role='assistant',content='<think>Look up.</think><tool_calls_begin>\n<tool_call>{"name":"search","arguments":{"query":["a","b"]}}</tool_call>\n</tool_calls_end>'),
        dict(role='user',content='<tool_response>search result</tool_response>'),
        dict(role='assistant',content='<tool_calls_begin><tool_call>{"name":"visit","arguments":{"url":["https://x"],"goal":"identity"}}</tool_call></tool_calls_end>'),
        dict(role='user',content='<tool_response>page evidence</tool_response>'),
        dict(role='assistant',content='<think>Done.</think><answer>Alice</answer>')])


def test_polar_does_not_replace_system_or_tool_arguments():
    raw=polar(); saved=copy.deepcopy(raw); row=normalize(raw)
    assert raw==saved
    assert row['messages']==raw['trajectory']
    assert parse_calls(row['messages'][4]['content'])[0]['arguments']['goal']=='identity'


def test_loss_and_regions_only_observation_bodies():
    row=normalize(polar()); sample=encode(row['messages'],Tokenizer())
    assert [''.join(chr(c) for c in sample['input_ids'][a:b]) for a,b in sample['regions']]==['search result','page evidence']
    for a,b in sample['regions']:assert set(sample['labels'][a:b])=={-100}
    supervised=''.join(chr(c) for c in sample['labels'] if c!=-100)
    assert 'search result' not in supervised and 'page evidence' not in supervised
    assert '"goal":"identity"' in supervised and '<answer>Alice</answer>' in supervised
    assert '<|im_end|>' in supervised


def test_every_inference_prefix_matches_training_exactly():
    row=normalize(polar());tok=Tokenizer(); full=encode(row['messages'],tok)
    for i,m in enumerate(row['messages']):
        if m['role']!='assistant':continue
        prefix=encode(row['messages'][:i],tok,generation_prompt=True)
        n=len(prefix['input_ids'])
        assert prefix['input_ids']==full['input_ids'][:n]
        assert prefix['regions']==[(a,b) for a,b in full['regions'] if b<=n]


def test_structured_calls_reasoning_null_content_and_ids():
    raw=dict(question='q',answer='a',tools=[dict(type='function',function=dict(name='custom',parameters={}))],messages=[
        dict(role='system',content='Original instructions.'),dict(role='user',content='q'),
        dict(role='assistant',content=None,reasoning_content='Reason',tool_calls=[dict(id='c1',function=dict(name='custom',arguments='{"n": 7}'))]),
        dict(role='tool',tool_call_id='c1',content='result'),dict(role='assistant',content='a')])
    row=normalize(raw)
    assert 'Original instructions.' in row['messages'][0]['content']
    assert '"name": "custom"' in row['messages'][0]['content']
    assert '<think>\nReason\n</think>' in row['messages'][2]['content']
    assert parse_calls(row['messages'][2]['content'])==[dict(name='custom',arguments={'n':7})]
    raw['messages'][3]['tool_call_id']='orphan'
    with pytest.raises(ValueError,match='orphan'):normalize(raw)


@pytest.mark.parametrize('text',[
    '<tool_call>{"name":"search","arguments":{}}</tool_call>',
    '<tool_calls_begin><tool_call>{broken}</tool_call></tool_calls_end>',
    '<tool_calls_begin><tool_call>{"name":"search","arguments":[]}</tool_call></tool_calls_end>',
    '<tool_calls_begin><tool_call>',
])
def test_invalid_calls_fail_closed(text):
    with pytest.raises(ValueError):parse_calls(text)


def test_missing_observation_or_final_answer_rejected():
    raw=polar();del raw['trajectory'][3]
    with pytest.raises(ValueError,match='observation'):normalize(raw)
    raw=polar();raw['trajectory'].pop()
    with pytest.raises(ValueError,match='final'):normalize(raw)


def test_full_replay_and_no_oracle_response_on_miss():
    row=normalize(polar());tok=Tokenizer()
    outputs=iter([m['content'] for m in row['messages'] if m['role']=='assistant'])
    def generate(encoded,limit):return next(outputs),True,1
    result=run_agent(row['messages'],tok,generate,ReplayTools(row['messages']))
    assert result['status']=='completed' and result['prediction']=='Alice'
    replay=ReplayTools(row['messages'])
    with pytest.raises(ToolError,match='replay_miss'):
        replay.execute([dict(name='search',arguments={'query':['different']})])
    assert replay.index==0


def test_registry_batches_and_unknown_tool():
    seen=[]
    def search(query):seen.extend(query);return 'search result'
    registry=Registry(dict(search=search,visit=lambda url,goal:goal))
    replies=registry.execute([dict(name='search',arguments={'query':['a','b']}),
                              dict(name='visit',arguments={'url':['x'],'goal':'g'})])
    assert seen==['a','b'] and 'g' in replies[1]['content']
    with pytest.raises(ToolError,match='unknown_tool'):registry.execute([dict(name='missing',arguments={})])


def test_length_guard_does_not_truncate(tmp_path):
    path=tmp_path/'data.jsonl';path.write_text(json.dumps(normalize(polar()))+'\n')
    ds=OpenSeekerDataset(path,Tokenizer(),max_length=10)
    with pytest.raises(ValueError,match='exceeds'):ds[0]


def test_generation_and_context_limits_do_not_score_partial_answer():
    row=normalize(polar());tok=Tokenizer()
    result=run_agent(row['messages'],tok,lambda e,n:('<answer>Alice</answer>',False,n),Registry({}))
    assert result['status']=='generation_limit' and result['prediction']==''
    result=run_agent(row['messages'],tok,lambda e,n:pytest.fail('must not generate'),Registry({}),max_context=1)
    assert result['status']=='context_limit'


@pytest.mark.parametrize('family',['qwen3','qwen35'])
def test_real_tiny_beacon_forward_with_tool_regions(tmp_path,family):
    import torch
    from test_qwen3_whole_information import tiny_qwen3, whole_config
    from test_beacon_qwen35_linear_memory import _tiny_model
    from search_comp.data.factory import make_dataset
    from search_comp.data.searchr1_dataset import SearchR1Collator
    path=tmp_path/'train.jsonl';path.write_text(json.dumps(normalize(polar()))+'\n')
    model=tiny_qwen3() if family=='qwen3' else _tiny_model()
    model.set_beacon_config(whole_config(model,'append',False))
    # Tiny models have small vocabularies; keep the same boundary lengths.
    class TinyTokenizer(Tokenizer):
        def encode(self,text,add_special_tokens=False):return [3+ord(c)%50 for c in text]
    tok=TinyTokenizer()
    dataset=make_dataset(path,tok,data_mode='openseeker')
    batch=SearchR1Collator(tok)([dataset[0]])
    with torch.no_grad():
        result=model(**batch)
        assert torch.isfinite(result.loss) and result.loss>0
        prefix=encode(normalize(polar())['messages'][:-1],tok,generation_prompt=True)
        generated=model.beacon_generate(input_ids=torch.tensor([prefix['input_ids']]),
            regions=prefix['regions'],max_new_tokens=2,reuse_cache=False)
        assert 1<=generated.shape[1]<=2


def test_structured_parallel_response_order_is_not_silently_changed():
    raw=dict(messages=[dict(role='system',content='s'),dict(role='user',content='q'),
        dict(role='assistant',tool_calls=[
            dict(id='a',function=dict(name='search',arguments={'query':['x']})),
            dict(id='b',function=dict(name='visit',arguments={'url':'u','goal':'g'}))]),
        dict(role='tool',tool_call_id='b',content='page'),
        dict(role='tool',tool_call_id='a',content='result'),dict(role='assistant',content='answer')])
    with pytest.raises(ValueError,match='out-of-order'):normalize(raw)


def test_refilter_correctness_and_stable_question_splits(tmp_path):
    import subprocess,sys
    from pathlib import Path
    parent=tmp_path/'prepared';parent.mkdir()
    sample=normalize(polar());sample['correctness']='Correct'
    sample['lengths']={'mock':{'total':100}}
    incorrect=copy.deepcopy(sample);incorrect.update(id='incorrect',question_id='bad',correctness='Incorrect')
    longer=copy.deepcopy(sample);longer.update(id='long',question_id='long',lengths={'mock':{'total':10000}})
    (parent/'normalized.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in [sample,sample,incorrect,longer]))
    (parent/'manifest.json').write_text(json.dumps(dict(tokenizers={'mock':{}},arguments=dict(
        input=[],seed=42,require_correct=False,test_fraction=.02,val_fraction=.02))))
    root=Path(__file__).resolve().parents[1];out=tmp_path/'filtered'
    subprocess.run([sys.executable,str(root/'scripts/refilter_openseeker.py'),'--prepared',str(parent),
                    '--output',str(out),'--max-length','1000'],check=True,capture_output=True)
    report=json.loads((out/'manifest.json').read_text())
    assert report['counts']['incorrect']==report['counts']['over_length']==report['counts']['duplicate_question']==1
    selected=[json.loads(line) for split in ['train','validation','test'] for line in (out/f'{split}.jsonl').read_text().splitlines()]
    assert len(selected)==1 and selected[0]['id']==sample['id']


def test_corpus_backend_executes_novel_queries_and_real_visits(tmp_path):
    from search_comp.evaluation.tool_agent import CorpusTools
    corpus=tmp_path/'corpus.jsonl'
    corpus.write_text(''.join(json.dumps(dict(id=str(i),url=f'https://example.org/{i}',
        title=f'Document {i}',text=text))+'\n' for i,text in enumerate([
            'Alice won a peace prize.', 'Bob studies astronomy.', 'Carla builds bridges.'])))
    tools=CorpusTools(corpus)
    replies=tools.execute([dict(name='search',arguments={'query':['Who is Alice?','astronomy']})])
    assert 'https://example.org/0' in replies[0]['content']
    replies=tools.execute([dict(name='visit',arguments={'url':['https://example.org/0'],'goal':'prize'})])
    assert 'Alice won a peace prize.' in replies[0]['content']
    with pytest.raises(ToolError,match='not present'):
        tools.execute([dict(name='visit',arguments={'url':'https://missing','goal':'prize'})])
