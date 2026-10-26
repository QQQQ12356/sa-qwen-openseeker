"""OpenSeeker wire protocol shared by preprocessing, SFT and agent inference.

Keep original Polar text and append a concise gold final_answer target;
render structured OpenAI calls into the same protocol.
Only observation bodies are compressible. No trajectory is silently truncated.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

CALL_RE = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.S)
RESPONSE_RE = re.compile(r'<tool_response>(.*?)</tool_response>', re.S)
CALL_BEGIN = '<tool_calls_begin>'
CALL_END = '</tool_calls_end>'
PROTOCOL = '''You may call the supplied tools. Emit one or more JSON calls as:
<tool_calls_begin>
<tool_call>{"name": "tool_name", "arguments": {}}</tool_call>
</tool_calls_end>
Tool observations arrive inside <tool_response>...</tool_response>. After reasoning,
return the final answer without further tool calls.'''
LEGACY_VERSION = 'openseeker-chatml-v1'
VERSION = 'openseeker-chatml-v2-final-answer'
FINAL_ANSWER_RE = re.compile(r'<final_answer>(.*?)</final_answer>', re.S)
FINAL_ANSWER_INSTRUCTION = '''# Final answer format
Keep reasoning and explanations outside <final_answer>...</final_answer>.
After reasoning and any tool calls, end your response with exactly one
<final_answer>...</final_answer> block containing only the concise answer to the
question. Do not include explanations, introductory phrases, or Markdown formatting
inside this block. For example: <final_answer>Beijing</final_answer>.
Do not make any further tool calls after the final answer.'''


def answer_prompt(messages):
    """Add the same answer instruction in preprocessing and inference, without gold."""
    result = copy.deepcopy(messages)
    if not result or result[0]['role'] != 'system':
        raise ValueError('expected system message')
    if FINAL_ANSWER_INSTRUCTION not in result[0]['content']:
        result[0]['content'] += '\n\n' + FINAL_ANSWER_INSTRUCTION
    return result


def extract_final_answer(text):
    """Require one closed, nonempty answer block; never fall back to explanation."""
    matches = FINAL_ANSWER_RE.findall(text)
    if (len(matches) != 1 or text.count('<final_answer>') != 1
            or text.count('</final_answer>') != 1):
        return ''
    return matches[0].strip()


def with_final_answer(row):
    """Upgrade a trajectory without changing its original reasoning or tool actions."""
    answer = row.get('answer')
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError('final_answer requires a nonempty string reference answer')
    answer = answer.strip()
    if '<final_answer' in answer or '</final_answer' in answer:
        raise ValueError('reference answer contains reserved final_answer tags')
    result = copy.deepcopy(row)
    result['messages'] = answer_prompt(result['messages'])
    final = result['messages'][-1]
    content = FINAL_ANSWER_RE.sub('', final['content']).rstrip()
    if '<final_answer' in content or '</final_answer' in content:
        raise ValueError('malformed final_answer in training trajectory')
    final['content'] = content + '\n\n<final_answer>' + answer + '</final_answer>'
    result['protocol'] = VERSION
    # Old measurements are invalid after adding prompt and supervised answer tokens.
    result.pop('lengths', None)
    return result


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def parse_calls(text):
    """Reject incomplete or malformed action blocks instead of scoring them as answers."""
    found = CALL_RE.findall(text)
    if not found:
        if any(tag in text for tag in ('<tool_call', '</tool_call')):
            raise ValueError('incomplete tool call')
        return []
    if text.count(CALL_BEGIN) != 1 or text.count(CALL_END) != 1:
        raise ValueError('missing/duplicate tool-call envelope')
    block = text.split(CALL_BEGIN, 1)[1].split(CALL_END, 1)[0]
    if CALL_RE.sub('', block).strip() or len(CALL_RE.findall(block)) != len(found):
        raise ValueError('invalid tool-call envelope')
    calls = []
    for raw in found:
        call = json.loads(raw)
        if not isinstance(call, dict) or not isinstance(call.get('name'), str):
            raise ValueError('invalid tool name')
        args = call.get('arguments')
        if isinstance(args, str):
            args = json.loads(args)
        if not isinstance(args, dict):
            raise ValueError('tool arguments must be an object')
        calls.append(dict(name=call['name'], arguments=args))
    return calls


def question_key(question):
    return hashlib.sha256(' '.join(question.split()).casefold().encode()).hexdigest()


def normalize(row, source='unknown', index=0):
    if row.get('protocol') == VERSION:
        validate(row)
        return row
    if row.get('protocol') == LEGACY_VERSION:
        validate(row)
        result = with_final_answer(row)
        validate(result)
        return result
    if row.get('protocol'):
        raise ValueError('unsupported OpenSeeker protocol: ' + str(row['protocol']))
    messages = copy.deepcopy(row.get('trajectory', row.get('messages')))
    if isinstance(messages, str):
        messages = json.loads(messages)
    if not isinstance(messages, list) or not messages:
        raise ValueError('missing messages/trajectory')
    tools = row.get('tools') or []
    if isinstance(tools, str):
        tools = json.loads(tools)
    if not isinstance(tools, list) or any(not isinstance(t, dict) for t in tools):
        raise ValueError('tools must be a list of schemas')
    out = []
    ids = {}
    pending = []
    for msg in messages:
        role = msg['role']
        content = msg.get('content') or ''
        if not isinstance(content, str):
            raise ValueError('non-text content requires an explicit multimodal adapter')
        structured = msg.get('tool_calls') or []
        if isinstance(structured, str):
            structured = json.loads(structured)
        if role == 'assistant' and structured:
            calls = []
            for c in structured:
                fn = c.get('function', c)
                args = fn.get('arguments', {})
                if isinstance(args, str):
                    args = json.loads(args)
                if not isinstance(args, dict):
                    raise ValueError('invalid structured arguments')
                calls.append(dict(name=fn['name'], arguments=args))
                if c.get('id'):
                    if c['id'] in ids:
                        raise ValueError('duplicate tool call id')
                    ids[c['id']] = fn['name']
                    pending.append(c['id'])
            reasoning = msg.get('reasoning_content') or msg.get('reasoning')
            if reasoning:
                content = '<think>\n' + reasoning + '\n</think>\n' + content
            content += '\n' + CALL_BEGIN + '\n' + '\n'.join(
                '<tool_call>' + json_text(c) + '</tool_call>' for c in calls) + '\n' + CALL_END
        elif role == 'assistant':
            reasoning = msg.get('reasoning_content') or msg.get('reasoning')
            if reasoning:
                content = '<think>\n' + reasoning + '\n</think>\n' + content
        if role == 'tool':
            cid = msg.get('tool_call_id')
            if cid:
                if cid not in pending:
                    raise ValueError('orphan/duplicate tool_call_id')
                if cid != pending[0]:
                    raise ValueError('out-of-order tool responses require explicit reordering')
                pending.remove(cid)
            elif pending:
                raise ValueError('missing tool_call_id for structured calls')
            role = 'user'
            if not RESPONSE_RE.fullmatch(content.strip()):
                content = '<tool_response>' + content + '</tool_response>'
        out.append(dict(role=role, content=content))
    if pending:
        raise ValueError('unanswered structured calls')
    if out[0]['role'] != 'system':
        out.insert(0, dict(role='system', content='You are a tool-augmented QA agent.'))
    if tools:
        out[0]['content'] += '\n\n# Tools\n<tools>\n' + '\n'.join(
            json_text(t.get('function', t)) for t in tools) + '\n</tools>\n' + PROTOCOL
    question = row.get('question') or next((m['content'] for m in out if m['role'] == 'user'), '')
    result = dict(id=f'{source}:{index}', protocol=VERSION, messages=out, tools=tools,
                  question=question, question_id=question_key(question), answer=row.get('answer', ''),
                  source=source, source_index=index,
                  correctness=row.get('trajectory correctness'),
                  declared_tool_calls=row.get('number of tool calls', row.get('n_tool_calls')))
    # Validate the original conversation before appending a training target.
    validate(result, require_final_answer=False)
    result = with_final_answer(result)
    validate(result)
    return result


def validate(row, require_final_answer=None):
    messages = row['messages']
    if len(messages) < 3 or [m['role'] for m in messages[:2]] != ['system', 'user']:
        raise ValueError('expected system, question, assistant trajectory')
    pending = False
    observed = False
    for m in messages[2:]:
        if m['role'] == 'assistant':
            if pending and not observed:
                raise ValueError('tool call without observation')
            pending = bool(parse_calls(m['content']))
            observed = False
        elif m['role'] == 'user':
            if not pending or not RESPONSE_RE.search(m['content']):
                raise ValueError('unexpected user message / orphan observation')
            if RESPONSE_RE.sub('', m['content']).strip():
                raise ValueError('text outside observation wrappers')
            observed = True
        else:
            raise ValueError('unexpected role')
    if messages[-1]['role'] != 'assistant' or parse_calls(messages[-1]['content']):
        raise ValueError('incomplete final answer')
    if require_final_answer is None:
        require_final_answer = row.get('protocol') == VERSION
    if require_final_answer:
        if FINAL_ANSWER_INSTRUCTION not in messages[0]['content']:
            raise ValueError('missing final_answer system instruction')
        answer = row.get('answer')
        if (not isinstance(answer, str) or not answer.strip()
                or extract_final_answer(messages[-1]['content']) != answer.strip()
                or not messages[-1]['content'].rstrip().endswith('</final_answer>')):
            raise ValueError('final_answer must end the trajectory and equal the reference answer')


def pieces(messages, generation_prompt=False):
    """(text, supervised, compressible) pieces; boundaries are protocol tokens."""
    for i, msg in enumerate(messages):
        role, text = msg['role'], msg['content']
        yield f'<|im_start|>{role}\n', False, False
        if role == 'user' and i > 1:
            pos = 0
            for match in RESPONSE_RE.finditer(text):
                yield text[pos:match.start(1)], False, False
                yield match.group(1), False, True
                pos = match.end(1)
            yield text[pos:], False, False
        else:
            yield text, role == 'assistant', False
        yield '<|im_end|>\n', role == 'assistant', False
    if generation_prompt:
        yield '<|im_start|>assistant\n', False, False


def encode(messages, tokenizer, generation_prompt=False, loss_segments=None,
           return_loss_segment_ids=False):
    from ..loss_segments import (LossSegmentConfig, SEGMENT_IGNORE, SEGMENT_IDS,
                                 classify_assistant_text)
    loss = LossSegmentConfig.from_dict(loss_segments)
    segmented = return_loss_segment_ids or not loss.is_default
    ids, labels, regions = [], [], []
    segment_ids = []
    for text, supervised, compressible in pieces(messages, generation_prompt):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if compressible and tokens:
            regions.append((len(ids), len(ids) + len(tokens)))
        ids.extend(tokens)
        labels.extend(tokens if supervised else [-100] * len(tokens))
        if segmented:
            if not supervised:
                segment_ids.extend([SEGMENT_IGNORE] * len(tokens))
            elif text == '<|im_end|>\n':
                segment_ids.extend([SEGMENT_IDS['other']] * len(tokens))
            else:
                segment_ids.extend(classify_assistant_text(text, tokenizer, expected_ids=tokens))
    return dict(input_ids=ids, labels=labels, regions=regions,
                question=messages[1]['content'], loss_segment_ids=segment_ids if segmented else None,
                n_turns=sum(m['role'] == 'assistant' for m in messages),
                n_searches=sum(len(parse_calls(m['content'])) for m in messages if m['role'] == 'assistant'))


class OpenSeekerDataset:
    """Lazy JSONL offsets avoid holding multi-GB trajectories in RAM."""
    def __init__(self, data_path, tokenizer, max_length=16384, **kwargs):
        if kwargs.get('doc_region_split'):
            raise ValueError('OpenSeeker supports whole observations only')
        self.loss_segments = kwargs.get('loss_segments')
        self.return_loss_segment_ids = kwargs.get('return_loss_segment_ids', False)
        self.path, self.tokenizer, self.max_length = Path(data_path), tokenizer, max_length
        self.offsets = []
        with self.path.open('rb') as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)
        if not self.offsets:
            raise ValueError('empty dataset')

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        with self.path.open('rb') as f:
            f.seek(self.offsets[index])
            row = json.loads(f.readline())
        if row.get('protocol') != VERSION:
            raise ValueError('OpenSeeker training requires v2 final_answer data; '
                             'run prepare_openseeker.py to regenerate and remeasure the dataset')
        row = normalize(row, index=index)
        validate(row)
        sample = encode(row['messages'], self.tokenizer, loss_segments=self.loss_segments,
                        return_loss_segment_ids=self.return_loss_segment_ids)
        if len(sample['input_ids']) > self.max_length:
            raise ValueError(f"{row['id']}: exceeds max_length={self.max_length}; run prepare first")
        return sample
