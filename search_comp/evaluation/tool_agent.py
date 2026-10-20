"""Full-history OpenSeeker agent: same serializer/regions as SFT, arbitrary tools.

Replay is an exact action match diagnostic, never a substitute for online metrics.
A gateway implements POST {name, arguments} -> {content: str} for live tools.
"""
from __future__ import annotations

import json
import re
import urllib.request
from ..data.openseeker import encode, parse_calls, json_text, CALL_END, RESPONSE_RE


class ToolError(RuntimeError):
    pass


class Registry:
    def __init__(self, handlers):
        self.handlers = handlers

    def execute(self, calls):
        responses = []
        for call in calls:
            if call['name'] not in self.handlers:
                raise ToolError('unknown_tool: ' + call['name'])
            try:
                content = self.handlers[call['name']](**call['arguments'])
            except Exception as error:
                raise ToolError(f'tool_execution_error: {error}') from error
            if not isinstance(content, str):
                content = json_text(content)
            responses.append(dict(role='user', content='<tool_response>' + content + '</tool_response>'))
        return responses


class HTTPTools:
    def __init__(self, endpoint, timeout=90):
        self.endpoint, self.timeout = endpoint, timeout

    def execute(self, calls):
        replies = []
        for call in calls:
            request = urllib.request.Request(self.endpoint, data=json_text(call).encode(),
                                            headers={'Content-Type': 'application/json'})
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.load(response)
                content = body['content']
                if not isinstance(content, str):
                    raise ValueError('gateway content must be text')
            except Exception as error:
                raise ToolError(f'gateway_error: {error}') from error
            replies.append(dict(role='user', content='<tool_response>' + content + '</tool_response>'))
        return replies


class ReplayTools:
    """Consume the next recorded batch only if every name and argument matches."""
    def __init__(self, messages):
        self.batches = []
        for i, msg in enumerate(messages):
            if msg['role'] != 'assistant':
                continue
            calls = parse_calls(msg['content'])
            if not calls:
                continue
            responses = []
            for later in messages[i+1:]:
                if later['role'] != 'user':
                    break
                responses.append(later)
            self.batches.append((calls, responses))
        self.index = 0

    def execute(self, calls):
        if self.index >= len(self.batches) or self.batches[self.index][0] != calls:
            raise ToolError('replay_miss: generated action differs from recorded action')
        replies = self.batches[self.index][1]
        self.index += 1
        return [dict(m) for m in replies]


class ModelGenerator:
    def __init__(self, model, tokenizer, beacon=False):
        self.model, self.tokenizer, self.beacon = model, tokenizer, beacon

    def __call__(self, encoded, max_new_tokens):
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList
        tokenizer = self.tokenizer
        ids = torch.tensor([encoded['input_ids']], device=next(self.model.parameters()).device)
        stops = [CALL_END, '<|im_end|>']
        with torch.inference_mode():
            if self.beacon:
                out = self.model.beacon_generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                    regions=encoded['regions'], max_new_tokens=max_new_tokens, tokenizer=tokenizer,
                    stop_texts=stops, reuse_cache=False)
                generated = out[0].tolist()
            else:
                start = ids.shape[1]
                class Stop(StoppingCriteria):
                    def __call__(self, input_ids, scores, **kwargs):
                        text = tokenizer.decode(input_ids[0, start:].tolist(), skip_special_tokens=False)
                        return any(s in text for s in stops)
                out = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                    max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    stopping_criteria=StoppingCriteriaList([Stop()]))
                generated = out[0, start:].tolist()
        text = tokenizer.decode(generated, skip_special_tokens=False)
        ended = any(s in text for s in stops)
        # ChatML assistant boundary is re-serialized by the shared encoder.
        text = text.split('<|im_end|>', 1)[0]
        return text, ended, len(generated)


def final_answer(text):
    matches = re.findall(r'<answer>(.*?)</answer>', text, re.S)
    if matches:
        return matches[-1].strip()
    return re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()


def run_agent(messages, tokenizer, generate, tools, max_turns=32,
              max_context=16384, max_new_tokens=8192, max_tool_tokens=8192):
    history = [dict(m) for m in messages[:2]]
    events = []; status = 'turn_limit'; prediction = ''; generated_tokens = 0
    for turn in range(max_turns):
        encoded = encode(history, tokenizer, generation_prompt=True)
        available = max_context - len(encoded['input_ids'])
        if available <= 0:
            status = 'context_limit'; break
        text, ended, n_tokens = generate(encoded, min(max_new_tokens, available))
        generated_tokens += n_tokens
        history.append(dict(role='assistant', content=text))
        if not ended:
            status = 'generation_limit'; break
        try:
            calls = parse_calls(text)
        except ValueError as error:
            status = 'invalid_call'; events.append(dict(error=str(error))); break
        if not calls:
            prediction = final_answer(text)
            status = 'completed' if prediction else 'empty_answer'; break
        try:
            replies = tools.execute(calls)
        except ToolError as error:
            status = 'tool_error'; events.append(dict(calls=calls, error=str(error))); break
        lengths = [len(tokenizer.encode(m['content'], add_special_tokens=False)) for m in replies]
        # Do not silently cut observations: filtering during training also preserves bodies.
        if any(n > max_tool_tokens for n in lengths):
            status = 'tool_budget'; events.append(dict(calls=calls, tokens=lengths)); break
        history.extend(replies)
        events.append(dict(calls=calls, observation_tokens=lengths))
    return dict(prediction=prediction, status=status, messages=history, events=events,
                generated_tokens=generated_tokens, turns=len(events))


class CorpusTools(Registry):
    """Executable offline search/visit over an independent URL-addressed corpus.

    Unlike replay, new queries are supported. This is a local environment,
    not a claim of reproducing a live web provider or goal-conditioned summarizer.
    """
    def __init__(self, corpus_path, topk=10):
        from urllib.parse import quote
        from ..data.retrieval import BM25Retriever
        self.retriever = BM25Retriever(str(corpus_path), tokenize_fn=lambda text:
            re.findall(r'[a-z0-9]+|[\u4e00-\u9fff]', text.lower()))
        self.topk = topk
        self.urls = {}
        self.documents = {}
        for doc in self.retriever.docs:
            url = doc.get('url') or 'corpus://' + quote(str(doc['id']), safe='')
            if url in self.documents:
                raise ValueError('duplicate corpus URL: ' + url)
            self.urls[doc['id']] = url
            self.documents[url] = doc
        super().__init__({'search': self.search, 'visit': self.visit})

    def search(self, query):
        if not isinstance(query, list) or not query or any(not isinstance(q, str) for q in query):
            raise ValueError('search.query must be a nonempty string array')
        batches = []
        for q in query:
            docs = self.retriever.retrieve(q, topk=self.topk)
            batches.append(f'### A local search for {q!r} found {len(docs)} results:\n\n' + '\n\n'.join(
                f"{i}. [{doc['title']}]({self.urls[doc['id']]})\n{doc['text'][:600]}"
                for i, doc in enumerate(docs, 1)))
        return '\n---\n'.join(batches)

    def visit(self, url, goal):
        urls = [url] if isinstance(url, str) else url
        if not isinstance(urls, list) or not urls or any(not isinstance(u, str) for u in urls):
            raise ValueError('visit.url must be a string or nonempty string array')
        if not isinstance(goal, str):
            raise ValueError('visit.goal must be text')
        if any(u not in self.documents for u in urls):
            raise ValueError('URL is not present in the independent corpus')
        return '\n\n'.join(f"URL: {u}\nGoal: {goal}\n{self.documents[u]['text']}" for u in urls)
