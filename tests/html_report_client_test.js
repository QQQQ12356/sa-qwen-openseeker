// Minimal DOM for offline rendering regression checks; no browser dependency.
const fs = require('fs'), vm = require('vm'), assert = require('assert');
class Element {
  constructor(tag) {
    this.tag = tag; this.children = []; this._text = ''; this.className = '';
    this.value = ''; this.dataset = {}; this.handlers = {};
    this.classList = {toggle: (cls, enabled) => {
      const names = new Set(this.className.split(/\s+/).filter(Boolean));
      if (enabled) names.add(cls); else names.delete(cls);
      this.className = [...names].join(' ');
    }};
  }
  set textContent(value) {this._text = String(value); this.children = [];}
  get textContent() {return this._text + this.children.map(n => typeof n === 'string' ? n : n.textContent).join('');}
  append(...nodes) {this.children.push(...nodes);}
  replaceChildren(...nodes) {this._text = ''; this.children = [...nodes];}
  addEventListener(event, handler) {this.handlers[event] = handler;}
  querySelectorAll(selector) {return tree(this).slice(1).filter(n => n.tag === selector);}
}
function tree(node) {return [node, ...node.children.filter(n => typeof n !== 'string').flatMap(tree)];}
function hasClass(node, cls) {return node.className.split(/\s+/).includes(cls);}
function findClass(node, cls) {return tree(node).filter(n => hasClass(n, cls));}
const markup = fs.readFileSync(process.argv[2], 'utf8');
const scripts = [...markup.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)].map(m => m[1]);
const ids = {};
for (const id of ['report-data','title','source','stats','query','status','em','sort','list','detail','count']) ids[id] = new Element(id);
ids['report-data'].textContent = scripts[0]; ids.sort.value = 'original';
const document = {getElementById: id => ids[id], createElement: tag => new Element(tag), createTextNode: text => text};
const context = vm.createContext({document, setTimeout, clearTimeout});
vm.runInContext(scripts[1], context);
const evaluate = source => vm.runInContext(source, context);
const system = evaluate('messageView(rows[0].messages[0], 0)');
assert.equal(system.open, false);
assert.equal(findClass(system, 'system-prompt').length, 1);
assert.equal(findClass(system, 'call').length, 0);
assert.equal(findClass(system, 'final').length, 0);
assert.equal(findClass(system, 'system-prompt')[0].children[1].textContent, evaluate('rows[0].messages[0].content'));
const assistant = evaluate('messageView(rows[0].messages[2], 2)');
assert.equal(findClass(assistant, 'call').length, 2);
assert.equal(findClass(assistant, 'call-title').map(n => n.textContent).join(','), 'search,visit');
assert.ok(findClass(assistant, 'call-arguments')[0].textContent.includes('\n'));
assert.equal(findClass(assistant, 'final').length, 1);
assert.equal(findClass(assistant, 'final')[0].children[1].textContent, 'Alice');
assert.equal(findClass(assistant, 'think')[0].open, false);
assert.ok(assistant.textContent.includes('<final_answer>example inside thought</final_answer>'));
const tool = evaluate('messageView(rows[0].messages[3], 3)');
assert.ok(hasClass(tool, 'tool'));
assert.equal(findClass(tool, 'call').length, 0);
assert.equal(findClass(tool, 'answer').length, 0);
assert.ok(tool.textContent.includes('<answer>literal webpage content</answer>'));
const user = evaluate('messageView({role:"user",content:"Explain <search>example</search>"}, 1)');
assert.equal(findClass(user, 'call').length, 0);
assert.ok(user.textContent.includes('<search>example</search>'));
const broken = evaluate('messageView({role:"assistant",content:"<tool_calls_begin><tool_call>broken"}, 2)');
assert.equal(findClass(broken, 'call').length, 0);
assert.ok(broken.textContent.includes('<tool_calls_begin><tool_call>broken'));
const injection = evaluate('messageView({role:"system",content:"<img src=x onerror=alert(1)>"}, 0)');
assert.equal(tree(injection).filter(n => n.tag === 'img').length, 0);
assert.ok(injection.textContent.includes('<img src=x onerror=alert(1)>'));
const structured = evaluate('messageView({role:"assistant",tool_calls:[{function:{name:"search",arguments:"{\\"query\\":[\\"Alice\\"]}"}}]}, 2)');
assert.equal(findClass(structured, 'call-title')[0].textContent, 'search');
assert.ok(findClass(structured, 'call-arguments')[0].textContent.includes('\n'));
assert.equal(ids.list.children.length, 2);
ids.list.children[1].onclick();
assert.ok(ids.detail.textContent.includes('Second question'));
ids.em.value = 'true'; ids.em.handlers.change(); assert.equal(ids.list.children.length, 1);
ids.em.value = ''; ids.sort.value = 'f1'; ids.sort.handlers.change(); assert.equal(ids.list.children.length, 2);
ids.query.value = 'no-such-question'; ids.sort.handlers.change(); assert.equal(ids.list.children.length, 0);
assert.ok(ids.detail.textContent.includes('没有匹配的记录'));
console.log('PASS: role boundaries, nested examples, batched/structured calls, raw text, folding, filters and navigation');
