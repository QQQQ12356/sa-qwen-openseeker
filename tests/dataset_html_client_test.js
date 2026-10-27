// Exercise the standalone report with a minimal DOM and no network/browser deps.
const fs = require('fs'), vm = require('vm'), assert = require('assert');
class Element {
  constructor(tag) {this.tag=tag;this.children=[];this._text='';this.className='';this.value='';this.handlers={};}
  set textContent(value) {this._text=String(value);this.children=[];}
  get textContent() {return this._text+this.children.map(n=>n.textContent).join('');}
  append(...nodes) {this.children.push(...nodes);}
  replaceChildren(...nodes) {this._text='';this.children=nodes;}
  addEventListener(event,handler) {this.handlers[event]=handler;}
}
function tree(node) {return [node,...node.children.flatMap(tree)];}
const markup=fs.readFileSync(process.argv[2],'utf8');
const scripts=[...markup.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)].map(m=>m[1]);
const ids={};
for(const id of ['dataset-data','title','source','manifest','manifestText','stats','search','observation','role','count','list','page','pagePrev','pageNext','prev','next','expand','collapse','position','detail'])ids[id]=new Element(id);
ids['dataset-data'].textContent=scripts[0];ids.role.value=ids.observation.value='all';
const document={getElementById:id=>ids[id],createElement:tag=>new Element(tag),
  querySelectorAll:selector=>tree(ids.detail).filter(n=>selector==='.'+n.className)};
const context=vm.createContext({document,setTimeout,clearTimeout});
vm.runInContext(scripts[1],context);
const evaluate=code=>vm.runInContext(code,context);
assert.equal(ids.list.children.length,2);
assert.ok(ids.detail.textContent.includes('中文问题 </script><script>alert(1)</script>'));
assert.equal(tree(ids.detail).filter(n=>n.tag==='script').length,0);
assert.equal(tree(ids.detail).filter(n=>n.className==='part observation').length,1);
assert.equal(tree(ids.detail).filter(n=>n.className==='part call').length,1);
assert.ok(ids.detail.textContent.includes('<answer>quoted text</answer>'));
const messages=tree(ids.detail).filter(n=>n.className==='message');
assert.equal(messages[0].open,false);
ids.collapse.onclick();assert.ok(messages.every(n=>!n.open));
ids.expand.onclick();assert.ok(messages.every(n=>n.open));
ids.next.onclick();assert.ok(ids.detail.textContent.includes('Second question'));
ids.prev.onclick();assert.ok(ids.detail.textContent.includes('中文问题'));
ids.role.value='system';ids.role.onchange();
assert.equal(tree(ids.detail).filter(n=>n.className==='part observation').length,0);
ids.observation.value='zero';ids.observation.onchange();assert.equal(ids.list.children.length,1);
assert.ok(ids.detail.textContent.includes('Second question'));
ids.search.value='no-such-value';evaluate('applyFilters()');
assert.equal(ids.list.children.length,0);assert.ok(ids.detail.textContent.includes('没有匹配样本'));
console.log('PASS: rendering, safe text, role boundaries, observation filter, search, navigation and folding');
