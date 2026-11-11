import {test} from 'node:test';
import assert from 'node:assert/strict';
import {TaskPlan,rankNextActions} from '../src/index.mjs';

test('lightweight tasks avoid a mandatory planning waterfall',()=>{
 const p=new TaskPlan({goal:'Fix a typo',maxActions:4});
 assert.equal(p.brief().mode,'DIRECT');assert.equal(p.stages.length,4);
 assert.equal(p.snapshot().native_actions,0);
});
test('repository scale work gets bounded plan without extra model call',()=>{
 const p=new TaskPlan({goal:'Fix a multi-file runtime bug while maintaining interface',maxActions:3});
 assert.equal(p.brief().mode,'ON_DEMAND_PLAN');
 assert.equal(p.stages.length,5);
 const initial=p.decision();assert.equal(initial.recommended,'INSPECT');
 const fail=p.onEvent({kind:'sandbox',status:'FAIL',reason:'SANDBOX_ASSERTION'});
 assert.equal(fail.recommended,'DIAGNOSE');
 const repaired=p.onEvent({kind:'sandbox',status:'PASS',progress:true,reason:'SANDBOX_PASS'});
 assert.equal(repaired.recommended,'PATCH');
 p.onEvent({kind:'native',status:'COMMITTED',progress:true,reason:'NATIVE_COMMITTED'});
 assert.equal(p.snapshot().native_actions,1);
});
test('finite budget and blocked state force STOP',()=>{
 const p=new TaskPlan({goal:'fix bug',maxActions:1});
 p.onEvent({kind:'native',status:'DENIED',reason:'NATIVE_DENIED'});
 assert.equal(p.decision().recommended,'STOP');
 assert.equal(p.snapshot().blocked,false);
 const q=new TaskPlan({goal:'fix bug'});q.onEvent({kind:'sandbox',status:'FAIL',blocked:true});
 assert.equal(q.decision().recommended,'STOP');
});
test('ranking is deterministic and respects evidence economics',()=>{
 const a=rankNextActions({stage:'failure'}),b=rankNextActions({stage:'failure'});
 assert.deepEqual(a,b);assert.equal(a[0].action,'DIAGNOSE');
 assert.deepEqual(rankNextActions({stage:'initial',budget:0}).map(x=>x.action),['STOP']);
 assert.throws(()=>rankNextActions({budget:-1}),TypeError);
});
test('long task text kept as hash only, never sliced or elevated',()=>{
 const goal='验证约束'.repeat(14000),p=new TaskPlan({goal,requiredFiles:['a','b','c']});
 assert.match(p.brief().goal_digest,/^[0-9a-f]{64}$/);assert.equal(p.brief().mode,'ON_DEMAND_PLAN');
 assert.ok(!JSON.stringify(p.snapshot()).includes('验证约束'));
});
