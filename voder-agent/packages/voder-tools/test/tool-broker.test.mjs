import {test} from 'node:test';
import assert from 'node:assert/strict';
import {HostToolRegistry,ToolEconomy,rankToolCapabilities,isHostApprovedReadTool} from '../src/index.mjs';
const S={type:'object',properties:{query:{type:'string',maxLength:80}},required:['query'],additionalProperties:false};
const make=({execute=async ({query})=>({rows:[query]}),...opts}={})=>new HostToolRegistry().register({name:'voder_ext_index',description:'Query approved index',parameters:S,execute,...opts});
test('host-installed tool is dynamically approved and shaped with closed schema',async()=>{
 const tool=make().makeTools()[0];assert.equal(isHostApprovedReadTool(tool),true);
 assert.equal(isHostApprovedReadTool({...tool}),false);
 assert.equal((await tool.execute({query:'needle'})).trust,'UNTRUSTED_PLUGIN_DATA');
 await assert.rejects(tool.execute({query:'x',command:'bash'}),{code:'INVALID_TOOL_ARGS'});
 await assert.rejects(tool.execute({query:'x'.repeat(81)}),{code:'INVALID_TOOL_ARGS'});
});
test('named permissions cannot inject bash, arbitrary writes or extra tools',()=>{
 assert.throws(()=>new HostToolRegistry().register({name:'bash',description:'command',parameters:S,execute:()=>{}}),{code:'INVALID_TOOL_REGISTRATION'});
 assert.throws(()=>make({capability:'write'}),{code:'INVALID_TOOL_REGISTRATION'});
 assert.throws(()=>make({risk:10}),{code:'INVALID_TOOL_REGISTRATION'});
 assert.throws(()=>make({execute:'model-provided instruction'}),{code:'INVALID_TOOL_REGISTRATION'});
 assert.throws(()=>make({parameters:{type:'object',properties:{query:{type:'object'}},required:[],additionalProperties:true}}),{code:'INVALID_TOOL_SCHEMA'});
});
test('economic governor prevents duplicate expensive calls, including successful ones',async()=>{
 let calls=0;const economy=new ToolEconomy();const tool=make({execute:async ({query})=>{calls++;return {query};}}).makeTools({economy})[0];
 await tool.execute({query:'a'});
 await assert.rejects(tool.execute({query:'a'}),{code:'DUPLICATE_NO_PROGRESS'});
 await tool.execute({query:'b'});assert.equal(calls,2);assert.equal(economy.snapshot().calls,2);
});
test('failed identical tool cannot retry and other strategy remains available',async()=>{
 let count=0;const econ=new ToolEconomy({maxFailuresPerTool:2});const tool=make({execute:async()=>{count++;throw Error('blocked');}}).makeTools({economy:econ})[0];
 await assert.rejects(tool.execute({query:'a'}),/blocked/);
 await assert.rejects(tool.execute({query:'a'}),{code:'DUPLICATE_NO_PROGRESS'});
 await assert.rejects(tool.execute({query:'b'}),/blocked/);
 await assert.rejects(tool.execute({query:'c'}),{code:'TOOL_CIRCUIT_OPEN'});
 assert.equal(count,2);assert.equal(econ.snapshot().failed_calls,2);
});
test('huge receipts fail without truncation and are treated as failures',async()=>{
 const econ=new ToolEconomy({maxSingleReturnBytes:1024});const tool=make({execute:async()=>({secret:'x'.repeat(2048)})}).makeTools({economy:econ})[0];
 await assert.rejects(tool.execute({query:'a'}),{code:'TOOL_RESULT_OVER_BUDGET'});
 assert.equal(econ.snapshot().return_bytes,0);
});
test('tools cannot be called after task epoch invalidation',async()=>{
 const tool=make().makeTools({stillCurrent(){throw Object.assign(Error('stale'),{code:'STALE_TASK'})}})[0];
 await assert.rejects(tool.execute({query:'a'}),{code:'STALE_TASK'});
});
test('tool scoring orders evidence gains against latency, token and risk deterministically',()=>{
 const list=rankToolCapabilities([
   {name:'voder_ext_fast',capability:'read',evidenceValue:3,estimatedTokens:100,estimatedMs:100,risk:0},
   {name:'voder_ext_slow',capability:'read',evidenceValue:3,estimatedTokens:2000,estimatedMs:5000,risk:0}
 ],{purpose:'inspect'});assert.equal(list[0].name,'voder_ext_fast');
});
test('stalled read-only adapter gets a timed failure without retrying',async()=>{
 const economy=new ToolEconomy();const tool=make({estimatedMs:1,execute:async (_args,signal)=>{
  await new Promise((resolve)=>signal.addEventListener('abort',resolve,{once:true}));
  return {status:'cancelled'};
 }}).makeTools({economy})[0];
 await assert.rejects(tool.execute({query:'stall'}),{code:'TOOL_TIMEOUT'});
 await assert.rejects(tool.execute({query:'stall'}),{code:'DUPLICATE_NO_PROGRESS'});
});
