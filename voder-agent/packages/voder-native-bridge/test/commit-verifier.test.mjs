import test from 'node:test';
import assert from 'node:assert/strict';
import { NativeVgrCommitVerifier } from '../src/commit-verifier.mjs';
const d='a'.repeat(64);
const input={session_id:'s-123',action_id:'action-1'};
const receipt={...input,run_id:'run-1',commit_digest:d,status:'COMMITTED'};
function make(overrides={}){
 let closed=0;
 const stub={
  async start(){return this},
  async resume(id){return {session_id:id,status:'READY'}},
  async state(){return {session_id:'s-123',runs:[{run_id:'run-1',status:'VGR_COMMITTED',delivery_allowed:true,final_commit_digest:d}]}},
  async replay(){return {events:[{sequence:12,event:{type:'vgr.final_committed',session:'s-123',run_id:'run-1',action_id:'action-1',commit_digest:d}}]}},
  close(){closed++}, ...overrides
 };
 return {verifier:new NativeVgrCommitVerifier({observerFactory:()=>stub}),closed:()=>closed};
}
test('independently confirms durable daemon state and final journal action event',async()=>{
 const v=make();const x=await v.verifier.confirmCommit(receipt,input);assert.equal(x.authority,'VODER_NATIVE_VGR_STATE');assert.equal(v.closed(),1);
});
test('rejects fabricated COMMITTED when native state is only observed',async()=>{
 const v=make({async state(){return {session_id:'s-123',runs:[{run_id:'run-1',status:'VGR_OBSERVED',delivery_allowed:false,final_commit_digest:null}]}}});
 await assert.rejects(v.verifier.confirmCommit(receipt,input),{code:'NO_AUTHORITATIVE_COMMIT'});assert.equal(v.closed(),1);
});
test('rejects wrong action even when native state says COMMITTED',async()=>{
 const v=make({async replay(){return {events:[{sequence:12,event:{type:'vgr.final_committed',session:'s-123',run_id:'run-1',action_id:'action-OTHER',commit_digest:d}}]}}});
 await assert.rejects(v.verifier.confirmCommit(receipt,input),{code:'COMMIT_EVENT_MISMATCH'});
});
test('rejects claimed commit when final event missing',async()=>{
 const v=make({async replay(){return {events:[]}}});
 await assert.rejects(v.verifier.confirmCommit(receipt,input),{code:'COMMIT_EVENT_MISSING'});
});
test('transport failure does not get converted into a PASS',async()=>{
 const v=make({async start(){throw Error('LOCKED')}});
 await assert.rejects(v.verifier.confirmCommit(receipt,input),{code:'NATIVE_OBSERVATION_FAILED'});assert.equal(v.closed(),1);
});
test('invalid receipt identity short circuits without querying daemon',async()=>{
 const v=make({async start(){throw Error('should not reach daemon')}});
 await assert.rejects(v.verifier.confirmCommit({...receipt,action_id:'intruder'},input),{code:'IDENTITY_MISMATCH'});assert.equal(v.closed(),0);
});
test('duplicate commit events fail closed',async()=>{
 const e={sequence:1,event:{type:'vgr.final_committed',session:'s-123',run_id:'run-1',action_id:'action-1',commit_digest:d}};
 const v=make({async replay(){return {events:[e,{...e,sequence:2}]}}});
 await assert.rejects(v.verifier.confirmCommit(receipt,input),{code:'COMMIT_EVENT_MISSING'});
});
