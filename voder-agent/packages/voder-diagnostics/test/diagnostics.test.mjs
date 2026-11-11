import {test} from 'node:test';
import assert from 'node:assert/strict';
import {classifyFailure,DiagnosticJournal,ImprovementCandidates} from '../src/index.mjs';
const sha='a'.repeat(64);
test('classifies observed failure categories without inventing cause',()=>{
 assert.equal(classifyFailure('AssertionError: expected 1 actual 2'),'ASSERTION');
 assert.equal(classifyFailure('No module named foo'),'ENVIRONMENT');
 assert.equal(classifyFailure('Timeout exceeded'),'TIMEOUT');
 assert.equal(classifyFailure('Permission denied to access workspace'),'POLICY');
 assert.equal(classifyFailure('unknown output'),'UNKNOWN');
});
test('sandbox failure drives falsifiable diagnostics; original log is not copied into journal',()=>{
 const j=new DiagnosticJournal();const r=j.observeSandbox({status:'FAIL',backend:'modal',diagnostic:'AssertionError: expected 2 actual 1 TOKEN=supersecret'}, {candidateDigest:sha});
 assert.equal(r.source,'REAL_SANDBOX_RECEIPT');assert.equal(r.classification,'ASSERTION');
 assert.equal(r.causal_claim,'HYPOTHESIS_NOT_VERIFIED');assert.equal(r.identical_count,1);
 assert.ok(!JSON.stringify(r).includes('supersecret'));assert.equal(j.snapshot().stalled,false);
});
test('different bytes can continue despite similar error text; exact retries still stop',()=>{
 const j=new DiagnosticJournal({maxIdenticalFailures:3});
 for(let i=0;i<3;i++){
  const r=j.observeSandbox({status:'FAIL',backend:'modal',diagnostic:'test failed at line '+(12+i)},{candidateDigest:String(i).repeat(64)});
  assert.equal(r.identical_count,1);
 }
 assert.equal(j.snapshot().stalled,false);assert.doesNotThrow(()=>j.assertMayProceed());
 for(let i=0;i<3;i++)j.observeSandbox({status:'FAIL',backend:'modal',diagnostic:'test failed at line 99'},{candidateDigest:'f'.repeat(64)});
 assert.equal(j.snapshot().stalled,true);assert.throws(()=>j.assertMayProceed(),{code:'DIAGNOSTIC_STAGNATION'});
});
test('new failure evidence receives a fresh signature, and PASS is not a root-cause claim',()=>{
 const j=new DiagnosticJournal({maxIdenticalFailures:3});
 j.observeSandbox({status:'FAIL',diagnostic:'Timeout exceeded'},{candidateDigest:sha});
 const next=j.observeSandbox({status:'FAIL',diagnostic:'AssertionError: test failed'},{candidateDigest:'b'.repeat(64)});
 assert.equal(next.identical_count,1);
 const pass=j.observeSandbox({status:'PASS',evidence_digest:sha},{candidateDigest:'c'.repeat(64)});
 assert.equal(pass.next_diagnostic,'VERIFY_BINDING_AND_COMMIT');
 assert.equal(j.snapshot().stalled,false);
});
test('RSI improvement proposals cannot be promoted by the model without independent verifier',async()=>{
 const registry=new ImprovementCandidates();const p=registry.propose({scope:'repair-policy',method:'Inspect stack before fixing',methodVersion:'1'});
 assert.equal(p.status,'QUARANTINED');
 const evidence={id:p.id,taskId:'caseA',baselinePass:false,candidatePass:true,baselineCost:5,candidateCost:4,regressionFree:true,receiptDigest:sha};
 await assert.rejects(registry.recordEvaluation(evidence),{code:'INDEPENDENT_EVALUATOR_REQUIRED'});
 assert.equal(registry.assess(p.id).status,'QUARANTINED');
});
test('cross-task independent gated evidence can recommend human review but never mutates running code',async()=>{
 const registry=new ImprovementCandidates({verifyEvaluation:async claim=>claim.receiptDigest===sha});
 const p=registry.propose({scope:'skill',method:'Inspect error first',methodVersion:'1'});
 for(const taskId of ['task-one','task-two']){
  const r=await registry.recordEvaluation({id:p.id,taskId,baselinePass:false,candidatePass:true,baselineCost:100,candidateCost:80,regressionFree:true,receiptDigest:sha});
  assert.equal(r.automatic_promotion,false);assert.equal(r.live_code_modified,false);
 }
 assert.equal(registry.assess(p.id).status,'ELIGIBLE_FOR_HUMAN_REVIEW');
 await assert.rejects(registry.recordEvaluation({id:p.id,taskId:'task-one',baselinePass:true,candidatePass:true,baselineCost:100,candidateCost:99,regressionFree:true,receiptDigest:sha}),TypeError);
 const bad=new ImprovementCandidates({verifyEvaluation:()=>false});const q=bad.propose({scope:'task-planning',method:'fake',methodVersion:'1'});
 await assert.rejects(bad.recordEvaluation({id:q.id,taskId:'test-a',baselinePass:false,candidatePass:true,baselineCost:100,candidateCost:1,regressionFree:true,receiptDigest:sha}),{code:'EVALUATION_NOT_VERIFIED'});
});
test('RSI fails closed when any independent task regresses',async()=>{
 const registry=new ImprovementCandidates({verifyEvaluation:()=>true}),p=registry.propose({scope:'skill',method:'Change tool preference',methodVersion:'2'});
 for(const [taskId,regressionFree] of [['one',true],['two',false]]){
  await registry.recordEvaluation({id:p.id,taskId,baselinePass:false,candidatePass:true,baselineCost:10,candidateCost:5,regressionFree,receiptDigest:sha});
 }
 assert.equal(registry.assess(p.id).status,'QUARANTINED');
});
