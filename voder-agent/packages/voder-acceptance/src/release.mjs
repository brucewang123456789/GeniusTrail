#!/usr/bin/env node
/** Independent release gate: never promotes offline or partial live tests. */
import {resolve,join} from 'node:path';
import {moduleRoot,readJson,privateDir,persist,taskRunId,time} from './common.mjs';
const args=process.argv.slice(2);const flag=n=>{const i=args.indexOf(n);return i<0?null:args[i+1]};
async function report(){
 const contract=await readJson(join(moduleRoot,'policies/release-contract.json'));
 const auditPath=flag('--audit'),livePath=flag('--live');let audit=null,live=null;
 try{if(auditPath)audit=await readJson(resolve(auditPath))}catch{};
 try{if(livePath)live=await readJson(resolve(livePath))}catch{};
 const gates=Object.fromEntries(contract.production_candidate_requires_all.map(id=>[id,{status:'NOT_VERIFIED',reasons:[]}]));
 for(const id of Object.keys(gates)){
  const local=audit?.gates?.[id];if(local?.status&&local.status!=='NOT_VERIFIED')gates[id].reasons.push(`offline_${local.status}`);
  if(id==='A09'||id==='A10'){
   if(['BLOCKED_CAPABILITY','NOT_VERIFIED'].includes(local?.status??''))gates[id].status='BLOCKED_CAPABILITY';
  }
 }
 const blockers=[];
 if(!audit||audit.schema!=='voder.module8.audit.v1'||audit.outcome!=='READY_FOR_LIVE_NOT_PRODUCTION')blockers.push('AUDIT_NOT_CLEAR');
 if(!live||live.schema!=='voder.module8.real.suite.v1'||live.result!=='SCENARIOS_PASS_NOT_PRODUCTION')blockers.push('REAL_SUITE_NOT_COMPLETE');
 if(live?.token_usage==='UNAVAILABLE'||live?.cost_usd==='UNAVAILABLE')blockers.push('A12_METERING_MISSING');
 if(gates.A09.status==='BLOCKED_CAPABILITY'||gates.A10.status==='BLOCKED_CAPABILITY')blockers.push('SDK_VSCODE_DIFF_APPROVAL_MISSING');
 // No self-attested PASS accepted: each gate must have independently measured witness.
 for(const id of contract.production_candidate_requires_all)if(gates[id].status!=='PASS')blockers.push(id+'_NO_INDEPENDENT_LIVE_WITNESS');
 const conclusion={schema:'voder.module8.release-decision.v1',time:time(),candidate:false,status:'BLOCKED',
  rationale:'Release decisions require A01-A12 independently corroborated real witnesses; neither local unit tests nor three real coding scenarios confer product readiness.',
  gates,blockers:[...new Set(blockers)],audit_reference:auditPath,live_reference:livePath};
 const outDir=await privateDir(resolve(process.env.VODER_M8_EVIDENCE_ROOT||join(moduleRoot,'..','..','repair','voder-module8-evidence')));
 const out=join(outDir,'RELEASE_BLOCKED_'+taskRunId()+'.json');await persist(out,conclusion);
 console.log(`VODER_M8_RELEASE=BLOCKED blockers=${conclusion.blockers.length} evidence=${out}`);process.exitCode=4;
}
report().catch(e=>{console.error('VODER_M8_RELEASE=BLOCKED code='+(e.code??'ERROR'));process.exitCode=5});
