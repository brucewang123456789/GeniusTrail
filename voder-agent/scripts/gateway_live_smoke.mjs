#!/usr/bin/env node
/** Operator-only acceptance probe: real native VGR + pinned Modal sandbox; optional paid model. */
import { mkdtemp, mkdir, writeFile, chmod, readFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { createHash, randomUUID } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { VgrBroker } from '../packages/voder-agent-core/src/broker.mjs';
import { TrustedVgrGatewayTransport, GatewayCommitVerifier } from '../packages/voder-native-bridge/src/trusted-gateway.mjs';
import { VoderSDK } from '../packages/voder-sdk/src/index.mjs';

const agentRoot=resolve(process.argv[2]??'/root/autodl-tmp/voder-agent');
const mode=process.argv[3]??'--native';
if(!['--native','--model'].includes(mode))throw Error('Use --native or --model');
if(process.env.VODER_ALLOW_MODAL_CHARGES!=='YES')throw Error('EXPLICIT_MODAL_CONSENT_REQUIRED: VODER_ALLOW_MODAL_CHARGES=YES');
if(mode==='--model' && process.env.VODER_ALLOW_MODEL_API!=='YES')throw Error('EXPLICIT_MODEL_CONSENT_REQUIRED: VODER_ALLOW_MODEL_API=YES');
const cli=process.env.VODER_MODAL_CLI;
if(!cli?.startsWith('/'))throw Error('VODER_MODAL_CLI is not an absolute path');
const realRunRoot=join(agentRoot,'evidence','trusted-gateway');
await mkdir(realRunRoot,{recursive:true});
const temp=await mkdtemp(join(realRunRoot,'accept-'));
const repo=join(temp,'repo');const target='src/input.txt';
await mkdir(join(repo,'src'),{recursive:true});
const before='VODER_GATEWAY_BEFORE\n';const after='VODER_GATEWAY_AFTER\n';
await writeFile(join(repo,target),before);
const git=(...args)=>execFileSync('/usr/bin/git',args,{cwd:repo,stdio:'pipe',env:{...process.env,GIT_CONFIG_NOSYSTEM:'1',GIT_CONFIG_GLOBAL:'/dev/null'}}).toString().trim();
git('init','-q');git('add',target);
git('-c','user.name=Voder Gateway Acceptance','-c','user.email=acceptance@example.invalid','commit','-qm','baseline');
const program='/usr/bin/grep';const args=['-Fx','VODER_GATEWAY_AFTER',target];
const debugArgs=`[${args.map(x=>JSON.stringify(x)).join(', ')}]`;
const sha=data=>createHash('sha256').update(data).digest('hex');
const policy={schema:'voder.gateway.policy.v1',repository_path:repo,target_path:target,
 validator_program:program,validator_args:args,validator_sha256:sha(`${program}\0v1\0${debugArgs}`)};
const policyFile=join(temp,'operator-policy.json');
await writeFile(policyFile,JSON.stringify(policy,null,2)+'\n',{mode:0o600});await chmod(policyFile,0o600);
const attemptStore=join(temp,'attempt-store');await mkdir(join(attemptStore,'attempts'),{recursive:true});
const nativeGateway={binary:join(agentRoot,'..','.voder-build','release','voder-gateway'),
  daemonBinary:join(agentRoot,'..','.voder-build','release','voderd'),
  policyFile,attemptStore,nodeBinary:process.execPath,modalCli:cli,targetPath:target};
console.log(`VODER_GATEWAY_EVIDENCE_ROOT=${temp}`);
console.log('ORIGINAL_REPOSITORY_PROTECTED=TRUE');
let exitStatus=1;
try{
 if(mode==='--native'){
   const native=new TrustedVgrGatewayTransport(nativeGateway);
   const verifier=new GatewayCommitVerifier(nativeGateway);
   const sessionId=`sess-${randomUUID()}`;
   const broker=new VgrBroker({native,sessionId,workspaceRoot:repo,commitVerifier:verifier});
   const result=await broker.dispatch({schema:'voder.agent.action.v1',session_id:sessionId,
     action_id:`action-${randomUUID()}`,idempotency_key:`idem-${randomUUID()}`,
     kind:'patch',relative_path:target,old_sha256:sha(before),new_content:after,
     workspace_scope:[target],evidence_obligations:['gateway-requirement'],budget:{max_runtime_ms:120000}});
   if(result.status!=='COMMITTED')throw Error('Authoritative journal did not commit');
   console.log('RUST_VGR_AUTHORIZED_PATCH_AND_FINAL_COMMIT=PASS');
   console.log('INDEPENDENT_DAEMON_COMMIT_VERIFICATION=PASS');
 } else {
   const sdk=new VoderSDK({cwd:repo,nativeGateway});
   const {sessionId}=sdk.createSession();
   try {
     const task={goal:`Use ONLY voder_authorized_action to replace the complete contents of ${target} with exactly ${JSON.stringify(after)}. The original content is ${JSON.stringify(before)}. Must provide kind=patch, relative_path, old_sha256, new_content; do not change any other file.`,maxActions:2,requiredFiles:[target]};
     const result=await sdk.submitTask(sessionId,task);
     if(result.status!=='VERIFIED')throw Error(`MODEL_VGR_FINAL_STATUS=${result.status}`);
     console.log('REAL_MODEL_TO_NATIVE_VGR_FINAL_COMMIT=PASS');
   } finally {sdk.dispose()}
 }
 if((await readFile(join(repo,target),'utf8'))!==before)throw Error('original repo unexpectedly modified');
 console.log('ORIGINAL_REPOSITORY_UNCHANGED=PASS');
 console.log('PRODUCTION_GATE=NOT_INFERRED_FROM_ONE_TEST');
 exitStatus=0;
} catch(e){
 console.error('VODER_GATEWAY_LIVE_GATE=FAIL');
 console.error(`ERROR_CODE=${e?.code??'ERROR'}`);
 console.error(String(e?.message??e));
 console.error('AMBIGUOUS_OUTCOME_REQUIRES_OPERATOR_RECONCILIATION=TRUE');
 console.error('EVIDENCE_RETAINED=TRUE');
} finally {process.exitCode=exitStatus}
