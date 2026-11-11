/** Continue only an observed output truncation in the SAME live agent, never replay a task or tool. */
import {createHash} from 'node:crypto';
import {VoderError} from './contracts.mjs';
const sha=s=>createHash('sha256').update(s).digest('hex');
export function installOutputRecovery(agent,{check=()=>{},onEvent=()=>{},mayContinue=()=>true,getProgress}={}){
 if(!agent||typeof agent.steer!=='function')throw new VoderError('INCOMPATIBLE_UPSTREAM','Agent steering hook required for safe output recovery');
 const previous=agent.finishTurn;let signature=null,recoveries=0,lastProgress;
 const stalled=()=>{onEvent({type:'output_recovery',code:'MODEL_TRUNCATION_STALLED',recovery_count:recoveries});throw new VoderError('MODEL_TRUNCATION_STALLED','Output truncated again without new completed tool evidence; inspect capacity or compact explicitly before continuing');};
 const hook=async function(turn,signal){
  const decision=await previous?.call(agent,turn,signal);
  if(decision?.action==='end'||signal?.aborted)return decision;
  if(turn?.message?.stopReason!=='length'){signature=null;return decision;}
  if(!mayContinue()){onEvent({type:'output_recovery',code:'OUTPUT_TRUNCATED_AFTER_COMMIT_NO_REPLAY',recovery_count:recoveries});return {action:'end'};}
  check();
  const next=sha(JSON.stringify((turn.message.content??[]).map(b=>b.type==='toolCall'?{type:b.type,name:b.name,arguments:b.arguments??null}:b)));
  const progress=getProgress?.();
  if(next===signature||(typeof getProgress==='function'&&lastProgress!==undefined&&progress===lastProgress))stalled();
  signature=next;lastProgress=progress;recoveries++;
  onEvent({type:'output_recovery',code:'MODEL_OUTPUT_TRUNCATED',recovery_count:recoveries,tool_call_count:turn.message.content?.filter(b=>b.type==='toolCall').length??0});
  agent.steer({role:'user',content:[{type:'text',text:'Host observed an output-token truncation. Continue this same task from existing evidence; the previous partial response is NOT completion. Any tool calls in that truncated response were rejected, so re-issue necessary calls with complete arguments. Prefer the exact compact edits form of voder_verify_candidate for small changes; avoid resending whole files or repeating analysis. Do not repeat any already committed native action. If provider/context capacity prevents a complete step, report that blocker explicitly.'}],timestamp:Date.now()});
  return {action:'continue'};
 };
 agent.finishTurn=hook;
 return Object.freeze({inspect(){if(agent.finishTurn!==hook)throw new VoderError('OUTPUT_RECOVERY_CHANGED','Output recovery hook was replaced');},count:()=>recoveries});
}
