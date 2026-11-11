'use strict';
/** Presentation is separate from native authority. No status here grants VGR verification. */
const {writeRequest}=require('./entry-orchestrator.cjs');
function responsePresentation(result,{goal='',readOnly=false,renderFailureCode=null}={}){
 const terminal=result?.status;
 const blocker=result?.task_acceptance?.blocker??result?.taskAcceptance?.blocker;
 if(blocker&&result?.verified!==true)return {kind:'HOST_BLOCKED',title:'Task blocked by host · '+String(blocker.code||'WORKSPACE'),footer:'Task stopped · '+(blocker.code==='HTML_BROWSER_BUDGET'?'browser inspection limit reached (8 attempts; failures count)':String(blocker.code||'host blocked')),responseLabel:'No verified changes delivered'};
 if(terminal==='CANCELLED'||result?.cancel_requested)return {kind:'CANCELLED',title:'Task cancelled',footer:'Ready · task cancelled'};
 const rendering=result?.task_acceptance?.rendering??result?.taskAcceptance?.rendering,medium=rendering?.items?.some(x=>x.kind==='HTML_BROWSER')?'HTML browser':'SVG',renderFailed=!!renderFailureCode||['FAILED','PARTIAL'].includes(rendering?.status);
 if(result?.verified===true&&terminal==='DELIVERED'&&renderFailed)return {kind:'DELIVERED_WITH_LIMITATION',title:medium==='SVG'?'Changes saved · SVG rendering incomplete':'Changes saved · HTML browser inspection incomplete',footer:'Ready · changes saved; rendering incomplete'};
 if(result?.verified===true&&terminal==='DELIVERED')return {kind:'DELIVERED',title:'Changes saved · configured checks passed',footer:'Ready · checked changes saved'};
 const execution=result?.task_acceptance?.execution??result?.taskAcceptance?.execution;
 const hasDraft=terminal==='DRAFT_UNVERIFIED'&&result?.verified===false&&typeof result.draft==='string'&&!!result.draft.trim()&&!result.model_error_code&&!result.error_code&&!result.errorCode&&!result.blocked_reason&&!(result.missing?.length);
 if(hasDraft&&renderFailed)return {kind:'PARTIAL',title:medium==='SVG'?'Response ended · SVG rendering incomplete':'Response ended · HTML browser inspection incomplete',footer:'Ready · rendering incomplete',responseLabel:'Response · rendering failed; visual inspection incomplete'};
 if(hasDraft&&(readOnly||!writeRequest(goal))&&execution?.write_requested!==true){
  if(rendering?.status==='RENDERED'){
   if(rendering.items?.some(x=>x.model_input!=='SUBMITTED_BY_ADAPTER'))return {kind:'PARTIAL',title:medium+' rendered · visual input unconfirmed',footer:'Ready · visual inspection unconfirmed',responseLabel:'Response · rendered images were not confirmed in model input'};
   return {kind:'RENDERED_RESPONSE',title:medium==='SVG'?'Response complete · SVG rendered':'Response complete · HTML browser inspected',footer:medium==='SVG'?'Ready · response complete; SVG rendered':'Ready · response complete; HTML browser inspected',responseLabel:'Response · no project files changed; visual quality not independently certified'};
  }
  return {kind:'ANSWER',title:'Response complete · no project files changed',footer:'Ready · response complete',responseLabel:'Response · no project files changed'};
 }
 if(hasDraft&&['DECLINED','CANCELLED'].includes(execution?.write_decision))return {kind:'DECLINED',title:'Write scope declined · no verified changes delivered',footer:'Ready · write scope declined',responseLabel:'Response · changes were not authorized'};
 return {kind:'UNDELIVERED',title:'Task ended · result not verified',footer:'Task ended · no verified delivery',responseLabel:'Draft response · no verified changes saved'};
}
module.exports={responsePresentation};
