/** Stable, Voder-owned boundary; a model tool result is never a VGR decision. */
export const SCHEMA = 'voder.agent.action.v1';
export const statuses = Object.freeze(['CREATED','INSPECTING','WAITING_FOR_INPUT','READY','PROPOSING','AUTHORIZING','EXECUTING','OBSERVING','EVALUATING','BLOCKED','FAILED','CANCELLED','VERIFIED','VERIFIED_NOT_DELIVERED','DELIVERED']);
export class VoderError extends Error {
  constructor(code, message) { super(message); this.name='VoderError';this.code=code; }
}
export function assertNonempty(value, name, max=4096) {
  if(typeof value!=='string'||!value.trim()||Buffer.byteLength(value,'utf8')>max) throw new VoderError('INVALID_ARGUMENT',name+' missing or too long');
  return value;
}
export function assertAction(action) {
  if(!action || typeof action!=='object'||Array.isArray(action)||action.schema!==SCHEMA) throw new VoderError('INVALID_ACTION','Invalid action schema');
  for(const k of ['session_id','action_id','kind','idempotency_key']) assertNonempty(action[k],k,256);
  if(!['read','patch','patch_batch','command','validate'].includes(action.kind)) throw new VoderError('INVALID_ACTION','Unknown kind');
  if(!Array.isArray(action.workspace_scope)||action.workspace_scope.length===0||action.workspace_scope.length>32) throw new VoderError('INVALID_ACTION','Invalid scope');
  if(action.workspace_scope.some(x=>typeof x!=='string'||!x||x.startsWith('/')||x.includes('..')||x.includes('\\')||x.includes('\0'))) throw new VoderError('INVALID_ACTION','Dangerous scope');
  if(!action.evidence_obligations || !Array.isArray(action.evidence_obligations)||!action.evidence_obligations.length) throw new VoderError('INVALID_ACTION','Missing evidence obligations');
  if(!action.budget||!Number.isInteger(action.budget.max_runtime_ms)||action.budget.max_runtime_ms<1) throw new VoderError('INVALID_ACTION','Missing runtime budget');
  return structuredClone(action);
}
export function requireNativeReceipt(response, request) {
  // Intentionally never derive delivery from `exit_code === 0` or agent text.
  if(!response||typeof response!=='object'||response.authority!=='VODER_NATIVE_VGR'||response.action_id!==request.action_id||response.session_id!==request.session_id) throw new VoderError('UNVERIFIED','VGR-native identity not authenticated');
  if(!['DENIED','OBSERVED','ROLLED_BACK','COMMITTED'].includes(response.status)) throw new VoderError('UNVERIFIED','Unrecognized native verdict');
  if(typeof response.journal_digest!=='string'||!/^[a-f0-9]{64}$/.test(response.journal_digest)) throw new VoderError('UNVERIFIED','Native journal digest absent');
  if(response.status==='COMMITTED'&&(!response.commit_digest|| !/^[a-f0-9]{64}$/.test(response.commit_digest))) throw new VoderError('UNVERIFIED','Native final commit digest absent');
  return Object.freeze(structuredClone(response));
}

/** v1 canonical authorized source path. Filesystem confinement and exact policy remain mandatory. */
export function isAuthorizedSourcePath(value){
 return typeof value==='string' && value.length>0 && value.length<=240 &&
  value.split('/').every(part=>/^[A-Za-z0-9_-][A-Za-z0-9._-]*$/.test(part) && !part.endsWith('.') &&
   !/^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)/i.test(part));
}
