/** Host-owned request budgets. null means no cumulative limit. */
const keys=['max_requests','max_search_requests','max_fetch_requests','task_timeout_ms'];
const fail=()=>{throw Object.assign(new Error('WEB_CONFIG_INVALID'),{code:'WEB_CONFIG_INVALID'})};
export function normalizeBudget(raw){
 if(!raw||typeof raw!=='object'||Array.isArray(raw)||Object.keys(raw).some(k=>!keys.includes(k)))fail();
 const out={};for(const k of keys){const v=raw[k]===undefined?null:raw[k];
  if(v!==null&&(!Number.isSafeInteger(v)||v<0||(k==='task_timeout_ms'&&(v===0||v>2147483647))))fail();out[k]=v;
 }return Object.freeze(out);
}
export function selectBudget({configured,explicit,legacy}={}){
 if(explicit!==undefined)return normalizeBudget(explicit);
 if(configured!==undefined)return normalizeBudget(configured);
 if(legacy!==null&&(!Number.isSafeInteger(legacy)||legacy<1))throw Object.assign(new Error('INVALID_SEARCH_POLICY'),{code:'INVALID_SEARCH_POLICY'});
 return normalizeBudget({max_requests:legacy});
}
