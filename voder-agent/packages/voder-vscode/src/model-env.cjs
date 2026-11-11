'use strict';
const fs=require('node:fs');
const os=require('node:os');
const path=require('node:path');
const ALLOWED=['VODER_MODEL_ID','VODER_MODEL_API_KEY','VODER_MODEL_VENDOR','VODER_MODEL_PROTOCOL','VODER_MODEL_ENDPOINT','VODER_MODEL_CONTEXT_WINDOW','VODER_MODEL_MAX_TOKENS','VODER_MODEL_INPUT'];
function modelEnv(parent=process.env){
 const base=path.join(os.homedir(),'.config','voder');
 const file=path.join(base,'model-provider.json');
 let st;
 try{st=fs.lstatSync(file);}catch(e){if(e.code==='ENOENT')return supplementEnv({...parent},base);throw e;}
 if(!st.isFile()||st.isSymbolicLink()||((st.mode&0o077)!==0)||st.uid!==process.getuid())throw Error('UNSAFE_PRIVATE_MODEL_CONFIG');
 const data=JSON.parse(fs.readFileSync(file,'utf8'));
 if(!data||typeof data!=='object'||Array.isArray(data)||Object.keys(data).some(x=>!ALLOWED.includes(x)))throw Error('INVALID_PRIVATE_MODEL_CONFIG');
 if(!data.VODER_MODEL_ID||!data.VODER_MODEL_API_KEY)throw Error('MODEL_ID_AND_API_KEY_REQUIRED');
 for(const [k,v] of Object.entries(data))if(typeof v!=='string'||v.length>4096||/[\r\n\0]/.test(v))throw Error('INVALID_PRIVATE_MODEL_FIELD');
 // Conflicting ambient variables must not override the private, explicit user choice.
 const merged={...parent,...data};
 if(merged.VODER_MODEL_INPUT===undefined&&merged.VODER_MODEL_ID==='deepseek-flash'&&(!merged.VODER_MODEL_VENDOR||merged.VODER_MODEL_VENDOR==='deepseek')&&(!merged.VODER_MODEL_ENDPOINT||/^https:\/\/api\.deepseek\.com(?:\/|$)/.test(merged.VODER_MODEL_ENDPOINT)))merged.VODER_MODEL_INPUT='text,image';
 return supplementEnv(merged,base);
}
function supplementEnv(merged,base){
 // Operator-provided optional search/Modal credentials are loaded from private files.
 for(const [name,pattern] of [['search-provider-credentials.json',/^VODER_SEARCH_[A-Z0-9_]+_API_KEY$/],['modal-provider.json',/^(MODAL_TOKEN_ID|MODAL_TOKEN_SECRET|VODER_MODAL_APP)$/]]){
  const f=path.join(base,name);let s;try{s=fs.lstatSync(f);}catch(e){if(e.code==='ENOENT')continue;throw e;}
  if(!s.isFile()||s.isSymbolicLink()||(s.mode&0o077)!==0||(process.getuid&&s.uid!==process.getuid())||s.size>65536)throw Error('UNSAFE_PRIVATE_CREDENTIAL_CONFIG');
  const vals=JSON.parse(fs.readFileSync(f,'utf8'));
  if(!vals||typeof vals!=='object'||Array.isArray(vals)||Object.entries(vals).some(([k,v])=>!pattern.test(k)||typeof v!=='string'||v.length>4096||/[\r\n\0]/.test(v)))throw Error('INVALID_PRIVATE_CREDENTIAL_CONFIG');
  Object.assign(merged,vals);
 }
 return merged;
}

module.exports={modelEnv};
