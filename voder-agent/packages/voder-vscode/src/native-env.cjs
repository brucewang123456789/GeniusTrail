'use strict';
/** Import the user's existing A12 Modal connection into the child RPC process, in memory only. */
const fs=require('node:fs');
const os=require('node:os');
const path=require('node:path');
const {spawnSync}=require('node:child_process');
const {modelEnv}=require('./model-env.cjs');
const {IMAGE}=require('./native-profile.cjs');
const KEYS=['MODAL_TOKEN_ID','MODAL_TOKEN_SECRET','VODER_MODAL_APP'];
function nativeEnv(parent=process.env,{load}={}){
 const env=modelEnv(parent);
 if(!env.VODER_MODEL_ID||!env.VODER_MODEL_API_KEY){const catalog=path.join(os.homedir(),'.config/voder/additional-models.json');if(env.VODER_MODEL_ID||env.VODER_MODEL_API_KEY||!fs.existsSync(catalog))throw Object.assign(Error('No configured model API credentials'),{code:'CHAT_MODEL_NOT_CONFIGURED'});}
 if(!KEYS.every(k=>typeof env[k]==='string'&&env[k].length)){
  const loader=path.join(os.homedir(),'.config/voder/load-modal.sh'),stat=fs.lstatSync(loader);
  if(!stat.isFile()||stat.isSymbolicLink()||(stat.mode&0o022)!==0||(process.getuid&&stat.uid!==process.getuid()))throw Object.assign(Error('Private Modal loader must be owned by the user and not group/other-writable'),{code:'CHAT_MODAL_LOADER_UNSAFE'});
  const reader='import os,json;print(json.dumps({k:os.environ.get(k) for k in ("MODAL_TOKEN_ID","MODAL_TOKEN_SECRET","VODER_MODAL_APP")}))';
  const result=load?load(loader,env):spawnSync('/bin/bash',['-c','source "$1" >/dev/null 2>&1; /usr/bin/python3 -c "$2"','voder-modal',loader,reader],{env,encoding:'utf8',timeout:15000,maxBuffer:16384});
  if(result.status!==0||typeof result.stdout!=='string')throw Object.assign(Error('Existing Modal loader failed; native submission blocked'),{code:'CHAT_MODAL_LOADER_FAILED'});
  let vals;try{vals=JSON.parse(result.stdout);}catch{throw Object.assign(Error('Modal loader returned invalid credentials'),{code:'CHAT_MODAL_LOADER_INVALID'});}
  if(!vals||KEYS.some(k=>typeof vals[k]!=='string'||!vals[k]||vals[k].length>4096||/[\r\n\0]/.test(vals[k])))throw Object.assign(Error('Modal credentials missing; native submission blocked'),{code:'CHAT_MODAL_CREDS_MISSING'});
  for(const k of KEYS)env[k]=vals[k];
 }
 // Same pinned sandbox backend/image as the previously validated A12 evaluator.
 env.VODER_MODAL_IMAGE=IMAGE;env.VODER_SANDBOX_BACKEND='modal';
 // Native use requires explicit profile approval in the VS Code UI (not inferred from draft mode).
 env.VODER_ALLOW_MODEL_API='YES';env.VODER_ALLOW_MODAL_CHARGES='YES';
 delete env.NODE_OPTIONS;delete env.NODE_PATH;delete env.PYTHONPATH;
 return env;
}
module.exports={nativeEnv};
