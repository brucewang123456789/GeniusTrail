'use strict';
/** Pure policy functions; no VS Code UI or execution authority. */
const {resolve,isAbsolute}=require('node:path');
function resolveFolderForTerminal(folders,terminal){
  if(!Array.isArray(folders)||!folders.length)return null;
  const cwd=terminal?.creationOptions?.cwd;
  const p=typeof cwd==='string'?cwd:(cwd?.fsPath||null);
  if(p&&isAbsolute(p)){
    const norm=resolve(p);const matches=folders.filter(f=>norm===resolve(f.uri.fsPath)||norm.startsWith(resolve(f.uri.fsPath)+require('node:path').sep));
    return matches.sort((a,b)=>b.uri.fsPath.length-a.uri.fsPath.length)[0]||null;
  }
  return folders.length===1?folders[0]:null;
}
function classifyStatus(payload){
  const status=payload?.status||payload?.result?.status||'UNKNOWN';
  // No editor-side synthesis of VGR commitment; VERIFIED requires an independent native summary field.
  const verified=['VERIFIED','VERIFIED_NOT_DELIVERED','DELIVERED'].includes(status) && (payload?.verified===true||payload?.result?.verified===true);
  return {status,verified,label:verified?(status==='DELIVERED'?'DELIVERED (verified workspace)':status==='VERIFIED_NOT_DELIVERED'?'VERIFIED (NOT DELIVERED)':'VERIFIED (native evidence only)'):'UNVERIFIED / '+status};
}
function canSubmitInMode(trusted,mode,gateway){return Boolean(trusted && (mode==='draft'||(mode==='native'&&typeof gateway==='string'&&gateway.trim())));}
module.exports={resolveFolderForTerminal,classifyStatus,canSubmitInMode};
