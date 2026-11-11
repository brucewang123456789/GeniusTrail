'use strict';
const {createHash}=require('node:crypto');
const digest=x=>createHash('sha256').update(x).digest('hex');
const hex=/^[a-f0-9]{64}$/;
function validateVerifiedDiff(result,sessionId){
 if(!result||result.status!=='VERIFIED_DELIVERED'||result.authority!=='VODER_NATIVE_VGR_STATE'||result.sessionId!==sessionId||!hex.test(result.commit_digest||'')||!/^action-[a-f0-9-]{36}$/i.test(result.action_id||'')||result.run_id!==`run-${result.action_id}`)throw Error('Verified diff identity mismatch');
 const rows=result.schema==='voder.session.diff.v2'?result.files:result.schema==='voder.session.diff.v1'?[result]:null;
 if(!Array.isArray(rows)||!rows.length||rows.length>16)throw Error('Unsupported verified diff schema');
 let bytes=0;
 for(const [i,row] of rows.entries()){
  const p=row?.relative_path;
  if(typeof p!=='string'||p.length>240||p.split('/').some(x=>!x||x.startsWith('.')||!/^[A-Za-z0-9._-]+$/.test(x))||i&&rows[i-1].relative_path>=p)throw Error('Unsafe or duplicated diff member');
  if(typeof row.before!=='string'||typeof row.after!=='string'||row.before.includes('\0')||row.after.includes('\0')||!hex.test(row.before_sha256||'')||!hex.test(row.after_sha256||'')||digest(row.before)!==row.before_sha256||digest(row.after)!==row.after_sha256)throw Error('Verified diff content digest mismatch');
  bytes+=Buffer.byteLength(row.before)+Buffer.byteLength(row.after);if(bytes>255000)throw Error('Verified diff review limit');
 }
 return rows;
}
module.exports={validateVerifiedDiff};
