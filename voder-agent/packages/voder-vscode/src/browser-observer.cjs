'use strict';
/** Browser diagnostics are UNTRUSTED observations, never a candidate PASS or write capability. */
const crypto=require('node:crypto');
function observerSource(token){
 return `(()=>{'use strict';const api=acquireVsCodeApi(),token=${JSON.stringify(token)};let frames=0,clicks=0,errors=[],samples=0,last='',changes=0,stopped=false;const start=performance.now();function safe(v){return String(v).replace(/https?:[^\\s]+/g,'[URL]').slice(0,180);}addEventListener('error',e=>{if(errors.length<8)errors.push(safe(e.message||'Browser error'));});addEventListener('unhandledrejection',()=>{if(errors.length<8)errors.push('Unhandled promise rejection');});addEventListener('click',e=>{if(e.target.closest?.('button'))clicks++;},true);function loop(){if(stopped)return;frames++;requestAnimationFrame(loop);}requestAnimationFrame(loop);function sample(){const canvas=document.querySelector('canvas');if(!canvas)return;try{const c=document.createElement('canvas');c.width=32;c.height=32;c.getContext('2d').drawImage(canvas,0,0,32,32);const data=c.getContext('2d').getImageData(0,0,32,32).data;let h=2166136261;for(let i=0;i<data.length;i++)h=Math.imul(h^data[i],16777619);if(last!==''&&last!==h)changes++;last=h;samples++;}catch{if(errors.length<8)errors.push('Canvas sampling unavailable');}}let sent=0;function report(){if(stopped||sent>=12)return;sample();sent++;api.postMessage({type:'runtime_observation',token,frames,clicks,samples,changes,elapsed:Math.round(performance.now()-start),width:innerWidth,height:innerHeight,errors:[...errors]});}const timer=setInterval(report,1000);setTimeout(()=>{report();stopped=true;clearInterval(timer);},11000);})();`;
}
function validateObservation(value,token){
 if(!value||typeof value!=='object'||Array.isArray(value)||JSON.stringify(value).length>4096||value.type!=='runtime_observation'||value.token!==token||Object.keys(value).some(k=>!['type','token','frames','clicks','samples','changes','elapsed','width','height','errors'].includes(k)))return null;
 for(const key of ['frames','clicks','samples','changes','elapsed','width','height'])if(!Number.isSafeInteger(value[key])||value[key]<0||value[key]>1e7)return null;
 if(!Array.isArray(value.errors)||value.errors.length>8||value.errors.some(e=>typeof e!=='string'||e.length>180))return null;
 return {authority:'UNTRUSTED_BROWSER_OBSERVATIONS_NOT_VGR_PROOF',frames:value.frames,clicks:value.clicks,samples:value.samples,changes:value.changes,elapsed_ms:value.elapsed,viewport:[value.width,value.height],errors:[...value.errors],visual_acceptance:'NOT_ESTABLISHED',control_correctness:'NOT_ESTABLISHED'};
}
function createObserver(panel,onReport){
 const token=crypto.randomUUID();let count=0,last=-1;
 const subscription=panel.webview.onDidReceiveMessage(msg=>{const row=validateObservation(msg,token);if(!row||count>=12||row.elapsed_ms<last)return;count++;last=row.elapsed_ms;onReport(row);});
 return {source:observerSource(token),dispose:()=>subscription.dispose()};
}
module.exports={observerSource,validateObservation,createObserver};
