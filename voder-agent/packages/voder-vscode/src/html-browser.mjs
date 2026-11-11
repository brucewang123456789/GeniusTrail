/** Separate browser inspection capability. Existing candidate sandbox and Rust authority are unchanged. */
import {createRequire} from 'node:module';import {readFile} from 'node:fs/promises';import {join} from 'node:path';import {createHash} from 'node:crypto';
const sha=b=>createHash('sha256').update(b).digest('hex');
const BROWSER_ARCHIVE_SHA256='3740bba3a3e93b4c7bd7c0f6e3c0e5f8cdce498bef13511e4d534cc99a8ce198';
export const BROWSER_IMAGE='mcr.microsoft.com/playwright:v1.61.1-noble';
const fail=(code,m)=>{throw Object.assign(Error(m),{code});};
export function browserIdleWatchdog(onTimeout,{idleMs=3600000,set=setTimeout,clear=clearTimeout}={}){
 let timer,closed=false;
 const touch=()=>{if(closed)return;clear(timer);timer=set(()=>{closed=true;onTimeout();},idleMs);};
 touch();return {touch,close(){closed=true;clear(timer);}};
}
async function consume(stream,limit,signal,onProgress=()=>{}){const reader=stream.getReader(),chunks=[];let size=0;try{for(;;){signal?.throwIfAborted();const{done,value}=await reader.read();if(done)break;const b=Buffer.from(typeof value==='string'?value:Buffer.from(value));size+=b.length;if(size>limit){await reader.cancel();fail('HTML_BROWSER_OUTPUT_LIMIT','Remote browser output exceeds limit');}if(b.length){onProgress();chunks.push(b);}}return Buffer.concat(chunks).toString('utf8');}finally{reader.releaseLock();}}
export function validateBrowserInput({path,content,checks=[],playback_seconds=16,preview_path,probe}={}){
 if(typeof path!=='string'||path.length>240||!/^[-A-Za-z0-9_][A-Za-z0-9._/-]*\.html$/.test(path)||path.split('/').some(x=>!x||x==='.'||x==='..'))fail('HTML_BROWSER_PATH','Safe relative .html label required');
 if(typeof content!=='string'||!/^\s*(?:<!doctype html[^>]*>\s*)?<html[\s>]/i.test(content))fail('HTML_BROWSER_CONTENT','Bounded complete HTML required');
 if(!Array.isArray(checks)||checks.length>8||checks.some(x=>typeof x!=='string'||!x.trim()||Buffer.byteLength(x)>8192))fail('HTML_BROWSER_CHECKS','At most eight bounded page expressions required');
 if(typeof playback_seconds!=='number'||!Number.isFinite(playback_seconds)||playback_seconds<0||playback_seconds>24)fail('HTML_BROWSER_DURATION','Playback must be between 0 and 24 seconds');
 if(probe!==undefined){
  const functionPath=p=>typeof p==='string'&&p.length<=160&&/^[A-Za-z_$][A-Za-z0-9_$]*(?:\.[A-Za-z_$][A-Za-z0-9_$]*)*$/.test(p)&&!p.split('.').some(x=>['__proto__','prototype','constructor'].includes(x));
  if(!probe||typeof probe!=='object'||Array.isArray(probe)||Object.keys(probe).some(k=>!['phase_function','state_function','controls'].includes(k))||['phase_function','state_function'].some(k=>probe[k]!==undefined&&!functionPath(probe[k])))fail('HTML_BROWSER_PROBE','Safe application function paths required');
  if(probe.controls!==undefined&&(!probe.controls||typeof probe.controls!=='object'||Array.isArray(probe.controls)||Object.keys(probe.controls).some(k=>!['play','pause','restart'].includes(k))||Object.values(probe.controls).some(p=>!functionPath(p))))fail('HTML_BROWSER_PROBE','Safe named control function paths required');
 }
 if(preview_path!==undefined&&(typeof preview_path!=='string'||!/^[-A-Za-z0-9_][A-Za-z0-9._/-]*\.png$/.test(preview_path)||preview_path.split('/').some(x=>!x||x==='.'||x==='..')))fail('HTML_BROWSER_PATH','Safe optional PNG path required');
 return {path,content,checks,playback_seconds,...(preview_path?{preview_path}:{}),...(probe?{probe}:{})};
}
export async function inspectHtml(input,{agentRoot,signal,driver,onStage=()=>{}}={}){
 const task=validateBrowserInput(input);signal?.throwIfAborted();
 if(!driver){
  if(process.env.VODER_ALLOW_MODAL_CHARGES!=='YES')fail('HTML_BROWSER_AUTH_REQUIRED','Browser inspection needs authorized remote sandbox usage');
  const require=createRequire(join(agentRoot,'../packages/voder-sandbox-modal/package.json'));
  const sdk=await import(require.resolve('modal')),ModalClient=sdk.ModalClient??sdk.default?.ModalClient;if(typeof ModalClient!=='function')fail('HTML_BROWSER_SETUP_FAILED','Installed Modal SDK does not expose ModalClient');const modal=new ModalClient();
  driver={async create(){const app=await modal.apps.fromName(process.env.VODER_MODAL_APP,{createIfMissing:true});return modal.sandboxes.create(app,modal.images.fromRegistry(BROWSER_IMAGE),{timeoutMs:86400000,idleTimeoutMs:3600000,memoryMiB:2048,memoryLimitMiB:2048,cpu:1,cpuLimit:1,blockNetwork:true,workdir:'/tmp'});}};
 }
 let sandbox,expired=false,rejectEnd;
 const end=new Promise((_,reject)=>{rejectEnd=reject;});end.catch(()=>{});
 // Independent bounds: IO activity can reset transport idle, never the verified-stage clock.
 const browserTimeout=()=>{if(expired)return;expired=true;rejectEnd(Object.assign(Error('Browser made no verified stage progress for 3600 seconds'),{code:'HTML_BROWSER_TIMEOUT'}));void sandbox?.terminate().catch(()=>{});};
 const watchdog=browserIdleWatchdog(browserTimeout);
 const stageWatchdog=browserIdleWatchdog(browserTimeout);
 const stop=()=>{rejectEnd(Object.assign(Error('Browser inspection cancelled'),{code:'CANCELLED'}));void sandbox?.terminate().catch(()=>{});};signal?.addEventListener('abort',stop,{once:true});
 // IO bytes keep the transport idle watchdog alive, but are not semantic browser progress.
 const ioActivity=()=>watchdog.touch();
 const progress=status=>{watchdog.touch();stageWatchdog.touch();if(status)onStage(status);};
 const wait=p=>Promise.race([p,end]).then(value=>{watchdog.touch();return value;});
 try{
  progress('BROWSER_SANDBOX_CREATE');sandbox=await wait(Promise.resolve(driver.create()).then(s=>{if(expired||signal?.aborted){void s.terminate().catch(()=>{});throw Object.assign(Error('Browser setup cancelled'),{code:expired?'HTML_BROWSER_TIMEOUT':'CANCELLED'});}return s;}));if(expired)fail('HTML_BROWSER_TIMEOUT','Browser setup exceeded deadline');signal?.throwIfAborted();
  // Upload only the candidate and immutable worker. No workspace snapshot, credentials or host environment.
  const worker=await readFile(new URL('./html-browser-worker.cjs',import.meta.url),'utf8'),archive=await readFile(new URL('./vendor/playwright-core-1.61.1.tgz',import.meta.url));
  if(sha(archive)!==BROWSER_ARCHIVE_SHA256)fail('HTML_BROWSER_SETUP_FAILED','Bundled browser controller checksum differs');
  await wait(sandbox.filesystem.writeText(worker,'/tmp/voder-html-worker.cjs'));await wait(sandbox.filesystem.writeText(JSON.stringify(task),'/tmp/voder-html-input.json'));
  await wait(sandbox.filesystem.writeBytes(archive,'/tmp/voder-playwright.tgz'));
  const setup=await wait(sandbox.exec(['/bin/tar','-xzf','/tmp/voder-playwright.tgz','-C','/tmp','--no-same-owner'],{}));
  const [setupOut,setupErr,setupRC]=await wait(Promise.all([consume(setup.stdout,8192,signal,ioActivity),consume(setup.stderr,8192,signal,ioActivity),setup.wait()]));if(setupRC!==0)fail('HTML_BROWSER_SETUP_FAILED','Remote browser controller extraction failed');
  progress('BROWSER_EXECUTION');const p=await wait(sandbox.exec(['/usr/bin/env','node','/tmp/voder-html-worker.cjs','/tmp/voder-html-input.json'],{workdir:'/tmp'}));
  const [stdout,stderr,rc]=await wait(Promise.all([consume(p.stdout,10*1024*1024,signal,ioActivity),consume(p.stderr,8192,signal,ioActivity),p.wait()]));signal?.throwIfAborted();
  if(rc!==0)fail('HTML_BROWSER_EXECUTION_FAILED','Remote browser failed: '+stderr.slice(-800));let result;try{result=JSON.parse(stdout);}catch{fail('HTML_BROWSER_RESULT','Malformed remote browser evidence');}
  progress('BROWSER_RESULTS_RECEIVED');if(result.remote_source_sha256!==sha(task.content)||result.status!=='BROWSER_INSPECTED'||!Array.isArray(result.images)||result.images.length!==8)fail('HTML_BROWSER_RESULT','Incomplete remote browser evidence');
  const images=result.images.map(x=>{const bytes=Buffer.from(x.data,'base64');if(bytes.length<24||bytes.length>1048576||!bytes.subarray(0,8).equals(Buffer.from([137,80,78,71,13,10,26,10])))fail('HTML_BROWSER_RESULT','Invalid browser PNG');return bytes;});
  const frames=result.images.map(({phase,scale,phase_control},i)=>({phase,scale,phase_control,width:images[i].readUInt32BE(16),height:images[i].readUInt32BE(20),sha256:sha(images[i])}));delete result.images;progress('BROWSER_EVIDENCE_VALIDATED');
  return {result:{...result,path:task.path,source_sha256:sha(task.content),source_bytes:Buffer.byteLength(task.content),sandbox_id:sandbox.sandboxId,backend:'modal_browser_inspection',frames,authority:'BROWSER_OBSERVATIONS_NOT_VGR_PROOF'},images};
 }catch(e){if(signal?.aborted)fail('CANCELLED','Browser inspection cancelled');if(expired)fail('HTML_BROWSER_TIMEOUT','Browser made no verified stage progress for 3600 seconds');if(/^(?:idleTimeoutMs|timeoutMs) must be (?:positive|a multiple)/.test(String(e?.message??''))){e.code='HTML_BROWSER_CONFIGURATION_ERROR';e.terminalTaskBlocker=true;}throw e;}
 finally{watchdog.close();stageWatchdog.close();signal?.removeEventListener('abort',stop);if(sandbox){onStage('BROWSER_SANDBOX_TERMINATE');let cleanupTimer;try{await Promise.race([sandbox.terminate(),new Promise((_,reject)=>{cleanupTimer=setTimeout(()=>reject(Object.assign(Error('Remote browser cleanup not confirmed; provider lifetime remains bounded'),{code:'HTML_BROWSER_CLEANUP_UNCONFIRMED'})),10000);})]);}finally{clearTimeout(cleanupTimer);}}}
}
