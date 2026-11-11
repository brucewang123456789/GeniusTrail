'use strict';
/** Immutable worker executed only in the remote browser sandbox. Page JavaScript never executes on the host. */
const fs=require('node:fs'),crypto=require('node:crypto');
const controllerPath=process.argv[4]||'/tmp/package';
const {chromium}=require(controllerPath),controllerVersion=require(controllerPath+'/package.json').version;
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
async function main(){
 const input=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
 if(typeof input.content!=='string')throw Error('HTML_SOURCE_LIMIT');
 const browser=await chromium.launch({headless:true,timeout:0,...(process.argv[3]?{executablePath:process.argv[3],args:['--no-sandbox','--disable-dev-shm-usage','--no-zygote','--single-process','--in-process-gpu','--use-gl=angle','--use-angle=swiftshader']}:{args:['--no-sandbox','--disable-dev-shm-usage']})});
 let context;
 // The host owns the rolling idle deadline; no worker total-duration deadline.
 const progress=()=>process.stderr.write('VODER_BROWSER_PROGRESS\n');
 try{
  context=await browser.newContext({viewport:{width:1060,height:900},deviceScaleFactor:1,serviceWorkers:'block',acceptDownloads:false,permissions:[]});
  context.setDefaultTimeout(0);context.setDefaultNavigationTimeout(0);progress();
  const errors=[],consoleErrors=[],blockedRequests=[];
  await context.route('**/*',r=>{if(blockedRequests.length<32)blockedRequests.push({protocol:r.request().url().split(':')[0]});return r.abort('blockedbyclient');});
  if(context.routeWebSocket)await context.routeWebSocket('**/*',ws=>ws.close());
  const page=await context.newPage();context.on('page',p=>{if(p!==page)void p.close();});
  page.on('dialog',d=>void d.dismiss());page.on('pageerror',e=>{if(errors.length<16)errors.push(String(e.message).slice(0,300));});page.on('console',m=>{if(m.type()==='error'&&consoleErrors.length<16)consoleErrors.push(m.text().slice(0,300));});
  const evaluate=async expression=>{const result=await page.evaluate(expression);progress();return result;};
  const probe=input.probe??{};
  const lookup=p=>`(()=>{let o=window;const parts=${JSON.stringify(p??'').replace(/</g,'\\u003c')}.split('.');for(const k of parts.slice(0,-1))o=o?.[k];return {o,key:parts[parts.length-1]};})()`;
  const invoke=(p,args=[],serialize=false)=>`(()=>{const r=${lookup(p)};if(typeof r.o?.[r.key]!=='function')return ${serialize?"'null'":"null"};const value=r.o[r.key](...${JSON.stringify(args)});return ${serialize?'JSON.stringify(value??null)':'value'};})()`;
  const state=()=>probe.state_function?evaluate(invoke(probe.state_function,[],true)):Promise.resolve('null');
  const setPhase=p=>evaluate(invoke(probe.phase_function,[p]));
  const controlExpression=method=>`(()=>{const r=${lookup(probe.controls?.[method])};if(typeof r.o?.[r.key]==='function'){r.o[r.key]();return 'APPLICATION_API';}const buttons=[...document.querySelectorAll('button')];const label=b=>(b.getAttribute('aria-label')??b.textContent).trim();const toggle=buttons.find(b=>/play|pause|播放|暂停/i.test(label(b)));if(${JSON.stringify(method)}==='restart'){const b=buttons.find(b=>/restart|重启|重新开始/i.test(label(b)));if(b){b.click();return 'BUTTON';}}else if(toggle){const isPlay=/play|播放/i.test(label(toggle));if((${JSON.stringify(method)}==='play'&&isPlay)||(${JSON.stringify(method)}==='pause'&&!isPlay))toggle.click();return 'BUTTON';}return 'UNAVAILABLE';})()`;
  const structure=()=>evaluate(`(()=>{const svg=document.querySelector('svg');const ids=[...document.querySelectorAll('[id]')].map(x=>x.id);return {viewBox:svg?.getAttribute('viewBox')??null,title:svg?.querySelector('title')?.textContent??null,description:svg?.querySelector('desc')?.textContent??null,duplicateIds:ids.filter((id,i)=>ids.indexOf(id)!==i),pageWidth:document.documentElement.scrollWidth,viewport:innerWidth,externalReferences:[...document.querySelectorAll('[src],[href]')].map(n=>n.getAttribute('src')??n.getAttribute('href')).filter(x=>x&&!x.startsWith('#')&&!x.startsWith('data:')),hasForeignObject:!!document.querySelector('foreignObject')}})()`);
  async function load(reduced){await page.emulateMedia({reducedMotion:reduced?'reduce':'no-preference'});await page.goto('about:blank');await page.setContent(input.content,{waitUntil:'load'});await sleep(100);progress();}
  await load(false);const initial=await state(),structureResult=await structure(),hasPhase=!!probe.phase_function&&await evaluate(`(()=>{const r=${lookup(probe.phase_function)};return typeof r.o?.[r.key]==='function';})()`);
  const phaseSamples=[],images=[];
  if(hasPhase)for(const p of [...Array.from({length:16},(_,i)=>i/16),1-1e-6,0]){await setPhase(p);phaseSamples.push({phase:p,state:await state()});}
  for(const p of [0,.25,.5,.75]){
   if(hasPhase)await setPhase(p);
   for(const scale of [1,.5]){
    // Capture native artwork and its responsive half-size view directly in the browser.
    if(scale===.5)await page.setViewportSize({width:548,height:600});
    const locator=page.locator('svg').first();const hasSvg=await locator.count();const bytes=hasSvg?await locator.screenshot({type:'png',timeout:0}):await page.screenshot({type:'png',timeout:0});
    progress();if(bytes.length>1048576)throw Error('HTML_BROWSER_IMAGE_LIMIT');images.push({phase:hasPhase?p:null,phase_control:hasPhase?'APPLICATION_API':'UNAVAILABLE',scale,data:bytes.toString('base64')});
    if(scale===.5)await page.setViewportSize({width:1060,height:900});
   }
  }
  const checks=[];
  for(const expression of input.checks??[]){try{checks.push({status:'EVALUATED',value:await evaluate('JSON.stringify(('+expression+'))')});}catch{checks.push({status:'ERROR',code:'PAGE_CHECK_FAILED'});}}
  const controls=[];
  for(const [name,method]of [['paused','pause'],['resumed','play'],['restarted','restart']]){const action=await evaluate(controlExpression(method)),before=await state();await sleep(name==='paused'?160:100);controls.push({name,action,before,after:await state()});}
  if(hasPhase)await setPhase(0);await evaluate(controlExpression('play'));
  const playback=[],started=performance.now(),seconds=input.playback_seconds??16;
  for(let i=0;i<=Math.ceil(seconds*4);i++){const raw=await state();let parsed;try{parsed=JSON.parse(raw);}catch{}const dom=await evaluate("document.querySelector('svg')?.outerHTML??document.body.innerHTML");playback.push({elapsed_seconds:(performance.now()-started)/1000,phase:typeof parsed?.phase==='number'?parsed.phase:null,playing:typeof parsed?.playing==='boolean'?parsed.playing:null,state_sha256:sha(String(raw)),svg_dom_sha256:sha(dom)});if(i<Math.ceil(seconds*4))await sleep(250);}
  await evaluate(controlExpression('pause'));
  await page.setViewportSize({width:360,height:780});const narrow=await structure();
  await load(true);const reducedMotionInitial=await state();
  process.stdout.write(JSON.stringify({status:'BROWSER_INSPECTED',remote_source_sha256:sha(input.content),browser:{product:'Chromium',version:browser.version(),controller:'Playwright',controller_version:controllerVersion},phase_api_available:hasPhase,structure:structureResult,initial,phaseSamples,checks,controls,playback,narrow,reducedMotionInitial,errors,consoleErrors,blockedRequests,images,limitations:['Page-side state and checks are untrusted application observations, not independent geometry proofs.','Timed samples and still phase frames do not independently certify smooth playback or visual quality.','Reduced images use a smaller browser viewport; responsive changes are included rather than hidden.']})+'\n');
 }finally{await context?.close();await browser.close();}
}
main().catch(e=>{process.stderr.write(String(e.message).slice(0,1200)+'\n');process.exitCode=1;});
