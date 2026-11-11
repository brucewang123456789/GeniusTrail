/** Render only validated, self-contained static SVG. No generated code or shell. */
import {spawn} from 'node:child_process';
import {createRequire} from 'node:module';
import {join,resolve,dirname} from 'node:path';
import {fileURLToPath} from 'node:url';
const require=createRequire(import.meta.url);
export function processBytes(program,args,input,{timeout=20000,maxBytes=4*1024*1024,signal}={}){
 return new Promise((resolve,reject)=>{
  if(signal?.aborted)return reject(Object.assign(Error('Cancelled'),{code:'CANCELLED'}));
  const child=spawn(program,args,{stdio:['pipe','pipe','pipe'],shell:false,env:{PATH:'/usr/local/bin:/usr/bin:/bin',LANG:'C.UTF-8'}});let stdout=Buffer.alloc(0),stderr=Buffer.alloc(0),reason=null;
  const stop=code=>{reason=code;child.kill('SIGKILL');};const abort=()=>stop('CANCELLED'),timer=setTimeout(()=>stop('PREVIEW_TIMEOUT'),timeout);signal?.addEventListener('abort',abort,{once:true});
  for(const [stream,kind] of [[child.stdout,'out'],[child.stderr,'err']])stream.on('data',b=>{if(stdout.length+stderr.length+b.length>maxBytes){stop('PREVIEW_OUTPUT_LIMIT');return;}if(kind==='out')stdout=Buffer.concat([stdout,b]);else stderr=Buffer.concat([stderr,b]);});
  child.stdin.on('error',()=>{});child.stdin.end(input);child.on('error',e=>{clearTimeout(timer);signal?.removeEventListener('abort',abort);reject(e);});
  child.on('close',code=>{clearTimeout(timer);signal?.removeEventListener('abort',abort);if(reason||code!==0)reject(Object.assign(Error(stderr.toString('utf8').slice(-1600)||reason||'Preview failed'),{code:reason||'SVG_CHECK_FAILED'}));else resolve(stdout);});
 });
}
export async function renderSvg(content,{agentRoot=resolve(dirname(fileURLToPath(import.meta.url)),'../../..'),signal}={}){
 if(typeof content!=='string')throw Object.assign(Error('Bounded SVG text required'),{code:'SVG_SIZE'});
 const check=JSON.parse((await processBytes('/usr/bin/python3',['-I','-B',fileURLToPath(new URL('../resources/svg-inspect.py',import.meta.url))],content,{signal})).toString());
 let sharpPath;for(const module of ['sharp',...(agentRoot?[join(agentRoot,'../.voder-tools/renderer/node_modules/sharp')]:[]),...(process.env.CODEX_PRIMARY_RUNTIME_NODE_MODULES?[join(process.env.CODEX_PRIMARY_RUNTIME_NODE_MODULES,'sharp')]:[])]){try{sharpPath=require.resolve(module);break;}catch{}}
 const images=[];let renderer;
 if(sharpPath){
  const raw=await processBytes(process.execPath,['--max-old-space-size=256',fileURLToPath(new URL('./svg-render-worker.mjs',import.meta.url)),sharpPath],JSON.stringify({...check,content}),{signal});
  const result=JSON.parse(raw.toString());renderer=result.renderer;for(const b of result.images)images.push(Buffer.from(b,'base64'));
 }else{
  // CairoSVG is an optional managed dependency. A missing renderer is an explicit capability limit.
  renderer='CairoSVG';
  const program='import sys,json,base64\nfrom cairosvg.surface import PNGSurface\ns=json.load(sys.stdin)\ndef no_fetch(*a,**kw): raise ValueError("External resources forbidden")\nout=[]\nfor scale in (1,.5):\n out.append(base64.b64encode(PNGSurface.convert(bytestring=s["content"].encode(),output_width=max(1,round(s["width"]*scale)),output_height=max(1,round(s["height"]*scale)),unsafe=False,url_fetcher=no_fetch)).decode())\nprint(json.dumps(out))';
  try{const raw=await processBytes('/usr/bin/python3',['-I','-B','-c',program],JSON.stringify({...check,content}),{signal});for(const b of JSON.parse(raw.toString()))images.push(Buffer.from(b,'base64'));}
  catch(e){if(/No module named ['"]cairosvg/.test(e.message))throw Object.assign(Error('No managed sharp or system CairoSVG renderer is installed. Source creation remains available; do not claim rendering or visual checks.'),{code:'SVG_RENDERER_UNAVAILABLE'});throw e;}
 }
 if(images.some(b=>b.length>1048576||!b.subarray(0,8).equals(Buffer.from([137,80,78,71,13,10,26,10]))))throw Object.assign(Error('Preview exceeds image budget'),{code:'SVG_IMAGE_LIMIT'});
 return {check,renderer,images};
}
