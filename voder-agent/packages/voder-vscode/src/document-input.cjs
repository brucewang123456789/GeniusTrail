'use strict';
const {spawn}=require('node:child_process');
const path=require('node:path');
const fs=require('node:fs'),os=require('node:os');
const {LIMITS,availableMemory}=require('./input-capacity.cjs');
function decoderPython(){
 if(process.env.VODER_DOCUMENT_PYTHON)return process.env.VODER_DOCUMENT_PYTHON;
 const file=path.join(os.homedir(),'.config/voder/document-runtime.json');
 try{const st=fs.lstatSync(file);if(!st.isFile()||st.isSymbolicLink()||st.mode&0o077||st.uid!==process.getuid())throw error('REFERENCE_DOCUMENT_DEPENDENCY','Unsafe document runtime settings');const cfg=JSON.parse(fs.readFileSync(file,'utf8'));if(Object.keys(cfg).join(',')!=='python'||typeof cfg.python!=='string'||!path.isAbsolute(cfg.python))throw error('REFERENCE_DOCUMENT_DEPENDENCY','Invalid document runtime settings');return cfg.python;}catch(e){if(e.code==='ENOENT')return 'python3';throw e;}
}

const OFFICE=new Set('.doc .docx .docm .dot .dotm .dotx .odt .ott .rtf .ppt .pptx .pptm .pot .potm .potx .pps .ppsm .ppsx .ppa .pwz .wiz .odp .otp .xls .xlsx .xlsm .xlt .xltm .xltx .xla .xlam .xlb .xlc .xlm .xlw .ods .ots .pages .key .keynote .numbers'.split(' '));
const EXTRA_IMAGES=new Set('.gif .bmp .tif .tiff .avif .heic .heif .ico'.split(' '));
const error=(code,message)=>Object.assign(Error(message),{code});
const ext=name=>path.extname(name).toLowerCase();
function needsVisual(name,bytes){return bytes?.subarray(0,5).toString()==='%PDF-'||ext(name)==='.pdf'||OFFICE.has(ext(name));}
function runDecoder(operation,name,bytes,{signal,destination,expandedBytes}={}){
 return new Promise((resolve,reject)=>{
  let proc,settled=false,size=0;const chunks=[];
  if(signal?.aborted){reject(error('REFERENCE_CANCELLED','File preparation cancelled'));return;}
  const done=(err,result)=>{if(settled)return;settled=true;clearTimeout(timer);signal?.removeEventListener('abort',abort);if(err&&proc){try{if(process.platform!=='win32'&&proc.pid)process.kill(-proc.pid,'SIGKILL');else proc.kill();}catch{}}err?reject(err):resolve(result);};
  const abort=()=>done(error('REFERENCE_CANCELLED','File preparation cancelled'));
  const timer=setTimeout(()=>done(error('REFERENCE_DOCUMENT_TIMEOUT','Document conversion timed out; no partial content was sent')),120000);
  // Only a trusted bundled decoder executes; filenames are never shell commands.
  const env=Object.fromEntries(['PATH','SystemRoot','WINDIR','TEMP','TMP','FONTCONFIG_PATH','LD_LIBRARY_PATH'].filter(k=>typeof process.env[k]==='string').map(k=>[k,process.env[k]]));
  Object.assign(env,{LANG:'C.UTF-8',LC_ALL:'C.UTF-8',VODER_INPUT_FILE_BYTES:String(LIMITS.fileBytes),VODER_INPUT_EXPANDED_BYTES:String(Number.isSafeInteger(expandedBytes)&&expandedBytes>=0?expandedBytes:LIMITS.expandedBytes),VODER_INPUT_MEMORY_BYTES:String(Math.min(LIMITS.memoryBytes,availableMemory()))});
  try{proc=spawn(decoderPython(),['-I','-B',path.join(__dirname,operation.startsWith('archive')?'archive-decode.py':'document-convert.py'),operation,ext(name),...(operation.startsWith('archive')?[destination||'',path.basename(name)]:[])],{shell:false,detached:process.platform!=='win32',windowsHide:true,stdio:['pipe','pipe','pipe'],env});}
  catch{done(error('REFERENCE_DOCUMENT_DEPENDENCY','Document decoder is unavailable'));return;}
  signal?.addEventListener('abort',abort,{once:true});
  proc.on('error',()=>done(error('REFERENCE_DOCUMENT_DEPENDENCY','Install the document runtime with the supplied setup command')));
  proc.stdin.on('error',()=>{});let diag=0;
  proc.stderr.on('data',b=>{diag+=b.length;if(diag>65536)done(error('REFERENCE_DOCUMENT_DECODE','Decoder diagnostic limit exceeded'));});
  proc.stdout.on('data',b=>{size+=b.length;if(size>Math.min(LIMITS.fileBytes*2,LIMITS.wireBytes*2))done(error('REFERENCE_LIMIT','Converted document exceeds request capacity'));else chunks.push(b);});
  proc.on('close',code=>{if(settled)return;try{const result=JSON.parse(Buffer.concat(chunks).toString('utf8'));if(result.error)done(error(result.error,result.error==='REFERENCE_ARCHIVE_RESOURCE'?'Archive exceeds inline parsing memory; use Voder: Extract Archive to stream extraction to disk, then select relevant members. No partial content was sent':'File decoding failed ('+result.error+'; '+(result.class||'decoder')+'); no partial content was sent'));else if(code!==0)done(error('REFERENCE_DOCUMENT_DECODE','Document decoder failed'));else done(null,result);}catch{done(error('REFERENCE_DOCUMENT_DECODE','Decoder produced invalid output'));}});
  proc.stdin.end(bytes);
 });
}
async function documentPdf(name,bytes,options){if(bytes.subarray(0,5).toString()==='%PDF-')return {pdf:bytes,structuredText:''};const result=await runDecoder('office',name,bytes,options);const pdf=Buffer.from(result.pdf||'','base64');if(pdf.subarray(0,5).toString()!=='%PDF-')throw error('REFERENCE_DOCUMENT_DECODE','Conversion did not produce a PDF');return {...result,pdf};}
module.exports={decoderPython,OFFICE,EXTRA_IMAGES,needsVisual,runDecoder,documentPdf};
