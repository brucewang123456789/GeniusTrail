'use strict';
/** Natural-language planning is data. Only the reviewed host scope authorizes writes. */
const fs=require('node:fs/promises'),path=require('node:path'),crypto=require('node:crypto');
const RESERVED=new Set(['.voder-task-check.py']);
function fail(message){throw Object.assign(Error(message),{code:'TASK_PLAN_INVALID'});}
function safePath(p){return typeof p==='string'&&p.length<=200&&!p.includes('..')&&!p.includes('\\')&&p.split('/').every(x=>/^[A-Za-z0-9_-][A-Za-z0-9._-]*$/.test(x)&&!x.endsWith('.')&&!/^(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)/i.test(x))&&!/(?:^|\/)(?:node_modules|venv|credentials(?:\.json)?|model-provider\.json)(?:\/|$)/i.test(p)&&!RESERVED.has(p);}
/** Extract one unambiguous JSON plan without repairing or inventing its scope. */
function extractPlanJson(text){
 if(typeof text!=='string'||Buffer.byteLength(text,'utf8')>131072)fail('Planner output absent or exceeds 128 KiB.');
 const found=new Map();
 const consider=(value,depth=0)=>{
  if(depth>3)return;
  if(typeof value==='string'){try{consider(JSON.parse(value),depth+1);}catch{}return;}
  if(!value||typeof value!=='object'||Array.isArray(value))return;
  if(value.action==='answer'||value.action==='execute'){found.set(JSON.stringify(value),value);return;}
  // Transport/report envelopes are decoded only as data, never executed.
  for(const key of ['plan','answer','draft'])if(Object.hasOwn(value,key))consider(value[key],depth+1);
 };
 const trimmed=text.replace(/^\uFEFF/,'').trim();
 try{consider(JSON.parse(trimmed));}catch{}
 // Balanced object scan respects braces inside escaped JSON strings. Never eval,
 // extract filenames from prose, repair truncated JSON, or accept competing plans.
 let start=-1,depth=0,quoted=false,escaped=false,objects=0;
 for(let i=0;i<trimmed.length;i++){
  const c=trimmed[i];
  if(start<0){if(c==='{'){start=i;depth=1;quoted=false;escaped=false;}continue;}
  if(quoted){if(escaped)escaped=false;else if(c==='\\')escaped=true;else if(c==='"')quoted=false;continue;}
  if(c==='"'){quoted=true;continue;}
  if(c==='{')depth++;
  if(c==='}'&&--depth===0){
   if(++objects>32)fail('Planner output contains too many JSON objects.');
   try{consider(JSON.parse(trimmed.slice(start,i+1)));}catch{}
   start=-1;
  }
 }
 if(found.size!==1)fail(found.size?'Planner returned competing plans; no scope was selected.':'Planner returned no complete JSON plan; no write execution started.');
 return [...found.values()][0];
}
function parsePlan(result){
 if(result?.status!=='DRAFT_UNVERIFIED'||result.verified!==false||typeof result.draft!=='string')fail('Task planning did not complete; no write execution started.');
 const plan=extractPlanJson(result.draft);
 if(!plan||!['answer','execute'].includes(plan.action)||typeof plan.answer!=='string'||plan.answer.length>12000)fail('Invalid planner action.');
 if(plan.action==='answer')return Object.freeze({action:'answer',answer:plan.answer});
 if(!Array.isArray(plan.files)||!plan.files.length||plan.files.length>16||plan.files.some(p=>!safePath(p))||new Set(plan.files).size!==plan.files.length)fail('Select 1..16 safe relative text source paths.');
 const files=[...plan.files].sort();
 if(files.some((p,i)=>files.some((q,j)=>i!==j&&q.startsWith(p+'/'))))fail('File/directory path conflict.');
 // The existing native candidate protocol is UTF-8 text. Binary output must be
 // produced by a later native-supported codec, never smuggled as text or claimed done.
 if(files.some(p=>/\.(?:png|jpe?g|gif|webp|pdf|zip|blend|glb|fbx|exe|dll|so)$/i.test(p)))fail('Native delivery does not yet support binary candidates. Select editable text sources and explicitly report binary-output limitations.');
 if(!Array.isArray(plan.checks)||plan.checks.length>8)fail('At most 8 explicitly reviewed sandbox commands.');
 const checks=plan.checks.map(argv=>{
  if(!Array.isArray(argv)||!argv.length||argv.length>24||argv.some(v=>typeof v!=='string'||!v||v.length>512||/[\0\r\n]/.test(v))||!['python3','node','sh','bash','git','cargo','go','gcc','g++','rustc'].includes(argv[0]))fail('Invalid sandbox command.');
  return Object.freeze([...argv]);
 });
 return Object.freeze({action:'execute',answer:plan.answer,files:Object.freeze(files),checks:Object.freeze(checks)});
}
function planningGoal(goal){return goal+'\n\n[VODER TASK PLANNING — READ ONLY PREPARATION, NOT TASK COMPLETION]\nDetermine the user intent from the ENTIRE request, without requiring magic phrases, existing Python files, a web project or animation. Use only relevant read/list/search tools if needed. Do not modify files, run programs, or claim the task completed in this planning phase. For a request to create, build, fix, edit, render or implement, return action execute with the concrete file paths to CREATE OR MODIFY INSIDE THE SELECTED TASK DIRECTORY. Existing files may be edited and absent files may be created after scope approval. Respect existing content and unrelated files. Plan any required text source types (SVG, HTML, CSS, JS, Python, OBJ, STL, JSON, Markdown, etc.); no forced filenames or framework. There is no arbitrary host shell. Exact optional argv checks will be reviewed and run ONLY inside the independent sandbox. No shell wrappers or encoded commands unless specifically required by the user. Native delivery currently accepts UTF-8 text candidates only: for requested optional binary previews plan editable source plus an honest verification record, exclude binary paths, and explicitly state the binary/visual limitation in answer. Never claim a missing renderer is an artwork defect or that syntax checks certify appearance. For a question or explicitly read-only request, answer it with action answer. Return JSON only: {"action":"execute","answer":"brief plan and real capability limitations","files":["relative/path"],"checks":[["python3","-B","test_example.py"]]} OR {"action":"answer","answer":"evidence-based response"}. A plan grants no write authority. Do not abandon an executable request merely because this preparation is read-only.\n[/VODER TASK PLANNING]';}
async function parsePlanWithDiagnostics(result,{storageRoot,workspace,goal}={}){
 try{return parsePlan(result);}catch(error){
  try{
   if(typeof storageRoot!=='string'||!path.isAbsolute(storageRoot)||await fs.realpath(storageRoot)!==storageRoot)throw Error('UNSAFE_DIAGNOSTIC_ROOT');
   const dir=path.join(storageRoot,'task-planning-diagnostics');
   await fs.mkdir(dir,{mode:0o700}).catch(e=>{if(e.code!=='EEXIST')throw e;});
   const stat=await fs.lstat(dir);
   if(!stat.isDirectory()||stat.isSymbolicLink()||(stat.mode&0o077)||(process.getuid&&stat.uid!==process.getuid()))throw Error('UNSAFE_DIAGNOSTIC_DIRECTORY');
   const draft=typeof result?.draft==='string'?result.draft:'';
   const record={schema:'voder.planner.failure.v1',captured_at:new Date().toISOString(),workspace,status:result?.status??null,verified:result?.verified??null,error_code:error.code??'TASK_PLAN_INVALID',reason:error.message,goal_sha256:crypto.createHash('sha256').update(String(goal||'')).digest('hex'),draft_bytes:Buffer.byteLength(draft),draft_sha256:crypto.createHash('sha256').update(draft).digest('hex'),draft_truncated:Buffer.byteLength(draft)>131072,draft:Buffer.from(draft).subarray(0,131072).toString('utf8')};
   const file=path.join(dir,'planner-'+crypto.randomUUID()+'.private.json'),handle=await fs.open(file,'wx',0o600);
   try{await handle.writeFile(JSON.stringify(record,null,2));await handle.sync();}finally{await handle.close();}
   error.message+=' Diagnostic: '+file;
  }catch(diagnosticError){error.message+=' Diagnostic capture failed ('+String(diagnosticError.code||'HOST_STORAGE_ERROR')+').';}
  throw error;
 }
}
module.exports={parsePlan,planningGoal,safePath,extractPlanJson,parsePlanWithDiagnostics};
