'use strict';
/** Passive language adapters. No generated program, package manager or shell is executed. */
const fs=require('node:fs/promises'),path=require('node:path'),crypto=require('node:crypto'),{spawn}=require('node:child_process');
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');
const ALLOWED=/\.(?:cjs|mjs|js|py|json|html|css|ts|tsx|jsx|go|rs|java|c|cc|cpp|h|toml|ya?ml|md)$|(?:^|\/)(?:package-lock\.json|requirements\.txt|Cargo\.lock|yarn\.lock|go\.mod|go\.sum)$/i;
const SECRET=/(^|\/)(?:\.git|node_modules|\.venv|venv|\.env(?:\..*)?|\.ssh|\.aws|\.kube|credentials(?:\.json)?|model-provider\.json|[^/]*private[^/]*key[^/]*)(\/|$)/i;
async function readScope(root,files){
 if(typeof root!=='string'||!path.isAbsolute(root)||await fs.realpath(root)!==root)throw Error('TOOLS_WORKSPACE_IDENTITY');
 if(!Array.isArray(files)||!files.length||files.length>48||new Set(files).size!==files.length)throw Error('TOOLS_FILE_SCOPE');
 const rows=[];
 for(const relative of files){
  if(typeof relative!=='string'||relative.length>240||relative.split('/').some(p=>!p||p==='.'||p==='..'||!/^[A-Za-z0-9._-]+$/.test(p))||SECRET.test(relative)||!ALLOWED.test(relative))throw Error('TOOLS_UNSAFE_FILE');
  let abs=root;for(const segment of relative.split('/')){abs=path.join(abs,segment);if((await fs.lstat(abs)).isSymbolicLink())throw Error('TOOLS_SYMLINK');}
  const s=await fs.lstat(abs);if(!s.isFile())throw Error('TOOLS_INPUT_BUDGET');
  const bytes=await fs.readFile(abs);if(bytes.includes(0))throw Error('TOOLS_BINARY_FILE');rows.push({relative,bytes,sha256:hash(bytes)});
 }
 return rows;
}
async function passiveProcess(program,args,input,{timeout=5000,maxBytes=12000}={}){
 return new Promise((resolve,reject)=>{
  const child=spawn(program,args,{env:{PATH:'/usr/local/bin:/usr/bin:/bin',LANG:'C.UTF-8'},stdio:['pipe','pipe','pipe'],shell:false});let output='',settled=false,reason=null;
  const stop=code=>{reason=code;child.kill('SIGKILL');};const timer=setTimeout(()=>stop('TIMEOUT'),timeout);
  for(const stream of [child.stdout,child.stderr])stream.on('data',chunk=>{if(Buffer.byteLength(output)+chunk.length>maxBytes){stop('OUTPUT_LIMIT');return;}output+=chunk.toString();});
  child.stdin.on('error',()=>{});child.stdin.end(input);child.on('error',e=>{clearTimeout(timer);if(!settled){settled=true;reject(e);}});
  child.on('close',(code,signal)=>{clearTimeout(timer);if(!settled){settled=true;resolve({status:reason?'UNAVAILABLE':code===0?'PASS':'FAIL',code:reason||code,diagnostic:output.slice(-maxBytes)});}});
 });
}
async function inspectLanguages(root,files,{node=process.execPath,python='/usr/bin/python3',run=passiveProcess}={}){
 const rows=await readScope(root,files),checks=await inspectSourceRows(rows,{node,python,run});
 // Recheck exact source bytes after subprocesses; observations cannot silently become stale.
 const now=await readScope(root,files);if(now.some((r,i)=>r.sha256!==rows[i].sha256))throw Error('TOOLS_SOURCE_CHANGED');
 return {schema:'voder.tools.language-report.v1',authority:'PASSIVE_PARSER_OBSERVATIONS_NOT_VGR_PROOF',workspace:root,checks,executed_user_code:false};
}
async function inspectSourceRows(rows,{node=process.execPath,python='/usr/bin/python3',run=passiveProcess}={}){
 if(!Array.isArray(rows)||rows.length>48||rows.some(r=>typeof r.relative!=='string'||!Buffer.isBuffer(r.bytes)))throw Error('TOOLS_INPUT_BUDGET');
 const checks=[];
 for(const row of rows){
  const text=row.bytes.toString('utf8');let r;
  try{
   if(/\.(?:cjs|mjs|js)$/.test(row.relative)){
    // stdin --check parses only. No imports, top-level code, scripts or loaders run.
    r=await run(node,['--check',/\.mjs$/.test(row.relative)?'--input-type=module':'--input-type=commonjs'],text);
    if(/\.js$/.test(row.relative)&&r.status==='FAIL'){const moduleCheck=await run(node,['--check','--input-type=module'],text);if(moduleCheck.status==='PASS')r={...moduleCheck,parsed_as:'ES module'};}
    // Syntax observation is not proof of package-mode compatibility, module resolution or execution.
   }else if(/\.py$/.test(row.relative))r=await run(python,['-I','-B','-c','import ast,sys; ast.parse(sys.stdin.read(), filename="scoped-source.py"); print("Python AST parsed; source was not executed")'],text);
   else if(/\.json$/.test(row.relative)){JSON.parse(text);r={status:'PASS',diagnostic:'JSON parsed'};}
   else r={status:'NOT_RUN',diagnostic:/\.(?:ts|tsx|jsx)$/.test(row.relative)?'TypeScript/JSX build requires a separately approved sandbox toolchain':'No executable validator selected for this language'};
  }catch(e){r={status:e.code==='ENOENT'?'UNAVAILABLE':'FAIL',diagnostic:String(e.message).slice(0,300)};}
  checks.push({file:row.relative,sha256:row.sha256,...r});
 }
 return checks;
}
function inspectDependencies(rows){
 const byName=new Map(rows.map(x=>[x.relative,x])),out=[];
 const pkg=byName.get('package.json');if(pkg){
  const p=JSON.parse(pkg.bytes.toString('utf8'));for(const section of ['dependencies','devDependencies','scripts'])if(p[section]!==undefined&&(p[section]===null||typeof p[section]!=='object'||Array.isArray(p[section])))throw Error('TOOLS_INVALID_PACKAGE_MANIFEST');
  const entries=Object.entries({...p.dependencies,...p.devDependencies});if(entries.length>1000)throw Error('TOOLS_DEPENDENCY_BUDGET');
  out.push({ecosystem:'npm',dependencies:entries.map(([name,version])=>({name,version:String(version).slice(0,200)})),scripts:Object.keys(p.scripts||{}),lockfile:['package-lock.json','npm-shrinkwrap.json','yarn.lock','pnpm-lock.yaml'].find(n=>byName.has(n))||null,installation:'NOT_RUN',warning:'Package scripts and dependency declarations are untrusted input; no host installation or service start is authorized by this report.'});
 }
 for(const [name,eco]of [['requirements.txt','pip'],['pyproject.toml','python'],['Cargo.toml','cargo'],['go.mod','go']])if(byName.has(name))out.push({ecosystem:eco,manifest:name,sha256:byName.get(name).sha256,installation:'NOT_RUN'});
 return {schema:'voder.tools.dependency-report.v1',authority:'DECLARATIVE_INVENTORY_ONLY',ecosystems:out,services_started:0};
}
module.exports={readScope,inspectLanguages,inspectDependencies,inspectSourceRows,passiveProcess,hash};
