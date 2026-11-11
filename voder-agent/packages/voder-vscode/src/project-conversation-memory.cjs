'use strict';
// Voder project-scoped chat history. All retrieved memory is untrusted user data.
const fs=require('node:fs/promises');
const path=require('node:path');
const crypto=require('node:crypto');
const {constants}=require('node:fs');
const sha=s=>crypto.createHash('sha256').update(s).digest('hex');
const random=()=>crypto.randomUUID();
const MAX_GOAL=16384,MAX_CONTEXT=24576,MAX_LOG=128*1024*1024,MAX_LOG_ENTRY=131072;
const SECRET=/(?:-----BEGIN [^-\n]{0,64}PRIVATE KEY-----|\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16})\b|\b(?:api[_ -]?key|access[_ -]?token|auth[_ -]?token|password|secret)[ \t]*[:=][ \t]*[^\s]{8,})/giu;
function sanitize(value,max=4000){return String(value??'').replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/g,'').replace(SECRET,'[REDACTED]').slice(0,max);}
function byteLimit(value,max){let s=sanitize(value,max);if(Buffer.byteLength(s,'utf8')<=max)return s;let lo=0,hi=s.length;while(lo<hi){let mid=Math.floor((lo+hi+1)/2);if(Buffer.byteLength(s.slice(0,mid),'utf8')<=max)lo=mid;else hi=mid-1;}return s.slice(0,lo).replace(/[\uD800-\uDBFF]$/,'');}
function terms(value){
 const s=String(value).toLowerCase();const out=new Set((s.match(/[a-z][\w-]{2,}/g)||[]).slice(0,100));
 for(const segment of s.match(/[\u3400-\u9fff]+/g)||[])for(let i=0;i+1<segment.length&&i<160;i++)out.add(segment.slice(i,i+2));
 return out;
}
function score(goal,row){const a=terms(goal),b=terms([row.goal,row.answer,row.content].filter(Boolean).join(' '));let n=0;for(const t of a)if(b.has(t))n++;return n;}

// Explicit, topic-free requests for recent project history only. This is a
// retrieval hint, never a policy override or authorization from remembered text.
function recentProjectRecap(goal){
 const text=String(goal??'').trim().replace(/[?？!！。.]+$/u,'').trim();
 const zh=text.replace(/\s+/g,'').replace(/^(?:请你|请|麻烦你|麻烦)/u,'').replace(/^(?:帮我|给我)/u,'');
 const past='(?:之前|此前|上次|最近|刚才|到目前为止|迄今)';
 const did='(?:做过|做了|完成过|完成了|讨论过|讨论了|进行过)';
 const what='(?:什么|些什么|哪些工作|哪些事|哪些事情|哪些任务)';
 if(new RegExp('^(?:回顾|总结|梳理|回忆)(?:一下)?(?:我们|咱们|我)?'+past+'?(?:都)?'+did+'(?:的)?'+what+'$','u').test(zh)||
    new RegExp('^(?:我们|咱们|我)'+past+'?(?:都)?'+did+'(?:的)?'+what+'$','u').test(zh)||
    new RegExp('^(?:回顾|总结|梳理|回忆)(?:一下)?(?:我们|咱们|我)?'+past+'(?:的)?(?:工作|进展|任务|对话|聊天|讨论|项目进展|历史)(?:记录)?$','u').test(zh)||
    /^(?:我们|咱们)?(?:上次|之前|此前)(?:的工作)?(?:做到哪(?:里|一步)?(?:了)?|进展到哪(?:里|一步)?(?:了)?)$/u.test(zh))return true;
 return /^(?:please\s+)?(?:recap|summari[sz]e|review)\s+(?:(?:our|the)\s+)?(?:previous|past|recent)\s+(?:work|progress|tasks|conversations?|discussions?)(?:\s+so\s+far)?$/i.test(text)||
  /^(?:please\s+)?(?:recap|summari[sz]e|review)\s+what\s+we(?:'ve|\s+have)?\s+(?:done|worked\s+on|discussed)(?:\s+(?:so\s+far|previously|last\s+time))?$/i.test(text)||
  /^(?:what\s+(?:did|have)\s+we\s+(?:do|done|work\s+on|discuss)(?:\s+(?:last\s+time|previously|so\s+far))?|where\s+did\s+we\s+leave\s+off)$/i.test(text);
}

function safeRelative(x){return typeof x==='string'&&x.length>0&&x.length<=240&&!x.includes('\\')&&!path.isAbsolute(x)&&!x.split('/').some(y=>!y||y==='.'||y==='..');}
function isReserved(root){const parts=root.split('/');return parts.includes('benchmark')||parts.includes('corpus')||parts.includes('repair')||parts.includes('.state');}
async function checkedDir(p,{create=false}={}){
 if(create)await fs.mkdir(p,{mode:0o700,recursive:false}).catch(e=>{if(e.code!=='EEXIST')throw e;});
 const st=await fs.lstat(p);if(!st.isDirectory()||st.isSymbolicLink()||(st.mode&0o022))throw Error('MEMORY_UNSAFE_DIRECTORY');
}
async function checkedFile(file,{optional=false,limit=MAX_LOG,allowPublic=false}={}){
 let st;try{st=await fs.lstat(file);}catch(e){if(optional&&e.code==='ENOENT')return null;throw e;}
 if(!st.isFile()||st.isSymbolicLink()||st.nlink!==1||(!allowPublic&&(st.mode&0o077))||st.size>limit)throw Error('MEMORY_UNSAFE_FILE');return st;
}
const {MemoryIndex,isCorrupt}=require('./project-memory-index.cjs');
const policyEngine=require('./project-memory-policy.cjs');
const portable=require('./project-memory-portable.cjs');
class ProjectConversationMemory {
 constructor(storageRoot,{segmentBytes=MAX_LOG}={}){
  if(!Number.isSafeInteger(segmentBytes)||segmentBytes<1024||segmentBytes>MAX_LOG)throw Error('MEMORY_SEGMENT_POLICY');
  this.storageRoot=path.resolve(storageRoot);this.windows=new Map();this.locks=new Map();this.segmentBytes=segmentBytes;
 }
 async locate(projectRoot){
  const root=await fs.realpath(projectRoot);if(!path.isAbsolute(root)||isReserved(root))throw Error('MEMORY_RESERVED_PROJECT');
  const base=path.join(this.storageRoot,'project-conversations'),hash=sha(root),dir=path.join(base,hash);
  await checkedDir(this.storageRoot);await checkedDir(base,{create:true});await checkedDir(dir,{create:true});
  return {root,hash,dir,file:path.join(dir,'history.jsonl'),exportRoot:path.join(this.storageRoot,'portable-memory-exports')};
 }
 async lock(key,operation){const prior=this.locks.get(key)||Promise.resolve();const current=prior.catch(()=>{}).then(operation);this.locks.set(key,current);try{return await current;}finally{if(this.locks.get(key)===current)this.locks.delete(key);}}
 async withDiskLock(p,operation){
  const lock=path.join(p.dir,'.history.lock');let acquired=false;
  for(let attempt=0;attempt<300;attempt++){
   try{await fs.mkdir(lock,{mode:0o700});acquired=true;break;}
   catch(e){if(e.code!=='EEXIST')throw e;}
   let st;try{st=await fs.lstat(lock);}catch(e){if(e.code==='ENOENT')continue;throw e;}
   if(!st.isDirectory()||st.isSymbolicLink()||(st.mode&0o077))throw Error('MEMORY_UNSAFE_LOCK');
   // Never steal an active lock. A crashed process is eligible after a grace period.
   if(Date.now()-st.mtimeMs>60000){
    let owner=null,missing=false;
    try{owner=JSON.parse(await fs.readFile(path.join(lock,'owner.json'),'utf8'));}
    catch(e){if(e.code==='ENOENT'||e instanceof SyntaxError)missing=true;else throw e;}
    let dead=missing;
    if(owner?.hostname===require('node:os').hostname()&&Number.isInteger(owner.pid)&&owner.pid>0){
     try{process.kill(owner.pid,0);}catch(err){if(err.code==='ESRCH')dead=true;}
    }
    if(dead){
     // The lstat/read/cleanup sequence is not atomic. Never remove a new owner
     // if the old lock disappeared or was replaced during stale-lock inspection.
     let still;try{still=await fs.lstat(lock);}catch(e){if(e.code==='ENOENT')continue;throw e;}
     if(still.dev!==st.dev||still.ino!==st.ino)continue;
     await fs.rm(lock,{recursive:true,force:true});continue;
    }
   }
   await new Promise(r=>setTimeout(r,100));
  }
  if(!acquired)throw Error('MEMORY_BUSY');
  const token=random();let ownerWritten=false;
  try{
   await fs.writeFile(path.join(lock,'owner.json'),JSON.stringify({pid:process.pid,hostname:require('node:os').hostname(),token}),{mode:0o600,flag:'wx'});
   ownerWritten=true;
   return await operation();
  }finally{
   // Never delete a different process's lock if recovery replaced ours.
   let ours=false;
   if(ownerWritten){try{const owner=JSON.parse(await fs.readFile(path.join(lock,'owner.json'),'utf8'));ours=owner.token===token;}
    catch(e){if(e.code!=='ENOENT'&&!(e instanceof SyntaxError))throw e;}}
   if(ours)await fs.rm(lock,{recursive:true,force:true});
  }
 }

 async getPolicy(projectRoot){
  const project=await this.locate(projectRoot);
  return this.lock(project.file,()=>this.withDiskLock(project,()=>this.readPolicy(project)));
 }
 async readPolicy(project){
  const file=path.join(project.dir,'memory-policy.json');
  const st=await checkedFile(file,{optional:true,limit:4096});
  if(!st)return policyEngine.empty();
  const fd=await fs.open(file,constants.O_RDONLY|(constants.O_NOFOLLOW||0));
  try{const actual=await fd.stat();if(actual.ino!==st.ino||actual.dev!==st.dev)throw Error('MEMORY_POLICY_CHANGED');
   let data;try{data=JSON.parse(await fd.readFile('utf8'));}catch{throw Error('MEMORY_POLICY_CORRUPT');}
   return policyEngine.validate(data);
  }finally{await fd.close();}
 }
 async setPolicy(projectRoot,newPolicy){
  const validated=policyEngine.validate(newPolicy),project=await this.locate(projectRoot);
  return this.lock(project.file,()=>this.withDiskLock(project,async()=>{
   const file=path.join(project.dir,'memory-policy.json');await checkedFile(file,{optional:true,limit:4096});
   const temp=path.join(project.dir,'.memory-policy-'+random()+'.tmp');
   try{
    const fd=await fs.open(temp,constants.O_CREAT|constants.O_EXCL|constants.O_WRONLY|(constants.O_NOFOLLOW||0),0o600);
    try{await fd.writeFile(JSON.stringify(validated,null,2)+'\n');await fd.sync();}finally{await fd.close();}
    await fs.rename(temp,file);
    const dir=await fs.open(project.dir,constants.O_RDONLY|constants.O_DIRECTORY);try{await dir.sync();}finally{await dir.close();}
   }finally{await fs.rm(temp,{force:true});}
   return validated;
  }));
 }
 async resetPolicy(projectRoot){return this.setPolicy(projectRoot,policyEngine.empty());}
 async append(project,row){return this.lock(project.file,()=>this.withDiskLock(project,async()=>{
  const line=JSON.stringify({schema:'voder.project.history.v2',project:project.hash,...row})+'\n';
  if(Buffer.byteLength(line)>MAX_LOG_ENTRY)throw Error('MEMORY_ENTRY_TOO_LARGE');
  let st=await checkedFile(project.file,{optional:true,limit:MAX_LOG+MAX_LOG_ENTRY});
  if(st&&st.size+Buffer.byteLength(line)>this.segmentBytes){
   const names=(await fs.readdir(project.dir)).filter(n=>/^archive-\d{8}\.jsonl$/.test(n)).sort();
   const next=names.length?Number(names[names.length-1].slice(8,16))+1:1;
   if(next>99999999)throw Error('MEMORY_ARCHIVE_SEQUENCE_EXHAUSTED');
   const archive=path.join(project.dir,`archive-${String(next).padStart(8,'0')}.jsonl`);
   await checkedFile(archive,{optional:true}).then(x=>{if(x)throw Error('MEMORY_ARCHIVE_COLLISION');});
   await fs.rename(project.file,archive);st=null;
  }
  // Preserve a damaged legacy trailing fragment, but never concatenate a new JSON row onto it.
  let separator='';
  if(st?.size){const fd=await fs.open(project.file,constants.O_RDONLY|(constants.O_NOFOLLOW||0));try{
   const b=Buffer.alloc(1);await fd.read(b,0,1,st.size-1);if(b[0]!==10)separator='\n';
  }finally{await fd.close();}}
  const fd=await fs.open(project.file,constants.O_APPEND|constants.O_CREAT|constants.O_WRONLY|(constants.O_NOFOLLOW||0),0o600);
  try{const check=await fd.stat();if(!check.isFile()||check.nlink!==1||(check.mode&0o077))throw Error('MEMORY_UNSAFE_FILE');
   if(check.size+Buffer.byteLength(separator+line)>MAX_LOG+MAX_LOG_ENTRY)throw Error('MEMORY_ENTRY_TOO_LARGE');
   await fd.writeFile(separator+line);await fd.sync();
  }finally{await fd.close();}
 }));}
 async records(project,{goal='',windowId='',all=false,limit=56,fallbackRecent=false}={}){
  return this.lock(project.file,()=>this.withDiskLock(project,async()=>{
   const index=new MemoryIndex(project,{terms,score,checkedFile});
   try{
    try{await index.open();await index.sync();}
    catch(e){
     if(!isCorrupt(e))throw e;
     await index.quarantineCorrupt();
     await index.open();await index.sync();
    }
    return index.find(goal,windowId,{all,limit,fallbackRecent});
   }finally{index.close();}
  }));
 }
 async forEachAllRecord(project,visitor){
  return this.lock(project.file,()=>this.withDiskLock(project,async()=>{
   const index=new MemoryIndex(project,{terms,score,checkedFile});
   try{
    try{await index.open();await index.sync();}
    catch(e){if(!isCorrupt(e))throw e;await index.quarantineCorrupt();await index.open();await index.sync();}
    for(const row of index.allRows())await visitor(row);
   }finally{index.close();}
  }));
 }
 async search(projectRoot,query,{limit=5}={}){
  if(typeof query!=='string'||!query.trim()||Buffer.byteLength(query)>MAX_GOAL)throw Error('MEMORY_SEARCH_QUERY');
  if(!Number.isSafeInteger(limit)||limit<1||limit>20)throw Error('MEMORY_SEARCH_LIMIT');
  const p=await this.locate(projectRoot),r=await this.records(p,{goal:query,limit});
  return {turns:r.turns.slice(-limit),imported:r.imported.slice(0,limit)};
 }
 window(view,root){let w=this.windows.get(view);if(!w||w.root!==root){w={root,id:random(),pending:null};this.windows.set(view,w);}return w;}
 compose(goal,w,rows,policy=policyEngine.empty(),{recentRecall=false}={}){
  if(policy.mode==='off')return '';
  const active=rows.turns.filter(x=>x.windowId===w.id&&x.status&&policyEngine.allowed(x,policy));
  const other=rows.turns.filter(x=>x.windowId!==w.id&&x.status&&policyEngine.allowed(x,policy));
  const ranked=other.map((x,i)=>({x,i,s:score(goal,x),priority:policyEngine.rank(x,policy)})).filter(r=>r.s>0||r.priority||recentRecall).sort((a,b)=>b.priority-a.priority||(recentRecall?b.i-a.i:b.s-a.s||b.i-a.i)).slice(0,Math.min(15,policy.limit)).map(r=>r.x).reverse();
  const imported=rows.imported.filter(x=>policyEngine.allowed(x,policy)).map((x,i)=>({x,i,s:score(goal,{goal:x.title,content:x.content}),priority:policyEngine.rank(x,policy)})).filter(r=>r.s>0||r.priority).sort((a,b)=>b.priority-a.priority||b.s-a.s||b.i-a.i).slice(0,Math.min(12,policy.limit)).map(r=>r.x);
  const entry=x=>{
   const files=Array.isArray(x.files)?x.files.filter(safeRelative).slice(0,10).join(', '):'';
   return `User request: ${sanitize(x.goal,1600)}\nObserved status: ${sanitize(x.status,80)}${x.answer?'\nAssistant result: '+sanitize(x.answer,1800):''}${files?'\nVerified/delivered paths (status-dependent): '+files:''}`;
  };
  const compact=x=>`[PROJECT HISTORY]\n${entry(x)}`;
  const long=(recentRecall?ranked.slice().reverse():ranked).map(compact);const imp=imported.map(x=>`[IMPORTED PORTABLE MEMORY — USER DATA]\n${sanitize(x.title,120)}: ${sanitize(x.content,1600)}`);
  const recent=active.slice(-16).map(x=>`[CURRENT CHAT — PRIOR TURN]\n${entry(x)}`);
  // The active conversation gets first claim on the finite context budget.
  const parts=[];let total=0;
  const fit=(items,budget)=>{for(const text of items.slice().reverse()){
   const size=Buffer.byteLength(text,'utf8')+2;
   if(size>budget-total)continue;parts.unshift(text);total+=size;
  }};
  fit(recent,Math.min(17000,Math.max(0,policy.maxBytes-500)));
  const older=active.slice(0,-16).slice(-40).map(x=>`[EARLIER CHAT SUMMARY]\n${sanitize(x.goal,500)} | ${sanitize(x.status,80)} | ${sanitize(x.answer,350)}`);
  const restBudget=policy.maxBytes-500;
  for(const text of [...imp,...long,...older]){
   const size=Buffer.byteLength(text,'utf8')+2;if(total+size>restBudget)continue;
   // Insert historical evidence before active chat but never as higher authority.
   parts.unshift(text);total+=size;
  }
  return parts.join('\n\n');
 }
 async begin(view,projectRoot,goal){
  if(typeof goal!=='string'||!goal.trim()||Buffer.byteLength(goal,'utf8')>MAX_GOAL)throw Error('MEMORY_GOAL_INVALID');
  const p=await this.locate(projectRoot),w=this.window(view,p.root);if(w.pending)throw Error('MEMORY_ACTIVE_TURN');
  let savedPolicy;
  try{savedPolicy=await this.getPolicy(p.root);}
  catch(e){if(!String(e.message||'').startsWith('MEMORY_POLICY_'))throw e;
   // Corrupt user policy is fail-closed for injection, but cannot block the coding task.
   savedPolicy={...policyEngine.empty(),mode:'off'};
  }
  const policy=policyEngine.effective(savedPolicy,goal);
  const retrievalGoal=policy.mode==='only'?policy.include.join(' '):[goal,...policy.include,...policy.prefer].join(' ');
  let history=policy.mode==='off'?{turns:[],imported:[]}:await this.records(p,{goal:retrievalGoal,windowId:w.id,limit:Math.max(56,policy.limit)});
  const recentRecall=policy.mode==='auto'&&recentProjectRecap(goal)&&!history.turns.some(x=>x.windowId===w.id&&x.status&&policyEngine.allowed(x,policy));
  if(recentRecall){
   // A prior recap can itself match the question. Explicitly select recent
   // completed turns instead of letting that match hide the original work.
   const recent=await this.records(p,{goal:'',windowId:w.id,limit:Math.max(56,policy.limit),fallbackRecent:true});
   history={...recent,turns:[...new Map(recent.turns.map(x=>[x.id,x])).values()].sort((a,b)=>a.seq-b.seq)};
  }
  const context=this.compose(retrievalGoal,w,history,policy,{recentRecall});
  const id=random();await this.append(p,{type:'begin',id,windowId:w.id,at:new Date().toISOString(),goal:sanitize(goal,MAX_GOAL)});
  w.pending=id;return context;
 }
 async finish(view,projectRoot,goal,result){
  const p=await this.locate(projectRoot),w=this.windows.get(view);if(!w||w.root!==p.root||!w.pending)return false;
  const status=sanitize(result?.status??'OUTCOME_UNKNOWN',96);
  const answer=byteLimit(result?.draft??result?.answer??result?.presentation?.title??'',64000);
  const files=(result?.taskAcceptance?.artifacts??[]).map(x=>x?.path).filter(safeRelative).slice(0,20);
  await this.append(p,{type:'finish',id:w.pending,windowId:w.id,status,answer,files,at:new Date().toISOString()});w.pending=null;return true;
 }
 async fail(view,projectRoot,code='OUTCOME_UNKNOWN'){
  const w=this.windows.get(view);if(!w?.pending)return false;
  return this.finish(view,projectRoot,'',{status:sanitize(code,96)});
 }
 async end(view){const w=this.windows.get(view);if(!w)return;this.windows.delete(view);
  try{const p=await this.locate(w.root);if(w.pending)await this.append(p,{type:'finish',id:w.pending,windowId:w.id,status:'OUTCOME_UNKNOWN',answer:'Unconfirmed interrupted task; do not assume it failed or succeeded',files:[]});
   await this.append(p,{type:'window_closed',windowId:w.id,at:new Date().toISOString()});}catch{/* persisted history remains, never prevent New Chat */}
 }
 async exportProject(projectRoot,request){
  const p=await this.locate(projectRoot),query=sanitize(request,2000);
  const format=portable.requestedFormat(request);
  const filters=policyEngine.exportFilters(query),all=filters.all;
  await checkedDir(p.exportRoot,{create:true});
  const profile='voder-project-'+sha(p.root).slice(0,12),paths=[];
  let pending=[],count=0;
  const prepare=arr=>{const main={schema:'voder.memory.bundle.v1',profile,entries:arr};return format==='md'?portable.renderMarkdown(main):JSON.stringify({...main,sha256:sha(JSON.stringify(main))},null,2)+'\n';};
  const flush=async()=>{
   if(!pending.length)return;
   const data=prepare(pending);
   const output=path.join(p.exportRoot,`Voder_Project_Memory_${new Date().toISOString().replace(/[:.]/g,'-')}_${random().slice(0,8)}.${format}`);
   const f=await fs.open(output,constants.O_WRONLY|constants.O_CREAT|constants.O_EXCL|(constants.O_NOFOLLOW||0),0o600);
   try{await f.writeFile(data);await f.sync();}finally{await f.close();}
   paths.push(output);pending=[];
  };
  const add=async x=>{
   if(!policyEngine.eligibleExport(x,filters))return;
   const entry={id:'record-'+(count+1).toString().padStart(4,'0'),kind:x.status==='IMPORTED'?'reference':x.status==='DELIVERED'?'decision':'open_item',scope:'project',
    title:byteLimit(x.goal||'Imported project memory',120)||'Project memory',
    content:byteLimit(`Status: ${x.status||'UNVERIFIED'}\nRequest: ${x.goal||''}\nResponse: ${x.answer||''}\nFiles: ${(Array.isArray(x.files)?x.files:[]).filter(safeRelative).join(', ')}`,4096),
    source_ref:byteLimit(x.id||'project-archive',150)};
   if(pending.length>=64||Buffer.byteLength(prepare([...pending,entry]))>131072)await flush();
   if(Buffer.byteLength(prepare([entry]))>131072)throw Error('MEMORY_ENTRY_TOO_LARGE');
   pending.push(entry);count++;
  };
  if(all||filters.since||filters.until)await this.forEachAllRecord(p,add);
  else{
   const queryText=filters.include.length?filters.include.join(' '):query;
   const rows=await this.records(p,{goal:queryText,limit:56,fallbackRecent:!filters.include.length});
   const candidates=[...rows.imported.map(x=>({goal:x.title||'',answer:x.content,status:'IMPORTED',id:x.id,at:x.at})),...rows.turns.filter(x=>x.status)];
   const selection=candidates.map((x,i)=>({x,i,s:score(queryText,x)})).sort((a,b)=>b.s-a.s||b.i-a.i);
   for(const x of selection.slice(0,56).map(r=>r.x))await add(x);
  }
  if(!count)throw Error('MEMORY_NOTHING_TO_EXPORT');
  await flush();
  return {path:paths[0],paths,entries:count,format,selection:'deterministic user filters applied; exported entries are retained records only'};
 }
 async importProject(projectRoot,file){
  const p=await this.locate(projectRoot);if(typeof file!=='string'||!path.isAbsolute(file))throw Error('MEMORY_IMPORT_PATH');
  // Reject symlinked parent directories as well as the final memory bundle.
  let cursor=path.parse(file).root;for(const part of path.relative(cursor,file).split(path.sep).filter(Boolean)){
   cursor=path.join(cursor,part);const segment=await fs.lstat(cursor);if(segment.isSymbolicLink())throw Error('MEMORY_UNSAFE_IMPORT_PATH');
  }
  const st=await checkedFile(file,{limit:131072,allowPublic:true});if(!st)throw Error('MEMORY_IMPORT_MISSING');
  const fd=await fs.open(file,constants.O_RDONLY|(constants.O_NOFOLLOW||0));let raw;
  try{const actual=await fd.stat();if(actual.ino!==st.ino||actual.dev!==st.dev)throw Error('MEMORY_IMPORT_CHANGED');raw=await fd.readFile('utf8');}finally{await fd.close();}
  const markdown=path.extname(file).toLowerCase()==='.md';
  let b;try{b=markdown?portable.parseMarkdown(raw):JSON.parse(raw);}catch(e){throw Error(markdown?'MEMORY_IMPORT_MARKDOWN: '+String(e.message||e):'MEMORY_IMPORT_JSON');}
  if(!b||b.schema!=='voder.memory.bundle.v1'||typeof b.profile!=='string'||!Array.isArray(b.entries)||!b.entries.length||b.entries.length>64)throw Error('MEMORY_IMPORT_SCHEMA');
  const main={schema:b.schema,profile:b.profile,entries:b.entries};
  // JSON keeps the original V3 checksum rule; Markdown is deliberately user-editable.
  if(!markdown&&b.sha256!==sha(JSON.stringify(main)))throw Error('MEMORY_IMPORT_CHECKSUM');
  const ids=new Set();for(const x of b.entries){
   if(!x||typeof x.id!=='string'||!/^[-a-z0-9]{1,64}$/.test(x.id)||ids.has(x.id)||!['user','project','session'].includes(x.scope)||!['preference','project','decision','constraint','open_item','reference'].includes(x.kind)||typeof x.title!=='string'||Buffer.byteLength(x.title)>120||typeof x.content!=='string'||Buffer.byteLength(x.content)>4096||SECRET.test(x.content))throw Error('MEMORY_IMPORT_CONTENT');
   SECRET.lastIndex=0;ids.add(x.id);
  }
  const entries=b.entries.map(x=>({id:sanitize(x.id,64),title:byteLimit(x.title,120),content:byteLimit(x.content,4096),kind:x.kind,scope:x.scope}));
  const windowId='import-'+random();
  for(const entry of entries)await this.append(p,{type:'import',windowId,entries:[entry],at:new Date().toISOString()});
  return {entries:entries.length,sha256:sha(raw)};
 }
}
require('./workspace-dialogue-memory.cjs').enhance(ProjectConversationMemory,{policyEngine,score});
module.exports={ProjectConversationMemory,sanitize,terms};
