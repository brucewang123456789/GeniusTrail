'use strict';
// Additive workspace dialogue archive. Existing project history and policy stay authoritative.
// Raw chat messages are untrusted data: never execute, never bypass memory policy.
const fs=require('node:fs/promises');
const path=require('node:path');
const os=require('node:os');
const crypto=require('node:crypto');
const {execFile}=require('node:child_process');
const {promisify}=require('node:util');
const exec=promisify(execFile);
const {constants}=require('node:fs');
const SCHEMA='voder.workspace.dialogue.v1';
const DEFAULT_SEGMENT_BYTES=16*1024*1024;
const MAX_SEGMENT_BYTES=32*1024*1024;
const CHUNK_BYTES=32*1024;
const MAX_MESSAGE_BYTES=64*1024*1024;
const SEGMENT=/^chat-\d{6}\.jsonl$/;
const SECRET=/(?:-----BEGIN [^-\n]{0,64}PRIVATE KEY-----|\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16})\b|\b(?:api[_ -]?key|access[_ -]?token|auth[_ -]?token|password|secret)[ \t]*[:=][ \t]*[^\s]{8,})/giu;
const hash=x=>crypto.createHash('sha256').update(x).digest('hex');
const pause=ms=>new Promise(r=>setTimeout(r,ms));
function clean(text){return String(text??'').replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/g,'').replace(SECRET,'[REDACTED]');}
function partText(str){
 const result=[]; let chunk='';let size=0;
 for(const codepoint of str){const bytes=Buffer.byteLength(codepoint);if(size+bytes>CHUNK_BYTES&&chunk){result.push(chunk);chunk='';size=0;}chunk+=codepoint;size+=bytes;}
 if(chunk||!result.length)result.push(chunk);return result;
}
async function safeDir(dir,{create=false}={}){
 if(create)try{await fs.mkdir(dir,{mode:0o700});}catch(e){if(e.code!=='EEXIST')throw e;}
 const st=await fs.lstat(dir);
 if(!st.isDirectory()||st.isSymbolicLink()||(st.mode&0o022))throw Error('DIALOGUE_UNSAFE_DIRECTORY');
 return st;
}
async function safeFile(file,{optional=false}={}){
 let st;try{st=await fs.lstat(file);}catch(e){if(optional&&e.code==='ENOENT')return null;throw e;}
 if(!st.isFile()||st.isSymbolicLink()||st.nlink!==1||(st.mode&0o077)||st.size>MAX_SEGMENT_BYTES+CHUNK_BYTES*2)throw Error('DIALOGUE_UNSAFE_FILE');
 return st;
}
function dialogueDir(project){return path.join(project.root,'.voder','memory','conversation');}
async function existingDir(project){
 const base=path.join(project.root,'.voder'),mid=path.join(base,'memory'),out=path.join(mid,'conversation');
 try{await safeDir(base);await safeDir(mid);await safeDir(out);return out;}
 catch(e){if(e.code==='ENOENT')return null;throw e;}
}
async function installIgnore(project){
 // Git metadata exclude is untracked and avoids moving the working tree's
 // verified diff or changing a tracked root .gitignore.
 let inside=false,exclude;
 try{
  const state=await exec('git',['-C',project.root,'rev-parse','--is-inside-work-tree'],{timeout:4000});
  inside=state.stdout.trim()==='true';
  if(inside){const r=await exec('git',['-C',project.root,'rev-parse','--git-path','info/exclude'],{timeout:4000});
   const value=r.stdout.trim();exclude=path.isAbsolute(value)?value:path.resolve(project.root,value);
   const top=(await exec('git',['-C',project.root,'rev-parse','--show-toplevel'],{timeout:4000})).stdout.trim();
   const rel=path.relative(await fs.realpath(top),project.root).replaceAll(path.sep,'/');
   if(rel.startsWith('..')||/[\[\]*?\\!#\r\n]/u.test(rel))throw Error('DIALOGUE_GIT_SCOPE_UNSAFE');
   project.gitExcludeRule='/'+(rel?rel+'/':'')+'.voder/memory/';}
 }catch(e){if(e.code!=='ENOENT'&&e.code!==128&&e.code!==1)throw e;}
 if(inside){
  const parent=path.dirname(exclude);const st=await fs.lstat(parent);if(!st.isDirectory()||st.isSymbolicLink())throw Error('DIALOGUE_GIT_EXCLUDE_UNSAFE');
  const existing=await fs.lstat(exclude).catch(e=>{if(e.code==='ENOENT')return null;throw e;});
  if(existing&&(!existing.isFile()||existing.isSymbolicLink()||existing.nlink!==1||existing.size>1e6))throw Error('DIALOGUE_GIT_EXCLUDE_UNSAFE');
  const content=existing?await fs.readFile(exclude,'utf8'):'';
  if(!content.split(/\r?\n/).some(x=>x.trim()===project.gitExcludeRule)){
   await fs.appendFile(exclude,(content&&!content.endsWith('\n')?'\n':'')+project.gitExcludeRule+'\n',{mode:0o600});
  }
  return;
 }
 // Without a Git repository, keep a scoped .gitignore for later git init.
 const ignore=path.join(project.root,'.voder','.gitignore');
 const st=await fs.lstat(ignore).catch(e=>{if(e.code==='ENOENT')return null;throw e;});
 if(!st)await fs.writeFile(ignore,'/memory/\n',{flag:'wx',mode:0o600});
 else{
  if(!st.isFile()||st.isSymbolicLink()||st.nlink!==1||st.size>65536)throw Error('DIALOGUE_GITIGNORE_UNSAFE');
  const content=await fs.readFile(ignore,'utf8');
  if(!content.split(/\r?\n/).some(x=>x.trim()==='/memory/'))await fs.appendFile(ignore,(content.endsWith('\n')?'':'\n')+'/memory/\n');
 }
}
async function prepareDir(project){
 await safeDir(project.root);
 let p=project.root;
 for(const name of ['.voder','memory','conversation']){p=path.join(p,name);await safeDir(p,{create:true});}
 await installIgnore(project);
 return p;
}
async function lock(dir,fn){
 const file=path.join(dir,'.dialogue.lock');const token=crypto.randomUUID();let held=false;
 for(let i=0;i<120;i++){
  try{
   const fd=await fs.open(file,constants.O_CREAT|constants.O_EXCL|constants.O_WRONLY|(constants.O_NOFOLLOW||0),0o600);
   try{await fd.writeFile(JSON.stringify({pid:process.pid,host:os.hostname(),token}));await fd.sync();}
   finally{await fd.close();}
   held=true;break;
  }catch(e){if(e.code!=='EEXIST')throw e;}
  const st=await safeFile(file);
  if(Date.now()-st.mtimeMs>120000){
   let owner;try{owner=JSON.parse(await fs.readFile(file,'utf8'));}catch{owner=null;}
   if(owner?.host===os.hostname()&&Number.isInteger(owner.pid)&&owner.pid>0){
    let alive=true;try{process.kill(owner.pid,0);}catch(e){if(e.code==='ESRCH')alive=false;}
    if(!alive){const next=await safeFile(file);if(next.dev===st.dev&&next.ino===st.ino){await fs.unlink(file);continue;}}
   }
  }
  await pause(50);
 }
 if(!held)throw Error('DIALOGUE_BUSY');
 try{return await fn();}
 finally{const owner=JSON.parse(await fs.readFile(file,'utf8').catch(()=>'null'));if(owner?.token===token)await fs.unlink(file);}
}
function recapIntent(q){const s=String(q||'').trim();if(s.length>240||/[`"“”]/.test(s))return false;
 return /^(?:所以)?(?:我们|你们|你|咱们)?(?:刚才|之前|前面|上次|最近)(?:在[^\n]{0,35})?(?:聊|说|讨论|做|完成|提到|讲)|^(?:帮我)?(?:回顾|回忆|总结|复盘)(?:一下)?(?:我们|一下|之前|刚才|最近|上次)|^what (?:did we|have we) (?:just |previously )?(?:discuss|do|talk)|^recap (?:our|the|what)/iu.test(s);
}
async function append(project,{role,turnId,windowId,text,status='',at=new Date().toISOString()},options={}){
 if(!['user','assistant'].includes(role)||typeof turnId!=='string'||!turnId||typeof windowId!=='string')throw Error('DIALOGUE_ROW_IDENTITY');
 const prepared=clean(text);if(Buffer.byteLength(prepared)>MAX_MESSAGE_BYTES)throw Error('DIALOGUE_MESSAGE_OVER_LIMIT');
 const segments=partText(prepared);const dir=await prepareDir(project),max=options.segmentBytes||DEFAULT_SEGMENT_BYTES;
 if(!Number.isSafeInteger(max)||max<1024||max>MAX_SEGMENT_BYTES)throw Error('DIALOGUE_SEGMENT_POLICY');
 const rows=segments.map((content,part)=>JSON.stringify({schema:SCHEMA,role,turnId,windowId,part,parts:segments.length,text:content,at,status:role==='assistant'?String(status).slice(0,80):''})+'\n');
 return lock(dir,async()=>{
  const files=(await fs.readdir(dir)).filter(x=>SEGMENT.test(x)).sort();
  let name=files.at(-1)||'chat-000001.jsonl';let number=Number(name.slice(5,11));let file=path.join(dir,name);
  let st=await safeFile(file,{optional:true}),size=st?.size||0;
  if(size){const read=await fs.open(file,constants.O_RDONLY|(constants.O_NOFOLLOW||0));let last=0;
   try{const b=Buffer.alloc(1);await read.read(b,0,1,size-1);last=b[0];}finally{await read.close();}
   if(last!==10){const write=await fs.open(file,constants.O_APPEND|constants.O_WRONLY|(constants.O_NOFOLLOW||0));
    try{await write.writeFile('\n');await write.sync();size++;}finally{await write.close();}}
  }
  for(const row of rows){const bytes=Buffer.byteLength(row);if(bytes>max)throw Error('DIALOGUE_SEGMENT_TOO_SMALL');
   if(size&&size+bytes>max){number++;if(number>999999)throw Error('DIALOGUE_SEGMENT_SEQUENCE_EXHAUSTED');
    name=`chat-${String(number).padStart(6,'0')}.jsonl`;file=path.join(dir,name);if(await safeFile(file,{optional:true}))throw Error('DIALOGUE_SEGMENT_COLLISION');size=0;}
   const f=await fs.open(file,constants.O_APPEND|constants.O_CREAT|constants.O_WRONLY|(constants.O_NOFOLLOW||0),0o600);
   try{const now=await f.stat();if(!now.isFile()||now.nlink!==1||(now.mode&0o077)||now.size!==size)throw Error('DIALOGUE_FILE_CHANGED');
    await f.writeFile(row);await f.sync();size+=bytes;}
   finally{await f.close();}
  }
  return {segments:rows.length,bytes:Buffer.byteLength(prepared)};
 });
}
function capBytes(s,max){const b=Buffer.from(String(s),'utf8');if(b.length<=max)return s;return b.subarray(0,max).toString('utf8').replace(/\uFFFD$/,'')+'… [excerpt; original remains in workspace archive]';}
function focusedExcerpt(text,query,max=3000){
 const raw=String(text||''),q=String(query||'').trim();
 if(Buffer.byteLength(raw)<=max)return raw;
 let location=0;
 const candidates=(q.match(/[a-z][\w-]{2,}|[\u3400-\u9fff]{2,}/giu)||[]).map(x=>x.toLowerCase());
 for(const word of candidates){const find=raw.toLowerCase().indexOf(word);if(find>=0){location=find;break;}}
 const around=raw.slice(Math.max(0,location-250));
 return (location>250?'… [earlier text in archive] ':'')+capBytes(around,max-70);
}

// An explicit archive-only request selects the protected memory API, never a
// general file tool. It does not grant any access to hidden workspace paths.
function archiveOnlyIntent(q){
 const s=String(q||'');
 if(s.length>16000)return false;
 if(/^(?:\s*)(?:修复|修改|实现|开发|创建|编写|重构|删除|检查代码)/u.test(s))return false;
 return /(?:仅|只|only)[^\n]{0,110}(?:依据|使用|读取|检索|from|using)[^\n]{0,200}(?:对话档案|原始对话|workspace dialogue|\.voder\/memory\/conversation|chat-\d{6}\.jsonl)/iu.test(s)
  || /(?:仅|只)[^\n]{0,100}(?:对话档案|原始对话)[^\n]{0,100}(?:回顾|复述|检索|查找)/u.test(s);
}
function recentRequested(q){return /(?:最近|刚才|上一|上次|recent|last)[^\n]{0,20}(?:\d+|[一二三四五六七八九十两]|几)?(?:轮|条|次|段|turn|message|conversation|对话)/iu.test(String(q));}
function archiveCount(q,limit){
 const m=/(?:最近|上次|last|recent)[^\n]{0,10}(\d{1,2}|[一二三四五六七八九十两])\s*(?:轮|条|次|段|turn|message)/iu.exec(String(q));
 const han={一:1,二:2,两:2,三:3,四:4,五:5,六:6,七:7,八:8,九:9,十:10};
 return m?Math.min(limit,Number(m[1])||han[m[1]]||limit):limit;
}
function enhanceIndex(MemoryIndex){
 const proto=MemoryIndex.prototype;
 if(proto.__voderDialogueIndexV1)return;
 Object.defineProperty(proto,'__voderDialogueIndexV1',{value:true});
 const open=proto.open;proto.open=async function(...args){await open.apply(this,args);this.db.exec(`
  CREATE TABLE IF NOT EXISTS workspace_dialogue(seq INTEGER PRIMARY KEY,turn_id TEXT NOT NULL,window_id TEXT NOT NULL,role TEXT NOT NULL,part INTEGER NOT NULL,parts INTEGER NOT NULL,content TEXT NOT NULL,at TEXT,status TEXT,source_file TEXT NOT NULL DEFAULT '');
  CREATE INDEX IF NOT EXISTS ix_workspace_dialogue_turn ON workspace_dialogue(turn_id,seq);
  CREATE VIRTUAL TABLE IF NOT EXISTS workspace_dialogue_fts USING fts5(words);`);
  const cols=this.db.prepare('PRAGMA table_info(workspace_dialogue)').all().map(x=>x.name);
  if(!cols.includes('source_file')){
   this.db.exec("ALTER TABLE workspace_dialogue ADD COLUMN source_file TEXT NOT NULL DEFAULT ''");
   // This SQLite index is derived and may safely be rebuilt from original JSONL
   // archives, preserving the old archive bytes and all project memory semantics.
   this.reset();
  }
  this.db.exec('CREATE INDEX IF NOT EXISTS ix_workspace_dialogue_source_seq ON workspace_dialogue(source_file,seq)');
 };
 const segments=proto.segments;proto.segments=async function(...args){const existing=await segments.apply(this,args),dir=await existingDir(this.project);
  if(!dir)return existing;const rows=(await fs.readdir(dir)).filter(x=>SEGMENT.test(x)).sort();for(const row of rows)await safeFile(path.join(dir,row));
  return [...existing,...rows.map(x=>'dialogue/'+x)];};
 const apply=proto.apply;proto.apply=function(x,segmentName){if(x?.schema!==SCHEMA)return apply.call(this,x);
  if(x.role!=='user'&&x.role!=='assistant')return;if(typeof x.turnId!=='string'||x.turnId.length>120||typeof x.windowId!=='string'||x.windowId.length>120||typeof x.text!=='string'||Buffer.byteLength(x.text)>CHUNK_BYTES||!Number.isInteger(x.part)||x.part<0||x.part>4096||!Number.isInteger(x.parts)||x.parts<1||x.parts>4096)return;
  const sourceFile=typeof segmentName==='string'&&/^dialogue\/chat-\d{6}\.jsonl$/.test(segmentName)?segmentName.slice(9):'';
  const d=this.db,r=d.prepare('INSERT INTO workspace_dialogue(turn_id,window_id,role,part,parts,content,at,status,source_file) VALUES(?,?,?,?,?,?,?,?,?)').run(x.turnId,x.windowId,x.role,x.part,x.parts,x.text,typeof x.at==='string'?x.at:'',typeof x.status==='string'?x.status:'',sourceFile);
  // Index the full archived chunk, not only the first 160 Chinese characters
  // used by the lightweight legacy keyword scorer.
  const allTerms=new Set((x.text.toLowerCase().match(/[a-z][\w-]{2,}/g)||[]).slice(0,5000));
  for(const section of x.text.toLowerCase().match(/[\u3400-\u9fff]+/g)||[]){for(let i=0;i+1<section.length;i++)allTerms.add(section.slice(i,i+2));}
  const tokens=[...allTerms].slice(0,12000).join(' ');
  d.prepare('INSERT INTO workspace_dialogue_fts(rowid,words) VALUES(?,?)').run(Number(r.lastInsertRowid),tokens);
 };
 const reset=proto.reset;proto.reset=function(){reset.call(this);this.db.exec('DELETE FROM workspace_dialogue;DELETE FROM workspace_dialogue_fts;');};
 const find=proto.find;proto.find=function(goal,windowId,opts={}){
  const result=find.call(this,goal,windowId,opts);if(opts.all)return {...result,dialogue:[]};
  const exclusive=archiveOnlyIntent(goal)||opts.archiveOnly===true,limit=Math.min(12,Math.max(1,opts.limit||12)),db=this.db,tokens=[...this.terms(goal)].filter(x=>x.length>=2&&x.length<=128).slice(0,32);
  const chosen=new Map();
  if(tokens.length){const query=tokens.map(x=>'"'+x.replace(/"/g,'""')+'"').join(' OR ');
   for(const row of db.prepare('SELECT d.turn_id,d.window_id,d.content,d.seq FROM workspace_dialogue_fts f JOIN workspace_dialogue d ON d.seq=f.rowid WHERE workspace_dialogue_fts MATCH ? AND d.window_id<>? ORDER BY bm25(workspace_dialogue_fts),d.seq DESC LIMIT 128').all(query,windowId||'')){
    const sc=Math.max(1,this.score(goal,{content:row.content}));const old=chosen.get(row.turn_id);if(!old||old.score<sc)chosen.set(row.turn_id,{score:sc,seq:row.seq});
   }
  }
  let ordered=[...chosen].sort((a,b)=>b[1].score-a[1].score||b[1].seq-a[1].seq).slice(0,limit).map(([id])=>id);
  // Recent recall is explicitly requested, not enabled for unrelated coding tasks.
  if((recapIntent(goal)||opts.archiveRecent===true)&&ordered.length<limit){
   for(const row of db.prepare(`SELECT turn_id,MAX(seq) AS n FROM workspace_dialogue WHERE window_id<>? GROUP BY turn_id ORDER BY n DESC LIMIT 24`).all(windowId||'')){
    if(!ordered.includes(row.turn_id))ordered.push(row.turn_id);if(ordered.length>=limit)break;
   }
  }
  if(opts.archiveRecent===true||(exclusive&&recentRequested(goal))){
   const take=archiveCount(goal,limit);
   ordered=db.prepare(`SELECT turn_id,MAX(seq) AS n FROM workspace_dialogue WHERE window_id<>? GROUP BY turn_id HAVING SUM(CASE WHEN role='user' THEN 1 ELSE 0 END)>0 AND SUM(CASE WHEN role='assistant' THEN 1 ELSE 0 END)>0 ORDER BY n DESC LIMIT ?`).all(windowId||'',take).map(x=>x.turn_id).reverse();
  }
  const dialogue=[];
  for(const id of ordered){const rows=db.prepare('SELECT * FROM workspace_dialogue WHERE turn_id=? ORDER BY seq LIMIT 4096').all(id);if(!rows.length)continue;
   const user=rows.filter(x=>x.role==='user').sort((a,b)=>a.part-b.part),assistant=rows.filter(x=>x.role==='assistant').sort((a,b)=>a.part-b.part);
   if(!user.length||!assistant.length)continue; // Exclude unfinished turns from completed-dialogue recall.
   const text=a=>{if(!a.length)return '';const expected=a[0].parts;if(a.length!==expected||a.some((x,i)=>x.part!==i))return '[Incomplete archived message]';return a.map(x=>x.content).join('');};
   const goalText=text(user),reply=text(assistant),status=assistant[0]?.status||'UNCONFIRMED';
   if(goalText==='[Incomplete archived message]'||reply==='[Incomplete archived message]')continue;
   const files=[...new Set(rows.map(x=>x.source_file).filter(x=>/^chat-\d{6}\.jsonl$/.test(x)))];
   // Precise provenance for indexable JSONL lines. Each valid indexed record
   // corresponds to one line of its source segment; this stays stable across
   // SQLite rebuilds and can span more than one physical segment.
   const sourcePositions=files.map(file=>{
    const ids=rows.filter(x=>x.source_file===file).map(x=>x.seq);
    const firstSeq=Math.min(...ids),lastSeq=Math.max(...ids);
    const firstLine=db.prepare('SELECT COUNT(*) AS n FROM workspace_dialogue WHERE source_file=? AND seq<=?').get(file,firstSeq).n;
    const lastLine=db.prepare('SELECT COUNT(*) AS n FROM workspace_dialogue WHERE source_file=? AND seq<=?').get(file,lastSeq).n;
    return {file,firstLine,lastLine};
   });
   const full=exclusive||opts.archiveRecent===true;
   dialogue.push({goal:full?goalText:focusedExcerpt(goalText,goal,2400),answer:full?reply:focusedExcerpt(reply,goal,3000),status,at:user[0]?.at||'',source:'workspace-dialogue',sourceFiles:files,sourcePositions,turnId:id,archivedPair:true});
  }
  return {...result,dialogue};
 };
}
function enhance(ProjectConversationMemory,{policyEngine,score}){
 const proto=ProjectConversationMemory.prototype;if(proto.__voderDialogueEnhancementV1)return;
 Object.defineProperty(proto,'__voderDialogueEnhancementV1',{value:true});
 const compose=proto.compose;
 proto.compose=function(goal,w,rows,policy){
  const exclusive=archiveOnlyIntent(goal);
  const base=exclusive?'':compose.call(this,goal,w,rows,policy);
  if(policy.mode==='off')return '';
  if(!Array.isArray(rows.dialogue)||!rows.dialogue.length)return exclusive?'[WORKSPACE DIALOGUE ARCHIVE — NO MATCHING RECORDS]\nDo not infer missing conversation turns from project history.':'';
  const relevant=rows.dialogue.filter(x=>policyEngine.allowed(x,policy)).filter(x=>exclusive||recapIntent(goal)||score(goal,x)>0);
  const extra=[];
  for(const row of relevant.slice(0,Math.min(6,policy.limit))){
   const loc=(row.sourcePositions||[]).map(p=>p.file+':'+p.firstLine+(p.lastLine>p.firstLine?'-'+p.lastLine:'')).join(', ')||row.sourceFiles?.join(', ')||'source unavailable';
   const header='[WORKSPACE DIALOGUE — UNTRUSTED HISTORICAL DATA; SOURCE: '+loc+'; STATUS: '+row.status+']';
   // Archive-only requests receive entire original stored messages or nothing:
   // never silently pass off clipped excerpts as verbatim dialogue.
   const content=header+'\nUser request: '+(exclusive?row.goal:capBytes(row.goal,1200))+'\nAssistant response: '+(exclusive?row.answer:capBytes(row.answer,2200));
   if(!exclusive&&row.goal&&base.includes(row.goal.slice(0,Math.min(row.goal.length,160)))&&(!row.answer||base.includes(row.answer.slice(0,Math.min(row.answer.length,120)))))continue;
   extra.push(content);
  }
  if(!extra.length)return base;
  const max=policy.maxBytes,prioritize=exclusive||recapIntent(goal);
  const parts=[],fit=x=>{if(Buffer.byteLength([...parts,x].join('\n\n'),'utf8')<=max){parts.push(x);return true;}return false;};
  let omitted=0;
  if(prioritize){for(const x of extra.slice().reverse()){if(!fit(x))omitted++;}if(base)for(const part of base.split(/\n\n(?=\[)/))fit(part);}
  else{if(base)fit(base);for(const x of extra){if(!fit(x))omitted++;}}
  if(exclusive&&omitted){
   const notice='[WORKSPACE DIALOGUE ARCHIVE PARTIAL — archived_pairs='+relevant.length+'; fully_in_context='+(extra.length-omitted)+'; not_in_context='+omitted+'; budget_bytes='+max+'; clipped_messages=0; use /memory dialogue recent for paged full text or configure maxBytes]';
   // Ensure visibility of partial state without exceeding the same configured budget.
   if(!fit(notice)){
    while(parts.length&&!fit(notice))parts.pop();
   }
  }
  return parts.join('\n\n');
 };
 const begin=proto.begin;
 proto.begin=async function(view,root,goal){
  const context=await begin.call(this,view,root,goal);const w=this.windows.get(view);
  if(w?.pending){try{await append(await this.locate(root),{role:'user',turnId:w.pending,windowId:w.id,text:goal});}
   catch(e){this.lastWorkspaceDialogueError=String(e?.message||e);console.warn('[Voder workspace dialogue] user archive skipped:',e?.code||this.lastWorkspaceDialogueError);}}
  return context;
 };
 const finish=proto.finish;
 proto.finish=async function(view,root,goal,result){
  const w=this.windows.get(view),id=w?.pending,windowId=w?.id;
  const answer=result?.draft??result?.answer??result?.presentation?.title??'';
  if(id){try{await append(await this.locate(root),{role:'assistant',turnId:id,windowId,text:answer,status:result?.status??'OUTCOME_UNKNOWN'});}
   catch(e){this.lastWorkspaceDialogueError=String(e?.message||e);console.warn('[Voder workspace dialogue] assistant archive skipped:',e?.code||this.lastWorkspaceDialogueError);}}
  return finish.call(this,view,root,goal,result);
 };
 const search=proto.search;
 proto.search=async function(projectRoot,query,options){
  const original=await search.call(this,projectRoot,query,options);
  const p=await this.locate(projectRoot),rows=await this.records(p,{goal:(options?.archiveRecent===true?'仅依据工作区对话档案回顾最近三轮':query),limit:options?.limit||5,archiveRecent:options?.archiveRecent===true,archiveOnly:options?.archiveOnly===true});
  const policy=policyEngine.effective(await this.getPolicy(projectRoot),query);
  return {...original,dialogue:(rows.dialogue||[]).filter(x=>policyEngine.allowed(x,policy)).slice(0,options?.limit||5)};
 };
}
function formatDialoguePage(rows,page=1,pageBytes=16000,continuation='/memory dialogue recent'){
 if(!Number.isSafeInteger(page)||page<1||page>100000)throw Error('DIALOGUE_INVALID_PAGE');
 const header='[受控对话档案 · 只读原文]\n已归档完整用户/助手配对: '+rows.length+' 条（DRAFT_UNVERIFIED 只表示回复未验证交付，不表示档案不完整）';
 const all=header+'\n\n'+(rows.length?rows.map((x,i)=>{
  const loc=(x.sourcePositions||[]).map(p=>p.file+':'+p.firstLine+(p.lastLine>p.firstLine?'-'+p.lastLine:'')).join(', ')||x.sourceFiles?.join(', ')||'来源未知';
  return '第 '+(i+1)+' 轮 | SOURCE='+loc+' | STATUS='+x.status+'\nUSER:\n'+String(x.goal||'')+'\nASSISTANT:\n'+String(x.answer||'');
 }).join('\n\n'):'没有匹配的已归档完整对话。');
 const buf=Buffer.from(all,'utf8');
 let start=0,end=0,pages=0,selection='';
 do{
  end=Math.min(buf.length,start+pageBytes);
  while(end<buf.length&&(buf[end]&0xc0)===0x80)end--;
  if(end<=start&&start<buf.length)throw Error('DIALOGUE_PAGING_INVALID');
  pages++;
  if(pages===page)selection=buf.subarray(start,end).toString('utf8');
  start=end;
 }while(start<buf.length);
 if(page>pages)return 'ARCHIVE_PAGE_OUT_OF_RANGE: page='+page+'; total_pages='+pages;
 return '[DIALOGUE PAGE '+page+'/'+pages+'; bytes='+buf.length+'; full_original_preserved=true; clipping=false]\n'+selection+(page<pages?'\n[CONTINUE: /memory dialogue recent page '+(page+1)+']':'');
}
function resolveSegmentPath(project,name){
 if(typeof name==='string'&&/^dialogue\/chat-\d{6}\.jsonl$/.test(name))return path.join(dialogueDir(project),name.slice(9));
 return path.join(project.dir,name);
}
module.exports={enhance,enhanceIndex,resolveSegmentPath,append,recapIntent,formatDialoguePage,SCHEMA,DEFAULT_SEGMENT_BYTES};
