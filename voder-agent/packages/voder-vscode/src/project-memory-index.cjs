'use strict';
// Derived, rebuildable search index for append-only Voder project history.
// The JSONL segments remain the source of truth. No external npm dependencies.
const fs=require('node:fs/promises');
const path=require('node:path');
const {constants}=require('node:fs');
const {DatabaseSync}=require('node:sqlite');
const MAX_ENTRY=131072;
const EMPTY=()=>({turns:[],imported:[],closed:new Set()});
const isCorrupt=e=>['SQLITE_NOTADB','SQLITE_CORRUPT'].includes(e?.code)||/database disk image is malformed|file is not a database/i.test(e?.message||'');
const arch=/^archive-\d{8}\.jsonl$/;
const termText=(text,terms)=>[...terms(String(text??''))].slice(0,800).join(' ');
function matchQuery(text,terms){
 const list=[...terms(text)].slice(0,32).filter(t=>t.length>=2&&t.length<=128);
 return list.length?list.map(t=>'"'+t.replace(/"/g,'""')+'"').join(' OR '):null;
}
class MemoryIndex{
 constructor(project,{terms,score,checkedFile}){this.project=project;this.terms=terms;this.score=score;this.checkedFile=checkedFile;this.db=null;}
 async open(){
  const file=path.join(this.project.dir,'history-index.sqlite');
  // Existing DB must be private, single-linked and never a symlink.
  let st=await this.checkedFile(file,{optional:true,limit:Number.MAX_SAFE_INTEGER});
  if(!st){const f=await fs.open(file,constants.O_CREAT|constants.O_EXCL|constants.O_WRONLY|(constants.O_NOFOLLOW||0),0o600).catch(e=>{if(e.code==='EEXIST')return null;throw e;});await f?.close();st=await this.checkedFile(file,{limit:Number.MAX_SAFE_INTEGER});}
  let db;
  try{db=new DatabaseSync(file);db.exec('PRAGMA busy_timeout=10000; PRAGMA journal_mode=DELETE; PRAGMA synchronous=FULL;');
   db.exec(`CREATE TABLE IF NOT EXISTS segments(name TEXT PRIMARY KEY, dev TEXT NOT NULL, ino TEXT NOT NULL, offset INTEGER NOT NULL);
    CREATE TABLE IF NOT EXISTS turns(seq INTEGER PRIMARY KEY, id TEXT NOT NULL UNIQUE, window_id TEXT NOT NULL, goal TEXT NOT NULL, status TEXT, answer TEXT, files TEXT, at TEXT);
    CREATE INDEX IF NOT EXISTS ix_turn_window ON turns(window_id,seq);
    CREATE TABLE IF NOT EXISTS imports(seq INTEGER PRIMARY KEY, window_id TEXT NOT NULL, id TEXT, title TEXT NOT NULL, content TEXT NOT NULL, kind TEXT, scope TEXT, at TEXT);
    CREATE TABLE IF NOT EXISTS closed(window_id TEXT PRIMARY KEY);
    CREATE VIRTUAL TABLE IF NOT EXISTS turn_fts USING fts5(words);
    CREATE VIRTUAL TABLE IF NOT EXISTS import_fts USING fts5(words);`);
   // Index from the first patch lacked timestamps; rebuild derived data once on upgrade.
   const turnsColumns=db.prepare('PRAGMA table_info(turns)').all().map(x=>x.name);
   const importColumns=db.prepare('PRAGMA table_info(imports)').all().map(x=>x.name);
   if(!turnsColumns.includes('at')){db.exec('ALTER TABLE turns ADD COLUMN at TEXT');db.exec('DELETE FROM segments');db.exec('DELETE FROM turns');db.exec('DELETE FROM imports');db.exec('DELETE FROM closed');db.exec('DELETE FROM turn_fts');db.exec('DELETE FROM import_fts');}
   else if(!importColumns.includes('at')){db.exec('DELETE FROM segments');db.exec('DELETE FROM turns');db.exec('DELETE FROM imports');db.exec('DELETE FROM closed');db.exec('DELETE FROM turn_fts');db.exec('DELETE FROM import_fts');}
   if(!importColumns.includes('at'))db.exec('ALTER TABLE imports ADD COLUMN at TEXT');
   this.db=db;
  }catch(e){db?.close();throw e;}
 }
 close(){this.db?.close();this.db=null;}
 async quarantineCorrupt(){
  this.close();
  const file=path.join(this.project.dir,'history-index.sqlite');
  await this.checkedFile(file,{limit:Number.MAX_SAFE_INTEGER});
  const target=path.join(this.project.dir,`history-index.corrupt-${Date.now()}-${require('node:crypto').randomUUID().slice(0,8)}.sqlite`);
  await fs.rename(file,target);
  return target;
 }
 async segments(){
  const all=await fs.readdir(this.project.dir);const files=all.filter(x=>arch.test(x)).sort();
  if(all.includes('history.jsonl'))files.push('history.jsonl');
  return files;
 }
 apply(x){
  const db=this.db;
  if(x?.schema!=='voder.project.history.v2'||x.project!==this.project.hash||typeof x.windowId!=='string')return;
  if(x.type==='begin'&&typeof x.id==='string'&&typeof x.goal==='string'){
   const r=db.prepare('INSERT OR IGNORE INTO turns(id,window_id,goal,at) VALUES(?,?,?,?)').run(x.id,x.windowId,x.goal,typeof x.at==='string'?x.at:null);
   if(r.changes){const seq=Number(r.lastInsertRowid);db.prepare('INSERT INTO turn_fts(rowid,words) VALUES(?,?)').run(seq,termText(x.goal,this.terms));}
  }else if(x.type==='finish'&&typeof x.id==='string'){
   const t=db.prepare('SELECT seq,goal,status,window_id FROM turns WHERE id=?').get(x.id);
   if(!t||t.window_id!==x.windowId||t.status!==null)return;
   db.prepare('UPDATE turns SET status=?,answer=?,files=? WHERE seq=?').run(typeof x.status==='string'?x.status:'OUTCOME_UNKNOWN',typeof x.answer==='string'?x.answer:'',JSON.stringify(Array.isArray(x.files)?x.files:[]),t.seq);
   db.prepare('DELETE FROM turn_fts WHERE rowid=?').run(t.seq);
   db.prepare('INSERT INTO turn_fts(rowid,words) VALUES(?,?)').run(t.seq,termText(t.goal+' '+(x.answer||''),this.terms));
  }else if(x.type==='window_closed')db.prepare('INSERT OR IGNORE INTO closed(window_id) VALUES(?)').run(x.windowId);
  else if(x.type==='import'&&Array.isArray(x.entries))for(const e of x.entries){
   if(!e||typeof e.content!=='string'||e.content.length>4096)continue;
   const r=db.prepare('INSERT INTO imports(window_id,id,title,content,kind,scope,at) VALUES(?,?,?,?,?,?,?)').run(x.windowId,typeof e.id==='string'?e.id:'',typeof e.title==='string'?e.title:'',e.content,e.kind||'',e.scope||'',typeof x.at==='string'?x.at:null);
   db.prepare('INSERT INTO import_fts(rowid,words) VALUES(?,?)').run(Number(r.lastInsertRowid),termText((e.title||'')+' '+e.content,this.terms));
  }
 }
 reset(){this.db.exec('DELETE FROM segments;DELETE FROM turns;DELETE FROM imports;DELETE FROM closed;DELETE FROM turn_fts;DELETE FROM import_fts;');}
 async sync(){
  const db=this.db;const files=await this.segments();
  const known=db.prepare('SELECT name,dev,ino,offset FROM segments').all();
  const info=[];
  for(const name of files){
   const file=require('./workspace-dialogue-memory.cjs').resolveSegmentPath(this.project,name),st=await this.checkedFile(file,{limit:128*1024*1024+MAX_ENTRY});
   info.push({name,file,st});
  }
  const existing=new Map(info.map(x=>[x.name,x]));
  // Rotation renames history.jsonl to a new archive. Move its index cursor instead of
  // treating an ordinary rotation as a reason to replay every archive from byte zero.
  const active=known.find(x=>x.name==='history.jsonl');
  if(active){
   const old=existing.get('history.jsonl');
   if(!old||old.st.ino.toString()!==active.ino||old.st.dev.toString()!==active.dev){
    const rotated=info.find(x=>arch.test(x.name)&&String(x.st.dev)===active.dev&&String(x.st.ino)===active.ino);
    if(rotated&&!known.some(x=>x.name===rotated.name)){
     db.prepare('UPDATE segments SET name=? WHERE name=?').run(rotated.name,'history.jsonl');
     active.name=rotated.name;
    }
   }
  }
  const invalid=known.some(k=>{
   const f=existing.get(k.name);return !f||k.dev!==String(f.st.dev)||k.ino!==String(f.st.ino)||k.offset>f.st.size;
  });
  if(invalid)db.exec('BEGIN IMMEDIATE');
  try{if(invalid){this.reset();db.exec('COMMIT');}}catch(e){db.exec('ROLLBACK');throw e;}
  for(const {name,file,st} of info){
   const record=db.prepare('SELECT offset FROM segments WHERE name=?').get(name);
   const initial=record?.offset??0;if(st.size<=initial)continue;
   const fd=await fs.open(file,constants.O_RDONLY|(constants.O_NOFOLLOW||0));
   let processed=initial,buffer=Buffer.alloc(0),began=false;
   try{
    const now=await fd.stat();if(now.dev!==st.dev||now.ino!==st.ino)throw Error('MEMORY_SOURCE_CHANGED');
    db.exec('BEGIN IMMEDIATE');began=true;
    for await(const chunk of fd.createReadStream({start:initial,autoClose:false,highWaterMark:65536})){
     buffer=buffer.length?Buffer.concat([buffer,chunk]):chunk;
     let nl;
     while((nl=buffer.indexOf(10))!==-1){
      const raw=buffer.subarray(0,nl);processed+=nl+1;
      if(raw.length&&raw.length<=MAX_ENTRY){try{this.apply(JSON.parse(raw.toString('utf8')),name);}catch(e){if(!(e instanceof SyntaxError))throw e;}}
      buffer=buffer.subarray(nl+1);
     }
     if(buffer.length>MAX_ENTRY)throw Error('MEMORY_OVERSIZE_RECORD');
    }
    db.prepare('INSERT INTO segments(name,dev,ino,offset) VALUES(?,?,?,?) ON CONFLICT(name) DO UPDATE SET dev=excluded.dev,ino=excluded.ino,offset=excluded.offset').run(name,String(st.dev),String(st.ino),processed);
    db.exec('COMMIT');began=false;
   }catch(e){if(began)db.exec('ROLLBACK');throw e;}
   finally{await fd.close();}
  }
 }
 turn(x){return {...x,windowId:x.window_id,files:JSON.parse(x.files||'[]')};}
 *allRows(){
  // Explicit full-history export iterates SQLite cursors, without loading all turns.
  for(const x of this.db.prepare('SELECT * FROM imports ORDER BY seq').iterate())yield {goal:x.title||'',answer:x.content,status:'IMPORTED',id:x.id||''};
  for(const x of this.db.prepare('SELECT * FROM turns WHERE status IS NOT NULL ORDER BY seq').iterate())yield this.turn(x);
 }
 find(goal,windowId,{all=false,limit=56,fallbackRecent=false}={}){
  const db=this.db;
  if(all){
   return {turns:db.prepare('SELECT * FROM turns WHERE status IS NOT NULL ORDER BY seq').all().map(x=>this.turn(x)),imported:db.prepare('SELECT * FROM imports ORDER BY seq').all(),closed:new Set()};
  }
  const recent=windowId?db.prepare('SELECT * FROM turns WHERE window_id=? AND status IS NOT NULL ORDER BY seq DESC LIMIT 56').all(windowId).reverse().map(x=>this.turn(x)):[];
  const query=matchQuery(goal,this.terms),turns=[],imports=[];
  if(query){
   const candidates=db.prepare('SELECT t.* FROM turn_fts f JOIN turns t ON t.seq=f.rowid WHERE turn_fts MATCH ? AND t.status IS NOT NULL AND t.window_id<>? ORDER BY bm25(turn_fts), t.seq DESC LIMIT 512').all(query,windowId);
   turns.push(...candidates.map(x=>({x:this.turn(x),s:this.score(goal,this.turn(x))})).filter(x=>x.s>0).sort((a,b)=>b.s-a.s).slice(0,limit).map(x=>x.x).reverse());
   const entries=db.prepare('SELECT i.* FROM import_fts f JOIN imports i ON i.seq=f.rowid WHERE import_fts MATCH ? ORDER BY bm25(import_fts), i.seq DESC LIMIT 256').all(query);
   imports.push(...entries.map(x=>({x,s:this.score(goal,{content:x.content,goal:x.title})})).filter(x=>x.s>0).sort((a,b)=>b.s-a.s).slice(0,Math.min(12,limit)).map(x=>x.x));
  }
  if(fallbackRecent&&!turns.length&&!imports.length){
   turns.push(...db.prepare('SELECT * FROM turns WHERE status IS NOT NULL ORDER BY seq DESC LIMIT ?').all(limit).map(x=>this.turn(x)));
   imports.push(...db.prepare('SELECT * FROM imports ORDER BY seq DESC LIMIT 12').all());
  }
  return {turns:[...turns,...recent],imported:imports,closed:new Set()};
 }
}
require('./workspace-dialogue-memory.cjs').enhanceIndex(MemoryIndex);
module.exports={MemoryIndex,EMPTY,isCorrupt};
