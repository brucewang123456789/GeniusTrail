'use strict';
// Portable format bridge only. V3's JSONL, SQLite and recall path are unchanged.
// Imported Markdown is untrusted user data, never model/system instructions.
const SCHEMA='voder.memory.bundle.v1';
const MARKER='<!-- voder.memory.markdown.v2 -->';
function requestedFormat(request){
 const s=String(request??'');
 if(/(?:--format(?:=|\s+)json\b|--json\b|导出(?:为|成)?\s*json)/iu.test(s))return 'json';
 if(/(?:--format(?:=|\s+)(?:md|markdown)\b|--md\b|\bmarkdown\b|\.md\b|(?:导出|整理|保存)(?:为|成|到)?\s*MD\b)/iu.test(s))return 'md';
 if(/^\s*\/memory\s+export\b/i.test(s))return 'json'; // legacy contract
 return 'md'; // natural-language export: editable Markdown
}
function heading(s){return String(s??'').replace(/[\r\n]/g,' ').replace(/[`*_[\]<>#]/g,'').trim().slice(0,120)||'Untitled memory';}
function renderMarkdown(bundle){
 if(bundle?.schema!==SCHEMA||!Array.isArray(bundle.entries))throw Error('MEMORY_MARKDOWN_SCHEMA');
 const lines=[MARKER,'# Voder 工程记忆','',`项目来源：${bundle.profile}`,'',
   '这是可编辑、可移植的工程记忆文件。直接修改各条目下面的引用正文，然后在其他工作区导入这个 MD 文件即可。',''];
 for(let i=0;i<bundle.entries.length;i++){
  const e=bundle.entries[i];
  const meta={id:e.id,kind:e.kind,scope:e.scope};
  lines.push(`## ${i+1}. ${heading(e.title)}`,'',`<!-- voder.entry.meta ${JSON.stringify(meta)} -->`,'<!-- voder.entry.content -->');
  for(const line of String(e.content).split('\n'))lines.push('> '+line.replace(/\r/g,''));
  lines.push('<!-- voder.entry.end -->','');
 }
 return lines.join('\n')+'\n';
}
function parseOfficialMarkdown(raw){
 const lines=raw.replace(/\r\n/g,'\n').split('\n');let item=null,mode='outside';const entries=[];
 function commit(){
  if(!item)return;
  if(mode!=='done'||!item.meta)throw Error('MEMORY_MARKDOWN_INCOMPLETE_ENTRY');
  const content=item.lines.join('\n');
  entries.push({...item.meta,title:item.title,content});
  if(entries.length>64)throw Error('MEMORY_MARKDOWN_TOO_MANY_ENTRIES');
  item=null;mode='outside';
 }
 for(const line of lines){
  const match=/^##\s+(?:\d+\.\s+)?(.+?)\s*$/.exec(line);
  if(match){commit();item={title:match[1],meta:null,lines:[]};mode='before';continue;}
  if(!item)continue;
  if(mode==='before'&&line.startsWith('<!-- voder.entry.meta ')){
   const m=/^<!-- voder\.entry\.meta (\{.*\}) -->$/.exec(line);
   if(!m)throw Error('MEMORY_MARKDOWN_META');
   try{item.meta=JSON.parse(m[1]);}catch{throw Error('MEMORY_MARKDOWN_META_JSON');}
   mode='metadata';continue;
  }
  if(mode==='metadata'&&line==='<!-- voder.entry.content -->'){mode='content';continue;}
  if(mode==='content'){
   if(line==='<!-- voder.entry.end -->'){mode='done';continue;}
   if(!line.startsWith('> '))throw Error('MEMORY_MARKDOWN_CONTENT_FORMAT');
   item.lines.push(line.slice(2));continue;
  }
  if(line.trim()&&mode!=='done')throw Error('MEMORY_MARKDOWN_LAYOUT');
 }
 commit();
 if(!entries.length)throw Error('MEMORY_MARKDOWN_ENTRIES_MISSING');
 return {schema:SCHEMA,profile:'portable-markdown',entries};
}
function parsePlainMarkdown(raw){
 const sections=[];let current=null;
 for(const line of raw.replace(/\r\n/g,'\n').split('\n')){
  const m=/^##\s+(.+?)\s*$/.exec(line);
  if(m){if(current)sections.push(current);current={title:m[1].trim(),body:[]};continue;}
  if(current)current.body.push(line);
  else if(line.trim()&&!/^#\s/.test(line)){if(!sections.length)current={title:'Imported Markdown memory',body:[line]};else throw Error('MEMORY_MARKDOWN_LAYOUT');}
 }
 if(current)sections.push(current);
 const filtered=sections.map(s=>({...s,body:s.body.join('\n').trim()})).filter(s=>s.body);
 if(!filtered.length||filtered.length>64)throw Error('MEMORY_MARKDOWN_ENTRIES');
 return {schema:SCHEMA,profile:'user-markdown',entries:filtered.map((s,i)=>({id:`record-${String(i+1).padStart(4,'0')}`,kind:'reference',scope:'project',title:s.title,content:s.body}))};
}
function parseMarkdown(raw){
 if(typeof raw!=='string'||!raw.trim())throw Error('MEMORY_MARKDOWN_EMPTY');
 if(raw.includes(MARKER))return parseOfficialMarkdown(raw);
 if(raw.includes('<!-- voder.entry.meta '))throw Error('MEMORY_MARKDOWN_MARKER_REQUIRED');
 return parsePlainMarkdown(raw);
}
module.exports={requestedFormat,renderMarkdown,parseMarkdown};
