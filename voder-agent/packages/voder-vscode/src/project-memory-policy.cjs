'use strict';
// Deterministic user-controlled memory selection. This parser NEVER executes instructions
// found in stored memories, and a natural-language directive never grants tool access.
const POLICY_SCHEMA='voder.memory.policy.v1';
const MAX_RULE=120,MAX_RULES=8;
const empty=()=>({schema:POLICY_SCHEMA,mode:'auto',include:[],exclude:[],prefer:[],limit:12,maxBytes:24576});
function clean(text){
 if(typeof text!=='string')throw Error('MEMORY_POLICY_VALUE');
 const value=text.trim().replace(/^['"]|['"]$/g,'').replace(/\s+/g,' ');
 if(!value||Buffer.byteLength(value,'utf8')>MAX_RULE||/[\u0000-\u001f<>]/.test(value))throw Error('MEMORY_POLICY_VALUE');
 return value;
}
function list(value){
 if(Array.isArray(value)){if(value.length>MAX_RULES)throw Error('MEMORY_POLICY_COUNT');return [...new Set(value.map(clean))];}
 if(typeof value!=='string')throw Error('MEMORY_POLICY_VALUE');
 const parts=value.split('|').filter(x=>x.trim());if(parts.length>MAX_RULES)throw Error('MEMORY_POLICY_COUNT');
 return [...new Set(parts.map(clean))];
}
function validate(input){
 if(!input||typeof input!=='object'||Array.isArray(input)||input.schema!==POLICY_SCHEMA)throw Error('MEMORY_POLICY_SCHEMA');
 if(!['auto','off','only'].includes(input.mode))throw Error('MEMORY_POLICY_MODE');
 if(!Number.isSafeInteger(input.limit)||input.limit<1||input.limit>20)throw Error('MEMORY_POLICY_LIMIT');
 if(!Number.isSafeInteger(input.maxBytes)||input.maxBytes<1024||input.maxBytes>131072)throw Error('MEMORY_POLICY_BUDGET');
 const result={schema:POLICY_SCHEMA,mode:input.mode,include:list(input.include),exclude:list(input.exclude),prefer:list(input.prefer),limit:input.limit,maxBytes:input.maxBytes};
 if(result.mode==='only'&&!result.include.length)throw Error('MEMORY_POLICY_ONLY_EMPTY');
 return result;
}
function setFromText(rest){
 const policy=empty();const seen=new Set();
 for(const phrase of rest.split(';').filter(x=>x.trim())){
  const match=/^\s*(mode|include|exclude|prefer|limit|maxBytes)\s*=\s*(.*?)\s*$/i.exec(phrase);
  if(!match)throw Error('MEMORY_POLICY_SYNTAX');
  const key=match[1]==='maxbytes'?'maxBytes':match[1];if(seen.has(key))throw Error('MEMORY_POLICY_DUPLICATE');seen.add(key);
  policy[key]=['limit','maxBytes'].includes(key)?Number(match[2]):['include','exclude','prefer'].includes(key)?list(match[2]):match[2].toLowerCase();
 }
 if(!seen.size)throw Error('MEMORY_POLICY_SYNTAX');
 return validate(policy);
}
function command(goal){
 if(typeof goal!=='string')return null;
 const s=goal.trim();let m;
 if((m=/^\/memory\s+policy\s+(show|reset)\s*$/i.exec(s)))return {kind:'policy_'+m[1].toLowerCase()};
 if((m=/^\/memory\s+policy\s+set\s+([^\n]{1,1600})$/i.exec(s)))return {kind:'policy_set',policy:setFromText(m[1])};
 if((m=/^(?:以后|今后)(?:在)?这个项目(?:的记忆)?优先检索\s*([^，。；;\n]{1,120})(?:[，。；;].*)?$/u.exec(s)))return {kind:'policy_set',policy:validate({...empty(),prefer:[clean(m[1])]})};
 if((m=/^\/memory\s+search\s+([^\n]{1,2000})$/i.exec(s)))return {kind:'search',query:cleanSearch(m[1])};
 if(/^\/memory\s+inject\b/i.test(s))throw Error('MEMORY_POLICY_SYNTAX');
 return null;
}
function cleanSearch(value){const x=String(value||'').trim();if(!x||Buffer.byteLength(x)>2000)throw Error('MEMORY_SEARCH_QUERY');return x;}
function perTurn(goal){
 const result={mode:null,include:[],exclude:[],prefer:[]};if(typeof goal!=='string')return result;
 let match;
 if(/(?:本次|这次|当前任务)[^\n]{0,70}(?:不注入任何(?:历史|工程)?记忆|禁用(?:历史|工程)?记忆|不使用(?:历史|工程)?记忆)/u.test(goal)||/\b(?:this task|this turn)\b[^\n]{0,70}\b(?:no memory|disable memory)\b/i.test(goal))result.mode='off';
 if((match=/(?:本次任务|本次|这次|当前任务)[^\n，。；;]{0,18}只(?:使用|读取|注入|检索)\s*([^\n，。；;]{1,120}?)\s*(?:相关)?(?:的)?(?:工程|历史|项目)?记忆/u.exec(goal))){
  result.include=[clean(match[1].replace(/(?:相关|的)$/u,''))];result.mode='only';
 }
 if((match=/(?:不要|禁止|排除|不再)(?:向上下文)?(?:注入|使用|检索|包含)?\s*([^\n，。；;]{1,120}?)\s*(?:相关)?(?:的)?(?:工程记忆|项目记忆|历史记忆|历史|记忆)/u.exec(goal)))result.exclude=[clean(match[1].replace(/(?:相关|的)$/u,''))];
 if((match=/\bmemory\s*:\s*only\s+([^;\n]{1,120})/i.exec(goal))){result.include=[clean(match[1])];result.mode='only';}
 if((match=/\bmemory\s*:\s*exclude\s+([^;\n]{1,120})/i.exec(goal)))result.exclude=[clean(match[1])];
 if(/\bmemory\s*:\s*off\b/i.test(goal))result.mode='off';
 return result;
}
function effective(saved,goal){
 const base=validate(saved||empty()),task=perTurn(goal);
 const policy={...base,include:task.include.length?task.include:base.include,exclude:[...new Set([...base.exclude,...task.exclude])],prefer:[...new Set([...base.prefer,...task.prefer])],mode:task.mode||base.mode};
 return validate(policy);
}
function match(row,needle){
 const text=[row.goal,row.title,row.answer,row.content,(row.files||[]).join(' ')].filter(x=>typeof x==='string').join(' ').toLocaleLowerCase();
 // Exact phrase containment: no fuzzy false positives for policy exclusions.
 return text.includes(needle.toLocaleLowerCase());
}
function allowed(row,p){return p.mode!=='off'&&!(p.exclude.some(q=>match(row,q)))&&(p.mode!=='only'||p.include.some(q=>match(row,q)));}
function rank(row,p){return p.prefer.some(q=>match(row,q))?1:0;}
function exportFilters(text,{now=new Date()}={}){
 const s=String(text??'');const opts={all:/(?:\ball\b|全部|所有|完整|full history)/i.test(s),since:null,until:null,include:[],exclude:[],statuses:[]};let m;
 if((m=/(?:--since\s+|自\s*)(\d{4}-\d{2}-\d{2})/i.exec(s)))opts.since=dateUTC(m[1]);
 else if((m=/(?:最近|过去)\s*(\d{1,3}|[一二三四五六七八九十两]{1,3})\s*(天|日|个月|月|年)/u.exec(s))){const zh={一:1,二:2,三:3,四:4,五:5,六:6,七:7,八:8,九:9,十:10,两:2};const n=/^\d+$/.test(m[1])?Number(m[1]):m[1]==='十'?10:m[1].startsWith('十')?10+(zh[m[1][1]]||0):m[1].endsWith('十')?(zh[m[1][0]]||0)*10:zh[m[1]];if(n>120)throw Error('MEMORY_EXPORT_TIME_RANGE');const d=new Date(now);if(m[2]==='年')d.setUTCFullYear(d.getUTCFullYear()-n);else if(m[2].includes('月'))d.setUTCMonth(d.getUTCMonth()-n);else d.setUTCDate(d.getUTCDate()-n);opts.since=d.toISOString();}
 if((m=/--until\s+(\d{4}-\d{2}-\d{2})/i.exec(s)))opts.until=dateUTC(m[1],true);
 if((m=/--include\s+([^;\n]+?)(?=\s+--|;|$)/i.exec(s)))opts.include=list(m[1]);
 else if((m=/关于\s*([^，。；;\n]{1,100}?)\s*(?:的)?(?:全部|所有|完整)?(?:工程|历史|项目)?记忆/u.exec(s)))opts.include=[clean(m[1].replace(/的$/u,''))];
 if((m=/--exclude\s+([^;\n]+?)(?=\s+--|;|$)/i.exec(s)))opts.exclude=list(m[1]);
 else if((m=/(?:不含|排除|不要包含)\s*([^，。；;\n]{1,100}?)\s*(?:相关)?(?:的)?(?:记忆|历史|内容)/u.exec(s)))opts.exclude=[clean(m[1])];
 if((m=/--status\s+([A-Z_]+)/i.exec(s)))opts.statuses=[m[1]];
 return opts;
}
function dateUTC(x,end=false){const d=new Date(x+'T00:00:00.000Z');if(!Number.isFinite(d.getTime())||d.toISOString().slice(0,10)!==x)throw Error('MEMORY_EXPORT_DATE');if(end)d.setUTCDate(d.getUTCDate()+1);return d.toISOString();}
function eligibleExport(row,filters){
 if(filters.include.length&&!filters.include.some(x=>match(row,x)))return false;
 if(filters.exclude.some(x=>match(row,x)))return false;
 if(filters.statuses.length&&!filters.statuses.includes(row.status))return false;
 if(filters.since||filters.until){if(typeof row.at!=='string'||!Number.isFinite(Date.parse(row.at)))return false;const t=Date.parse(row.at);if(filters.since&&t<Date.parse(filters.since))return false;if(filters.until&&t>=Date.parse(filters.until))return false;}
 return true;
}
module.exports={POLICY_SCHEMA,empty,validate,command,perTurn,effective,allowed,rank,exportFilters,eligibleExport};
