'use strict';
/** Read-only, opt-in PDF vision: bounded renderer + OCR fallback, no credentials or shell. */
const {spawn}=require('node:child_process');
const {randomUUID}=require('node:crypto');
const {digest,imageMime,LIMITS}=require('./reference-contract.cjs');
const err=(code,message)=>Object.assign(Error(message),{code});
const parserEnv=()=>({...Object.fromEntries(['PATH','SystemRoot','WINDIR','TEMP','TMP'].filter(k=>typeof process.env[k]==='string').map(k=>[k,process.env[k]])),LANG:'C.UTF-8',LC_ALL:'C'});
function run(program,args,input,{signal,timeoutMs=45000,maxBytes=4*1024*1024}={}){
 return new Promise((resolve,reject)=>{
  if(signal?.aborted){reject(err('REFERENCE_CANCELLED','PDF operation cancelled'));return;}
  let settled=false,proc,buffer=[],size=0,stderrSize=0;
  const done=(e,result)=>{if(settled)return;settled=true;clearTimeout(timer);signal?.removeEventListener('abort',abort);if(e)proc?.kill();e?reject(e):resolve(result);};
  const abort=()=>done(err('REFERENCE_CANCELLED','PDF operation cancelled'));
  const timer=setTimeout(()=>done(err('REFERENCE_PDF_TIMEOUT','PDF parser timed out; no partial output was accepted')),timeoutMs);
  try{proc=spawn(program,args,{stdio:['pipe','pipe','pipe'],shell:false,windowsHide:true,env:parserEnv()});}
  catch{done(err('REFERENCE_PDF_UNAVAILABLE',program+' is unavailable'));return;}
  signal?.addEventListener('abort',abort,{once:true});
  proc.on('error',()=>done(err('REFERENCE_PDF_UNAVAILABLE',program+' is unavailable on the extension host')));
  proc.stdout.on('data',b=>{size+=b.length;if(size>maxBytes)done(err('REFERENCE_LIMIT','PDF parser output exceeds per-page limit; nothing was truncated'));else buffer.push(b);});
  proc.stderr.on('data',b=>{stderrSize+=b.length;if(stderrSize>32768)done(err('REFERENCE_PDF_ERROR','PDF parser diagnostic overflow'));});
  proc.stdin.on('error',()=>{});
  proc.on('close',code=>{if(settled)return;if(code!==0)done(err('REFERENCE_PDF_ERROR',program+' returned nonzero status ('+code+')'));else done(null,Buffer.concat(buffer));});
  proc.stdin.end(input);
 });
}
async function pageCount(bytes,{signal}={}){
 if(!Buffer.isBuffer(bytes)||bytes.subarray(0,5).toString()!=='%PDF-')throw err('REFERENCE_FORMAT','Invalid PDF signature');
 const out=await run('pdfinfo',['-'],bytes,{signal,timeoutMs:30000,maxBytes:16384});
 const m=out.toString('utf8').match(/^Pages:\s+(\d+)\s*$/m);
 if(!m||!Number.isSafeInteger(+m[1])||+m[1]<1)throw err('REFERENCE_PDF_ERROR','PDF page count could not be verified');
 return +m[1];
}
async function renderPage(bytes,page,{count,signal,pixels=1700}={}){
 if(!Number.isSafeInteger(page)||page<1||(count&&page>count))throw err('REFERENCE_RANGE','Invalid PDF page index');
 if(!Number.isSafeInteger(pixels)||pixels<400||pixels>2000)throw err('REFERENCE_RANGE','Invalid PDF rendering size');
 const buf=await run('pdftoppm',['-f',String(page),'-l',String(page),'-singlefile','-scale-to',String(pixels),'-png','-'],bytes,{signal,timeoutMs:45000,maxBytes:LIMITS.imageBytes});
 if(imageMime(buf)!=='image/png')throw err('REFERENCE_PDF_ERROR','PDF rendering did not produce a valid PNG');
 return buf;
}
async function ocrDocument(bytes,{count,signal,maxPages=LIMITS.blocks,selectedPages}={}){
 const pages=count??await pageCount(bytes,{signal});
 const wanted=selectedPages||Array.from({length:pages},(_,i)=>i+1);
 if(wanted.length>maxPages)throw err('REFERENCE_OCR_LIMIT','Scanned PDF exceeds automatic OCR page limit; original document was not truncated');
 const available=(await run('tesseract',['--list-langs'],Buffer.alloc(0),{signal,timeoutMs:8000,maxBytes:8192})).toString('utf8');
 const hasLang=lang=>new RegExp('(?:^|\\n)'+lang+'(?:\\r?\\n|$)').test(available);
 const language=(process.env.VODER_OCR_LANG||(hasLang('chi_sim')&&hasLang('eng')?'eng+chi_sim':hasLang('eng')?'eng':'')).trim();
 if(!/^[a-zA-Z0-9_+]{2,48}$/.test(language)||language.split('+').some(x=>!hasLang(x)))throw err('REFERENCE_OCR_LANGUAGE','Requested OCR language data is missing; install Tesseract traineddata');
 const blocks=[];let bytesTotal=0;
 for(const p of wanted){
  if(!Number.isSafeInteger(p)||p<1||p>pages)throw err('REFERENCE_RANGE','Invalid OCR page selection');
  if(signal?.aborted)throw err('REFERENCE_CANCELLED','PDF OCR cancelled');
  const png=await renderPage(bytes,p,{count:pages,signal,pixels:1700});
  // Language list is operator configurable; do not fetch/install dictionaries silently.
  const text=(await run('tesseract',['stdin','stdout','-l',language,'--psm','3'],png,{signal,timeoutMs:60000,maxBytes:LIMITS.textBytes})).toString('utf8');
  const block='[PDF page '+p+'/'+pages+' | OCR, verify transcription against image]\n'+text.trim()+'\n';
  bytesTotal+=Buffer.byteLength(block);if(bytesTotal>LIMITS.textBytes)throw err('REFERENCE_LIMIT','OCR text exceeds request capacity; no partial document was sent');
  blocks.push(block);
 }
 if(blocks.every(x=>!x.replace(/\[PDF page[^\]]*\]/g,'').trim()))throw err('REFERENCE_PDF_NO_TEXT','OCR found no readable text; use a vision-capable model to inspect page images');
 return {text:blocks.join('\n'),language,pages,blocks};
}
function visualRequested(goal){return /(?:图表|图形|插图|截图|示意图|流程图|架构图|图示|图像|图片|图片里|视觉|排版|版式|扫描件|看图|图\s*\d+|figure\s*\d+|fig\.\s*\d+|chart|diagram|visual|layout|image\s+in|scan(?:ned)?\s+page)/i.test(goal||'');}
function indexedFigures(text,count){
 const entries=new Map();
 if(typeof text!=='string'||text.length>12*1024*1024)return entries;
 const marker=/\[PDF PHYSICAL PAGE (\d+)\/(\d+)\]\n/g;
 const found=[...text.matchAll(marker)];
 for(let i=0;i<found.length;i++){
  const page=+found[i][1];
  if(+found[i][2]!==count||page<1||page>count)continue;
  const block=text.slice(found[i].index+found[i][0].length,found[i+1]?.index??text.length);
  // Require a genuine figure caption line, not an incidental body reference.
  for(const line of block.split(/\r?\n/)){
   const m=line.match(/^\s*(?:figure|fig\.)\s*(\d{1,3})\s*[.:：]\s*\S/i);
   if(m){const n=+m[1],arr=entries.get(n)||[];if(!arr.includes(page))arr.push(page);entries.set(n,arr);}
  }
 }
 return entries;
}
function selectPages(goal,count,documentText){
 const result=new Set();let m;
 const inclusive=(first,last)=>{if(last<first||last-first>=8)throw err('REFERENCE_VISUAL_LIMIT','A visual page range may contain at most 8 pages');for(let p=first;p<=last;p++)if(p>=1&&p<=count)result.add(p);};
 const ranges=[/(?:pages?\s*|第\s*)(\d+)\s*(?:-|–|—|~|至|到)\s*(\d+)\s*(?:页)?/gi];
 for(const re of ranges)while((m=re.exec(goal)))inclusive(+m[1],+m[2]);
 const patterns=[/第\s*(\d+)\s*(?:页|张)/g,/(?:page|pages|p\.)\s*(\d+)\b/gi,/#P(\d+)\b/gi];
 for(const re of patterns)while((m=re.exec(goal))){const p=+m[1];if(p>=1&&p<=count)result.add(p);}
 const tail=goal.match(/最后\s*(\d+)\s*页/);if(tail)inclusive(Math.max(1,count-(+tail[1])+1),count);
 const head=goal.match(/(?:前|最初)\s*(\d+)\s*页/);if(head)inclusive(1,Math.min(count,+head[1]));
 if(/最后一页|末页|last\s+page/i.test(goal))result.add(count);
 if(/首页|第一页|first\s+page/i.test(goal))result.add(1);
 // For a figure-number request with no physical page specified, resolve
 // figure captions from the PDF's verified physical-page text mapping.
 if(!result.size&&typeof documentText==='string'){
  const nums=new Set();let match;
  const figureRe=/(?:图\s*|figure\s*|fig\.\s*)(\d{1,3})(?!\d)/gi;
  while((match=figureRe.exec(goal)))nums.add(+match[1]);
  if(nums.size){
   const figures=indexedFigures(documentText,count);
   for(const n of nums){
    const matches=figures.get(n)||[];
    if(matches.length!==1)throw err('REFERENCE_VISUAL_SCOPE','Figure '+n+' cannot be uniquely located from verified PDF captions; give a physical page number');
    result.add(matches[0]);
   }
  }
 }
 // Never silently claim visual coverage over pages that are not actually sent.
 if(!result.size){
  if(count>8)throw err('REFERENCE_VISUAL_SCOPE','Visual analysis of every page needs a retrieval workflow; name the relevant pages to avoid silently skipping figures');
  for(let p=1;p<=count;p++)result.add(p);
 }
 if(result.size>8)throw err('REFERENCE_VISUAL_LIMIT','A single visual request supports up to 8 explicitly named PDF pages');
 return [...result].sort((a,b)=>a-b);
}
async function visualReferences(bytes,{name,source,goal,signal,documentText,range,originalDigest}={}){
 const count=await pageCount(bytes,{signal});
 const pages=range?Array.from({length:range.end-range.start+1},(_,i)=>range.start+i):Array.from({length:count},(_,i)=>i+1);
 if(pages.length>LIMITS.blocks)throw err('REFERENCE_VISUAL_LIMIT','Document page metadata exceeds host capacity; no pages were silently omitted');
 const refs=[];let total=0;
 for(const p of pages){
  if(p<1||p>count)throw err('REFERENCE_RANGE','Requested page does not exist');
  const png=await renderPage(bytes,p,{count,signal});total+=png.length;
  if(total>LIMITS.imageBytes)throw err('REFERENCE_LIMIT','Complete document images exceed the image budget; no partial document was sent');
  refs.push({id:randomUUID(),kind:'image',source,name:name+' [PDF page '+p+'/'+count+']',mimeType:'image/png',data:png.toString('base64'),sha256:digest(png),note:'Rendered PDF page '+p+' of '+count+'; complete selected visual coverage; original source SHA256 '+(originalDigest||digest(bytes))});
 }
 return refs;
}
module.exports={pageCount,renderPage,ocrDocument,visualRequested,indexedFigures,selectPages,visualReferences};
