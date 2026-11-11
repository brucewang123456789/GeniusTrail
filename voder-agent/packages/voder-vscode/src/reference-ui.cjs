'use strict';
/** VS Code file access is reachable only from explicit attachment gestures/commands. */
const path=require('node:path');
const {needsVisual}=require('./document-input.cjs');
const {loadReferences}=require('./archive-input.cjs');
const {validateReferences}=require('./reference-contract.cjs');
const {ReferenceStore,safeRead,fromBytes,decodeBase64,LIMITS}=require('./reference-files.cjs');
const fail=(code,message)=>{throw Object.assign(Error(message),{code});};
function localUri(uri){if(!uri||uri.scheme&&uri.scheme!=='file'||typeof uri.fsPath!=='string'||!uri.fsPath)fail('REFERENCE_URI','Select a regular file on the extension host');return uri.fsPath;}
class ReferenceController{
 constructor({vscode}={}){this.vscode=vscode;this.states=new Map();}
 api(){return this.vscode||require('vscode');}
 state(view){let state=this.states.get(view);if(!state){state={store:new ReferenceStore(),loading:false,abort:new AbortController()};this.states.set(view,state);}return state;}
 dispose(view){const s=this.states.get(view);if(s){s.abort.abort();s.store.dispose();this.states.delete(view);}}
 async files(view,uris){
  const s=this.state(view),api=this.api(),selected=uris||await api.window.showOpenDialog({title:'Voder · Add reference files (read-only)',canSelectFiles:true,canSelectFolders:false,canSelectMany:true,openLabel:'Add to context'});
  if(!selected?.length)return;
  if(selected.length>LIMITS.items)fail('REFERENCE_LIMIT','Selected batch exceeds host metadata capacity');
  const items=[],pdfPairs=[];let rawBytes=0;
  for(const uri of selected){
   const file=localUri(uri),bytes=await safeRead(file,{signal:s.abort.signal});rawBytes+=bytes.length;if(rawBytes>LIMITS.fileBytes)fail('REFERENCE_LIMIT','Selected source bytes exceed available host memory; no files were partially added');
   // User selects the file once. Full PDF text is extracted by default; explicit @path#P ranges remain supported.
   const range=undefined;
   const document=api.workspace?.textDocuments?.find(d=>d.isDirty&&d.uri.scheme==='file'&&path.resolve(d.uri.fsPath)===path.resolve(file));
   const loaded=await loadReferences({name:path.basename(file),bytes:document?Buffer.from(document.getText()):bytes,range,signal:s.abort.signal});
   items.push(...loaded.items.map(item=>document?{...item,note:'Editor text snapshot, including unsaved changes.'}:item));
   pdfPairs.push(...loaded.visualSources.map(({item,bytes})=>[item.id,bytes]));validateReferences([...s.store.snapshot(),...items]);
  }
  s.store.add(items);for(const [id,bytes] of pdfPairs)s.store.pdfSources.set(id,bytes);
 }
 async editor(view,{selection=false}={}){
  const s=this.state(view),api=this.api(),editor=api.window.activeTextEditor;
  if(!editor)fail('REFERENCE_SELECTION','Open a text file in the editor first');
  const file=localUri(editor.document.uri);
  if(selection&&editor.selection.isEmpty)fail('REFERENCE_SELECTION','Select code or text first');
  const content=editor.document.getText(selection?editor.selection:undefined);
  const item=await fromBytes({name:api.workspace.asRelativePath?.(editor.document.uri,false)||path.basename(file),bytes:Buffer.from(content),source:selection?'selection':'workspace'});
  const start=selection?editor.selection.start.line+1:1;
  const end=selection?Math.max(start,editor.selection.end.line+(editor.selection.end.character===0?0:1)):Math.max(1,editor.document.lineCount||1);
  s.store.add([{...item,range:{unit:'line',start,end},note:selection?'Exact editor selection; may begin or end mid-line. Snapshot includes unsaved editor changes.':'Editor text snapshot, including unsaved changes.'}]);
 }
 async dropped(view,msg){
  if(!Array.isArray(msg.files)||!msg.files.length||msg.files.length>LIMITS.items)fail('REFERENCE_LIMIT','Invalid attachment batch');
  const s=this.state(view),items=[],pdfPairs=[];let total=0;
  for(const f of msg.files){
   if(!f||Object.keys(f).some(k=>!['name','data'].includes(k))||typeof f.name!=='string'||f.name.includes('/')||f.name.includes('\\'))fail('REFERENCE_SCHEMA','Drag files as bytes, not paths');
   const bytes=decodeBase64(f.data);total+=bytes.length;if(total>LIMITS.fileBytes)fail('REFERENCE_LIMIT','Dropped bytes exceed available host transfer memory; use Add files for host-local files');
   const loaded=await loadReferences({name:f.name,bytes,signal:s.abort.signal});items.push(...loaded.items);
   pdfPairs.push(...loaded.visualSources.map(({item,bytes})=>[item.id,bytes]));validateReferences([...s.store.snapshot(),...items]);
  }
  s.store.add(items);for(const [id,bytes] of pdfPairs)s.store.pdfSources.set(id,bytes);
 }
 async prepare(view,root,goal){
  const api=this.api(),s=this.state(view);
  return s.store.prepare(root,goal,{signal:s.abort.signal,read:async(file,options)=>{
   const disk=await safeRead(file,options);
   const doc=api.workspace.textDocuments?.find(d=>d.isDirty&&d.uri.scheme==='file'&&path.resolve(d.uri.fsPath)===path.resolve(file));
   return doc?Buffer.from(doc.getText(),'utf8'):disk;
  }});
 }
 async requestContext(folder,goal,references=[]){
  if(!Array.isArray(references)||references.length>LIMITS.items)fail('REFERENCE_LIMIT','Too many chat file references');
  const key={},s=this.state(key),api=this.api();let rawBytes=0;
  try{
   for(const reference of references){
    const value=reference?.value,uri=value?.uri||value;
    if(uri?.scheme!=='file')fail('REFERENCE_URI','This chat reference is not a supported file; add it through Voder Chat');
    const file=localUri(uri),bytes=await safeRead(file,{signal:s.abort.signal});rawBytes+=bytes.length;if(rawBytes>LIMITS.fileBytes)fail('REFERENCE_LIMIT','Selected source bytes exceed available host memory; no files were partially added');
    let item;
    if(value?.range){
     const document=await api.workspace.openTextDocument(uri),content=document.getText(value.range);
     item=await fromBytes({name:path.basename(file),source:'selection',bytes:Buffer.from(content)});
     item={...item,range:{unit:'line',start:value.range.start.line+1,end:Math.max(value.range.start.line+1,value.range.end.line+(value.range.end.character===0?0:1))},note:'Exact VS Code chat selection, including unsaved text; may start or end mid-line.'};
    }else{
     const document=api.workspace.textDocuments?.find(d=>d.isDirty&&d.uri.scheme==='file'&&path.resolve(d.uri.fsPath)===path.resolve(file));
     const loaded=await loadReferences({name:path.basename(file),bytes:document?Buffer.from(document.getText()):bytes,signal:s.abort.signal});
     s.store.add(loaded.items);for(const {item,bytes} of loaded.visualSources)s.store.pdfSources.set(item.id,bytes);
     continue;
    }
    s.store.add([item]);
    if(needsVisual(file,bytes))s.store.pdfSources.set(item.id,bytes);
   }
   return await this.prepare(key,folder.uri.fsPath,goal);
  }finally{this.dispose(key);}
 }
}
module.exports={ReferenceController,localUri};
