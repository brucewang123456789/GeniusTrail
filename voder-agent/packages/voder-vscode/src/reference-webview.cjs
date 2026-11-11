'use strict';
/** Static code is inserted under the existing nonce; runtime content uses textContent only. */
module.exports=String.raw`
 let referencesLoading=false,referenceRows=[];
 let hasEditor=false,hasSelection=false;
 const addTrigger=el('add-context'),addMenu=el('add-menu'),addHint=el('reference-hint');
 function menuOpen(open){
  if(!addMenu||!addTrigger)return;
  addMenu.hidden=!open;
  addTrigger.setAttribute('aria-expanded',String(!!open));
  if(open)api.postMessage({type:'reference_editor_state_request'});
  else if(addHint)addHint.textContent='';
 }
 function referenceControls(){
  const unavailable=busy||referencesLoading;
  if(addTrigger)addTrigger.disabled=unavailable;
  if(el('ref-files'))el('ref-files').disabled=unavailable;
  if(el('ref-selection'))el('ref-selection').disabled=unavailable||!hasSelection;
  if(el('ref-editor'))el('ref-editor').disabled=unavailable||!hasEditor;
  if(unavailable)menuOpen(false);
  if(el('send'))el('send').disabled=unavailable;
 }
 function referenceRequest(type,extra={}){if(busy||referencesLoading)return;el('warn').textContent='';api.postMessage({type,...extra});}
 addTrigger?.addEventListener('click',()=>{if(busy||referencesLoading)return;menuOpen(!!addMenu.hidden);});
 for(const [id,type] of [['ref-files','reference_files'],['ref-selection','reference_selection'],['ref-editor','reference_editor']])el(id)?.addEventListener('click',()=>{menuOpen(false);referenceRequest(type);});
 document.addEventListener?.('keydown',e=>{if(e.key==='Escape'&&!addMenu.hidden){menuOpen(false);addTrigger.focus?.();}});
 document.addEventListener?.('click',e=>{if(!addMenu.hidden&&!addTrigger.contains?.(e.target)&&!addMenu.contains?.(e.target))menuOpen(false);});
 function renderReferences(){
  const list=el('references');if(!list)return;list.replaceChildren();
  for(const item of referenceRows){
   const row=document.createElement('div');row.className='reference-chip';
   const preview=document.createElement('button');preview.type='button';preview.textContent=item.name+(item.range?' · '+item.range.unit+' '+item.range.start+'–'+item.range.end:'')+' · '+item.bytes+' B';preview.title='Preview the exact snapshot';preview.disabled=busy||referencesLoading;
   preview.addEventListener('click',()=>referenceRequest('reference_preview',{id:item.id}));
   const remove=document.createElement('button');remove.type='button';remove.textContent='×';remove.title='Remove '+item.name;remove.setAttribute('aria-label','Remove '+item.name);remove.disabled=busy||referencesLoading;
   remove.addEventListener('click',()=>referenceRequest('reference_remove',{id:item.id}));row.append(preview,remove);list.append(row);
  }
  const note=el('reference-note');if(note)note.textContent=referencesLoading?'Preparing document text and page images…':referenceRows.length?referenceRows.length+' reference(s) · for this message only; cleared once the task is accepted, retained if not started. Included content is sent to your selected model.':'Add files or use @path to bring your code into context.';
  referenceControls();
 }
 const dropZone=el('reference-dropzone');
 async function attachDropped(files){
  if(busy||referencesLoading)return;
  if(!files.length)return;
  referencesLoading=true;renderReferences();
  try{
   const batch=[];
   for(const f of files){
    const data=await new Promise((resolve,reject)=>{const reader=new FileReader();reader.onerror=()=>reject(Error('Unable to read dropped file'));reader.onload=()=>resolve(String(reader.result).split(',')[1]);reader.readAsDataURL(f);});
    batch.push({name:f.name,data});
   }
   api.postMessage({type:'reference_drop',files:batch});
  }catch(e){referencesLoading=false;renderReferences();el('warn').textContent=String(e.message||e);}
 }
 dropZone?.addEventListener('dragover',e=>{e.preventDefault();if(e.dataTransfer)e.dataTransfer.dropEffect=busy?'none':'copy';});
 dropZone?.addEventListener('drop',e=>{e.preventDefault();void attachDropped(Array.from(e.dataTransfer?.files||[]));});
 input.addEventListener('paste',e=>{const files=Array.from(e.clipboardData?.files||[]);if(files.length){e.preventDefault();void attachDropped(files);}});
 const originalSetBusy=setBusy;
 setBusy=function(value){originalSetBusy(value);referenceControls();renderReferences();};
 function handleReferenceMessage(m){
  if(m.type==='reference_editor_state'){
   hasEditor=m.hasEditor===true;hasSelection=hasEditor&&m.hasSelection===true;
   referenceControls();
   if(addHint&&!addMenu.hidden)addHint.textContent=!hasEditor?'Open a text file in the editor to use Selection or Open file.':!hasSelection?'Select text in the editor to enable Selection.':'';
  }
  if(m.type==='reference_hint'){
   if(addHint)addHint.textContent=String(m.message||'Open a text editor first');
  }
  if(m.type==='references_loading'){referencesLoading=!!m.loading;renderReferences();}
  if(m.type==='references'){el('reference-preview')?.replaceChildren();referenceRows=Array.isArray(m.items)?m.items:[];referencesLoading=!!m.loading;renderReferences();}
  if(m.type==='reference_error'){
   referencesLoading=false;if(m.blocking){setBusy(false);if(typeof m.goal==='string'&&!input.value)input.value=m.goal;if(runBubble)runBubble.content.textContent='Reference preparation failed · task not started';}
   renderReferences();el('warn').textContent=String(m.message||m.code||'Unable to load reference');
  }
  if(m.type==='references_included'&&runBubble){
   const box=document.createElement('details');box.className='reference-receipt';const label=document.createElement('summary');label.textContent='Included reference blocks ('+m.items.length+')';box.append(label);
   for(const item of m.items){const line=document.createElement('div');line.textContent=item.name+' · '+item.bytes+' B'+(item.range?' · '+item.range.unit+' '+item.range.start+'–'+item.range.end:'')+' · SHA256 '+item.sha256+(item.note?' · '+item.note:'');box.append(line);}runBubble.node.append(box);
  }
  if(m.type==='reference_preview'){
   const box=el('reference-preview');if(!box)return;box.replaceChildren();const close=document.createElement('button');close.textContent='Close preview';close.addEventListener('click',()=>box.replaceChildren());box.append(close);
   const title=document.createElement('div');title.textContent=m.item.name+' · SHA256 '+m.item.sha256;box.append(title);
   if(m.item.kind==='text'){const text=document.createElement('pre');text.textContent=m.item.content;box.append(text);}else if(['image/png','image/jpeg','image/webp'].includes(m.item.mimeType)){const img=document.createElement('img');img.alt=m.item.name;img.src='data:'+m.item.mimeType+';base64,'+m.item.data;box.append(img);}
   if(m.item.note){const note=document.createElement('div');note.textContent=m.item.note;box.append(note);}
  }
 }
`;
