'use strict';
/** Presentation of an already received answer. Never requests or alters model output. */
function createProgressiveOutput(document,{render,requestFrame,cancelFrame,now=()=>Date.now(),reducedMotion=()=>false,onUpdate=()=>{}}={}){
 const active=new Set();
 function show(raw,container){
  render(raw,container);
  if(typeof requestFrame!=='function'||reducedMotion()||document.hidden)return {finish(){},cancel(){}};
  const entries=[],gates=[];let total=0;
  const segmenter=typeof Intl!=='undefined'&&Intl.Segmenter?new Intl.Segmenter(undefined,{granularity:'grapheme'}):null;
  const units=text=>segmenter?Array.from(segmenter.segment(text),x=>x.segment):Array.from(text);
  function visit(node,parents){
   if(node.nodeType===3){const text=node.textContent||'';if(!text)return;const parts=units(text);node.textContent='';entries.push({node,parts,offset:0,parents});total+=parts.length;return;}
   const tag=String(node.tagName||'').toLowerCase(),classes=String(node.className||'');
   if(tag==='button'||/(?:^|\s)(?:response-math(?:-block)?|katex)(?:\s|$)/.test(classes)){
    node.hidden=true;gates.push(node);entries.push({atomic:node,parents,weight:Math.max(1,Math.min(24,units(node.textContent||'').length))});total+=entries.at(-1).weight;return;
   }
   let next=parents;if(tag==='tr'||tag==='li'){node.hidden=true;gates.push(node);next=[...parents,node];}
   const children=Array.from(node.childNodes||[]);
   if(!children.length){node.hidden=true;gates.push(node);entries.push({atomic:node,parents,weight:1});total++;}
   for(const child of children)visit(child,next);
  }
  for(const block of Array.from(container.childNodes||[])){
   if(block.nodeType===3){visit(block,[]);continue;}
   block.hidden=true;gates.push(block);visit(block,[block]);
  }
  if(!entries.length){for(const gate of gates)gate.hidden=false;return {finish(){},cancel(){}};}
  container.setAttribute?.('aria-busy','true');container.setAttribute?.('aria-live','off');
  let frame=null,index=0,done=false,revealed=0;const started=now(),duration=Math.min(3200,Math.max(450,450+total/8));
  function open(entry){for(const p of entry.parents)p.hidden=false;if(entry.atomic)entry.atomic.hidden=false;}
  function take(entry,n){open(entry);if(entry.atomic)return entry.weight;
   const amount=Math.min(n,entry.parts.length-entry.offset),text=entry.parts.slice(entry.offset,entry.offset+amount).join('');
   if(typeof entry.node.appendData==='function')entry.node.appendData(text);else entry.node.textContent+=text;
   entry.offset+=amount;return amount;
  }
  function settled(){done=true;active.delete(control);if(frame!==null&&typeof cancelFrame==='function')cancelFrame(frame);frame=null;
   for(const gate of gates)gate.hidden=false;container.setAttribute?.('aria-busy','false');container.removeAttribute?.('aria-live');onUpdate(container);
  }
  function finish(){if(done)return;for(;index<entries.length;index++){const e=entries[index];take(e,e.atomic?e.weight:e.parts.length-e.offset);}settled();}
  function cancel(){if(done)return;done=true;active.delete(control);if(frame!==null&&typeof cancelFrame==='function')cancelFrame(frame);frame=null;}
  function tick(){frame=null;if(done)return;if(document.hidden||reducedMotion()){finish();return;}
   const elapsed=Math.max(0,now()-started);if(elapsed>=duration+250){finish();return;}
   const target=Math.min(total,Math.max(1,Math.floor(total*Math.min(1,elapsed/duration))));
   let allowance=Math.min(1000,Math.max(0,target-revealed)),steps=0;
   while(index<entries.length&&allowance>0&&steps++<250){const e=entries[index],used=take(e,allowance);revealed+=used;allowance-=used;if(e.atomic||e.offset===e.parts.length)index++;}
   onUpdate(container);if(index===entries.length){settled();return;}frame=requestFrame(tick);
  }
  const control={finish,cancel};active.add(control);while(active.size>12)active.values().next().value.finish();frame=requestFrame(tick);return control;
 }
 document.addEventListener?.('visibilitychange',()=>{if(document.hidden)for(const c of [...active])c.finish();});
 return {render:show,finishAll(){for(const c of [...active])c.finish();},cancelAll(){for(const c of [...active])c.cancel();},get pending(){return active.size;}};
}
module.exports={createProgressiveOutput};
