import {open,readFile,stat} from 'node:fs/promises';
import {sha,insist} from './safety.mjs';
export class Journal {
 constructor(path){this.path=path;this.seq=0;this.head='GENESIS';this.closed=false;}
 async write(event,data={}){insist(!this.closed,'JOURNAL_CLOSED');const entry={seq:++this.seq,prev:this.head,event,data};entry.hash=sha(JSON.stringify(entry));const f=await open(this.path,'a',0o600);try{await f.writeFile(JSON.stringify(entry)+'\n');await f.sync();}finally{await f.close();}this.head=entry.hash;return entry;}
 async verify(){let raw;try{raw=await readFile(this.path,'utf8')}catch(e){if(e.code==='ENOENT')return [];throw e;}
  const lines=raw.split('\n');insist(lines.at(-1)==='', 'JOURNAL_TORN_WRITE');lines.pop();let seq=0,prev='GENESIS';const entries=[];
  for(const line of lines){const obj=JSON.parse(line);const {hash,...base}=obj;insist(obj.seq===++seq&&obj.prev===prev&&hash===sha(JSON.stringify(base)),'JOURNAL_TAMPERED');prev=hash;entries.push(obj);}
  this.seq=seq;this.head=prev;return entries;
 }
 close(){this.closed=true;}
}
