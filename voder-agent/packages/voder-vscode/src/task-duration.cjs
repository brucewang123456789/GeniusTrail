'use strict';
/** Pure browser-side clock, frozen at first final-content presentation. */
function formatTaskDuration(ms){
 const seconds=Math.max(0,Math.floor((Number.isFinite(ms)?ms:0)/1000));
 const hours=Math.floor(seconds/3600),minutes=Math.floor(seconds%3600/60),rest=seconds%60;
 const parts=[];if(hours)parts.push(hours+'h');if(minutes)parts.push(minutes+'m');if(rest||!parts.length)parts.push(rest+'s');return parts.join(' ');
}
function createTaskClock(now){
 let start=null,frozen=null;
 return {
  start(){if(start===null)start=now();},
  reset(){start=null;frozen=null;},
  elapsed(){return frozen??(start===null?0:Math.max(0,now()-start));},
  freeze(){if(frozen===null)frozen=start===null?0:Math.max(0,now()-start);return frozen;},
  get started(){return start!==null;},get frozen(){return frozen!==null;}
 };
}
module.exports={formatTaskDuration,createTaskClock};
