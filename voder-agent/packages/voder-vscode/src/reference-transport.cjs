'use strict';
const {validateReferences}=require('./reference-contract.cjs');
function referenceTask(rpc,task,references=[]){
 const items=validateReferences(references);
 if(!items.length)return task; // Preserve legacy task bytes and recovery behavior.
 if(rpc.referenceContextCapable!==true)throw Object.assign(Error('This Voder runtime does not advertise file-context support; update the matching runtime before sending references'),{code:'REFERENCE_RUNTIME_UNSUPPORTED'});
 return {...task,referenceContext:items};
}
module.exports={referenceTask};
