import { assertAction, requireNativeReceipt, VoderError } from './contracts.mjs';
/** Native provider is a privileged local transport; never infer trust from the model payload. */
export class VgrBroker {
 constructor({native,sessionId,workspaceRoot,commitVerifier,delivery}){
   if(!native||typeof native.executeAuthorized!=='function') throw new VoderError('NO_NATIVE_VGR','No authenticated VGR native transport');
   this.native=native;this.sessionId=sessionId;this.workspaceRoot=workspaceRoot;this.commitVerifier=commitVerifier;this.delivery=delivery;this.deliveryResult=null;this.postCommitError=null;
   this.inflight=new Set();this.results=new Map();this.beforeNativeDispatch=null;this.afterNativeDispatch=null;
 }
 async dispatch(action, signal){
   const input=assertAction(action);
   if(input.session_id!==this.sessionId) throw new VoderError('SCOPE','Session mismatch');
   if(this.results.has(input.idempotency_key)) throw new VoderError('DUPLICATE','Side effect was already attempted');
   if(this.inflight.has(input.idempotency_key)) throw new VoderError('DUPLICATE','Concurrent duplicate operation');
   this.inflight.add(input.idempotency_key);
   // Mark attempted before crossing into native: a crash/timeout cannot auto-reissue.
   this.results.set(input.idempotency_key,{status:'OUTCOME_UNKNOWN'});
   try {
     if(signal?.aborted) throw new VoderError('CANCELLED','Operation aborted');
     // fsync the exact action identity before any privileged native dispatch.
     if(this.beforeNativeDispatch)await this.beforeNativeDispatch(input);
     const raw=await this.native.executeAuthorized(input,{signal,workspaceRoot:this.workspaceRoot});
     const receipt=requireNativeReceipt(raw,input);
     // A syntactically plausible native JSON object is NOT a final VGR grant.
     // The independent Rust daemon must reconstruct its durable event journal
     // and confirm both the final commit digest and the action identity.
     if(receipt.status==='COMMITTED'){
       if(!this.commitVerifier || typeof this.commitVerifier.confirmCommit!=='function')
         throw new VoderError('NO_NATIVE_COMMIT_VERIFIER','Authoritative Rust VGR commit verification unavailable');
       const verified=await this.commitVerifier.confirmCommit(receipt,input);
       if(verified?.verified!==true || verified.authority!=='VODER_NATIVE_VGR_STATE' ||
           verified.action_id!==input.action_id || verified.session_id!==input.session_id ||
           verified.run_id!==receipt.run_id || verified.commit_digest!==receipt.commit_digest)
         throw new VoderError('UNVERIFIED_COMMIT','Rust VGR state did not confirm the claimed final commit');
     }
     this.results.set(input.idempotency_key,receipt);
     if(this.afterNativeDispatch){
       try{await this.afterNativeDispatch(input,receipt);}catch(error){
         if(receipt.status!=='COMMITTED')throw error;
         // Rust confirmation above is authoritative. A failed host checkpoint cannot undo it.
         this.postCommitError='POST_COMMIT_CHECKPOINT_FAILED';
         this.deliveryResult=Object.freeze({status:'VERIFIED_NOT_DELIVERED',code:this.postCommitError});
         return receipt; // Do not deliver or re-dispatch when durable host reporting failed.
       }
     }
     if(receipt.status==='COMMITTED' && this.delivery){
       try{this.deliveryResult=await this.delivery.deliver({input,receipt,signal});}
       catch(e){this.deliveryResult=Object.freeze({status:'VERIFIED_NOT_DELIVERED',code:e?.code??'DELIVERY_FAILURE'});}
     }
     return receipt;
   } finally {this.inflight.delete(input.idempotency_key);}
 }
}
