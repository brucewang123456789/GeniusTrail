/** Only this file knows the Modal JS SDK. Other providers implement the same generic contract. */
export interface RemoteProcess {
  stdout:ReadableStream<string|Uint8Array>;
  stderr:ReadableStream<string|Uint8Array>;
  wait():Promise<number>;
}
export interface RemoteSandbox {
  sandboxId:string;
  filesystem:{writeBytes(bytes:Uint8Array,path:string):Promise<void>;writeText(text:string,path:string):Promise<void>};
  exec(command:string[],options?:{workdir?:string;timeoutMs?:number}):Promise<RemoteProcess>;
  terminate():Promise<void>;
}
export interface ModalDriver {
  create(appName:string,image:string,options:{timeoutMs:number;idleTimeoutMs:number;memoryMiB:number;cpu:number;blockNetwork:boolean}):Promise<RemoteSandbox>;
}
export async function createModalDriver():Promise<ModalDriver>{
  const moduleName='modal'; // Late binding; SDK is a separately installed provider dependency.
  const {ModalClient}=await import(moduleName);
  const modal=new ModalClient();
  return {async create(appName,image,opts){
    const app=await modal.apps.fromName(appName,{createIfMissing:true});
    const img=modal.images.fromRegistry(image);
    return modal.sandboxes.create(app,img,{
      timeoutMs:opts.timeoutMs,idleTimeoutMs:opts.idleTimeoutMs,
      memoryMiB:opts.memoryMiB,memoryLimitMiB:opts.memoryMiB,
      cpu:opts.cpu,cpuLimit:opts.cpu,blockNetwork:true,workdir:'/tmp',
    });
  }};
}
