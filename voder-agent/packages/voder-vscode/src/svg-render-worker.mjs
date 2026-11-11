/** Trusted renderer worker. Receives validated SVG bytes, never executable user code. */
import {createRequire} from 'node:module';
const require=createRequire(import.meta.url);
let input='';for await(const chunk of process.stdin){input+=chunk;}
const {content,width,height}=JSON.parse(input),sharp=require(process.argv[2]);sharp.cache(false);sharp.concurrency(1);
const images=[];for(const scale of [1,.5]){const png=await sharp(Buffer.from(content),{limitInputPixels:2048*1536}).resize(Math.max(1,Math.round(width*scale)),Math.max(1,Math.round(height*scale))).png().toBuffer();if(png.length>1048576)throw Error('SVG_IMAGE_LIMIT');images.push(png.toString('base64'));}
process.stdout.write(JSON.stringify({renderer:'sharp/'+sharp.versions.sharp+'; librsvg/'+sharp.versions.rsvg,images}));
