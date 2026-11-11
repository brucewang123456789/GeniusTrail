import {readFileSync} from 'node:fs';
export const VERIFICATION_RUNTIME_POLICY=readFileSync(new URL('../policies/verification.md',import.meta.url),'utf8').trim();
