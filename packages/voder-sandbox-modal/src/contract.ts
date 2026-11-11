/** Voder Modal execution contract. This boundary is non-authoritative: VGR owns admission. */
export type SandboxTask = {
  workspace: string;
  program: string;
  args: string[];
  requiredFiles?: string[];
  timeoutMs: number;
  setupBudgetMs?: number;
  memoryMiB: number;
  cpu: number;
  maxOutputBytes: number;
  runId: string;
  actionId: string;
};
export type Receipt = {
  protocol: 'voder-sandbox-receipt-v1';
  provider: string;
  runId: string; actionId: string;
  sandboxId: string; image: string;
  inputDigest: string; files: number; bytes: number;
  exitCode: number | null; timeout: boolean; outputLimited: boolean;
  stdoutSha256: string; stderrSha256: string;
  stdout: string; stderr: string;
  startedAt: string; endedAt: string;
  status: 'EXECUTED' | 'BLOCKED';
  reason?: string;
};
/** Keep remote output in the separately bounded process streams, not twice
 * inside the JSON control receipt. This also prevents JSON escaping from
 * expanding a valid 16 MiB output into an oversized receipt. */
export function receiptMetadata(receipt: Receipt): Omit<Receipt,'stdout'|'stderr'> {
  const {stdout: _stdout, stderr: _stderr, ...metadata} = receipt;
  return metadata;
}
const ID = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/;
export function assertRelative(name: string): string {
  if (typeof name !== 'string' || !name || name.length > 1024 || name.startsWith('/') ||
      name.includes('\\') || name.includes('\0') || /[\x00-\x1f\x7f]/.test(name) ||
      name.split('/').some(p => !p || p === '.' || p === '..')) {
    throw new Error('INVALID_RELATIVE_PATH');
  }
  return name;
}
export function validateTask(t: SandboxTask): SandboxTask {
  if (!t || !ID.test(t.runId) || !ID.test(t.actionId)) throw new Error('INVALID_IDENTITY');
  if (typeof t.workspace !== 'string' || !t.workspace.startsWith('/')) throw new Error('INVALID_WORKSPACE');
  if (!t.program || !t.program.startsWith('/') || /[\x00-\x1f]/.test(t.program)) throw new Error('ABSOLUTE_PROGRAM_REQUIRED');
  if (!Array.isArray(t.args) || t.args.length > 128 || t.args.some(a=>typeof a !== 'string' || a.length > 8192 || a.includes('\0'))) throw new Error('INVALID_ARGUMENTS');
  if (!Number.isSafeInteger(t.timeoutMs) || t.timeoutMs < 100 || t.timeoutMs > 3_600_000) throw new Error('INVALID_TIMEOUT');
  if (t.setupBudgetMs !== undefined && (!Number.isSafeInteger(t.setupBudgetMs) || t.setupBudgetMs < 300_000 || t.setupBudgetMs > 1_800_000)) throw new Error('INVALID_SANDBOX_SETUP_BUDGET');
  if (!Number.isSafeInteger(t.memoryMiB) || t.memoryMiB < 128 || t.memoryMiB > 32768) throw new Error('INVALID_MEMORY');
  if (!Number.isFinite(t.cpu) || t.cpu < 0.25 || t.cpu > 16) throw new Error('INVALID_CPU');
  if (!Number.isSafeInteger(t.maxOutputBytes) || t.maxOutputBytes < 1 || t.maxOutputBytes > 16*1024*1024) throw new Error('INVALID_OUTPUT_LIMIT');
  if (t.requiredFiles && (!Array.isArray(t.requiredFiles) || t.requiredFiles.length > 256)) throw new Error('INVALID_REQUIRED_FILES');
  for (const file of t.requiredFiles ?? []) assertRelative(file);
  return t;
}
export function validateConfig(env: NodeJS.ProcessEnv): {app: string; image: string} {
  const app = env.VODER_MODAL_APP;
  const image = env.VODER_MODAL_IMAGE;
  if (!app || !/^[a-z][a-z0-9-]{2,62}$/.test(app)) throw new Error('VODER_MODAL_APP_REQUIRED');
  if (!image || !/@sha256:[0-9a-f]{64}$/.test(image)) throw new Error('MODAL_IMAGE_DIGEST_REQUIRED');
  if (!env.MODAL_TOKEN_ID || !env.MODAL_TOKEN_SECRET) throw new Error('MODAL_CREDENTIALS_REQUIRED');
  return {app, image};
}
