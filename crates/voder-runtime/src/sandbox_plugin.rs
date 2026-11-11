//! Audited cloud sandbox-plugin adapter for Rust Voder Runtime.
//! Non-authoritative: Core admission -> durable journal permit -> ToolExecutor -> here.
//! The plugin can execute only after upstream authorization. It never commits a VGR verdict.
use crate::{
    artifact_store::sha256,
    process::{self, ProcessResult},
    sandbox::{private, SandboxAvailability, SandboxBackend, SandboxPolicy},
    Result, RuntimeError,
};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};
use voder_protocol::json::{self, Value};

static INVOCATION: AtomicU64 = AtomicU64::new(0);

pub struct PluginSandbox {
    node: PathBuf,
    cli: PathBuf,
    provider: String,
}
impl PluginSandbox {
    pub fn new(node: PathBuf, cli: PathBuf, provider: &str) -> Result<Self> {
        if provider.is_empty()
            || provider.len() > 40
            || !provider.as_bytes()[0].is_ascii_lowercase()
            || !provider
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
        {
            return Err(RuntimeError::Invalid("invalid sandbox provider id".into()));
        }
        for path in [&node, &cli] {
            if !path.is_absolute()
                || !path.is_file()
                || fs::symlink_metadata(path)?.file_type().is_symlink()
            {
                return Err(RuntimeError::UnsafePath(
                    "trusted Node or plugin CLI must be an absolute, regular file".into(),
                ));
            }
        }
        Ok(Self {
            node,
            cli,
            provider: provider.to_string(),
        })
    }
}
impl private::Sealed for PluginSandbox {}
impl SandboxBackend for PluginSandbox {
    fn probe(&self) -> SandboxAvailability {
        if !self.node.is_file() || !self.cli.is_file() {
            return SandboxAvailability::Unavailable("trusted Node/plugin CLI missing".into());
        }
        if self.provider == "modal" {
            let token = std::env::var("MODAL_TOKEN_ID").unwrap_or_default();
            let secret = std::env::var("MODAL_TOKEN_SECRET").unwrap_or_default();
            let image = std::env::var("VODER_MODAL_IMAGE").unwrap_or_default();
            let app = std::env::var("VODER_MODAL_APP").unwrap_or_default();
            if token.is_empty()
                || secret.is_empty()
                || !image.contains("@sha256:")
                || app.is_empty()
            {
                return SandboxAvailability::Unavailable(
                    "Modal credentials, app, or pinned image missing".into(),
                );
            }
        }
        SandboxAvailability::Available
    }
    fn execute(
        &self,
        policy: &SandboxPolicy,
        program: &str,
        args: &[String],
        capture_dir: &Path,
    ) -> Result<ProcessResult> {
        policy.validate()?;
        if !program.starts_with('/') || program.chars().any(char::is_control) || args.len() > 128 {
            return Err(RuntimeError::Invalid("invalid sandbox command".into()));
        }
        if args
            .iter()
            .any(|arg| arg.contains('\0') || arg.len() > 8192)
        {
            return Err(RuntimeError::Invalid("invalid sandbox arguments".into()));
        }
        if !matches!(self.probe(), SandboxAvailability::Available) {
            return Err(RuntimeError::CapabilityUnavailable(
                "sandbox provider unavailable",
            ));
        }
        let limits = policy.limits()?;
        fs::create_dir_all(capture_dir)?;
        if fs::symlink_metadata(capture_dir)?.file_type().is_symlink() {
            return Err(RuntimeError::UnsafePath("capture directory symlink".into()));
        }
        let root = capture_dir.canonicalize()?;
        let invocation = INVOCATION.fetch_add(1, Ordering::Relaxed);
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| RuntimeError::Invalid("system clock error".into()))?
            .as_nanos();
        let run_id = format!("sandbox-{}-{}", std::process::id(), stamp);
        let action_id = format!("execute-{invocation}");
        let input = Value::object([
            (
                "workspace".into(),
                Value::String(
                    policy
                        .worktree
                        .canonicalize()?
                        .to_string_lossy()
                        .into_owned(),
                ),
            ),
            ("program".into(), Value::String(program.to_string())),
            (
                "args".into(),
                Value::Array(args.iter().cloned().map(Value::String).collect()),
            ),
            (
                "timeoutMs".into(),
                Value::Number(limits.timeout.as_millis().to_string()),
            ),
            (
                "memoryMiB".into(),
                Value::Number((limits.memory_bytes / (1024 * 1024)).to_string()),
            ),
            ("cpu".into(), Value::Number("1".into())),
            (
                "maxOutputBytes".into(),
                Value::Number(limits.max_output_bytes.to_string()),
            ),
            ("runId".into(), Value::String(run_id.clone())),
            ("actionId".into(), Value::String(action_id.clone())),
        ]);
        let request = root.join("sandbox-request.json");
        let receipt = root.join("sandbox-receipt.json");
        let remote_stdout = root.join("sandbox-program.stdout.raw");
        let remote_stderr = root.join("sandbox-program.stderr.raw");
        let mut opts = fs::OpenOptions::new();
        opts.create_new(true).write(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        opts.open(&request)?.write_all(input.encode().as_bytes())?;
        let mut cmd = Command::new(&self.node);
        cmd.arg("--experimental-strip-types")
            .arg(&self.cli)
            .arg(&request)
            .arg(&receipt)
            .arg(&remote_stdout)
            .arg(&remote_stderr)
            .env("VODER_SANDBOX_BACKEND", &self.provider)
            .env_remove("NODE_OPTIONS");
        // Wait for the Modal driver's authenticated receipt. The sandbox program
        // deadline remains in the signed request; it must not kill the client
        // while Modal uploads inputs, schedules the sandbox, or seals evidence.
        let transport = process::run_until_completion(cmd, &root, limits)?;
        if transport.exit != Some(0) || transport.timed_out || transport.output_limited {
            return Err(RuntimeError::Execution(
                "sandbox-plugin transport failed: see bounded captured stderr".into(),
            ));
        }
        let bytes = fs::read(&receipt)
            .map_err(|_| RuntimeError::Integrity("missing plugin receipt".into()))?;
        if bytes.len() > 16 * 1024 * 1024 {
            return Err(RuntimeError::Integrity("oversized plugin receipt".into()));
        }
        let record = json::parse(&bytes)
            .map_err(|_| RuntimeError::Integrity("invalid plugin receipt".into()))?;
        // The Node process stderr includes SDK/runtime diagnostics and can
        // contain experimental TypeScript warnings. Only these dedicated,
        // exclusively-created files hold remote command output evidence.
        fn read_remote_stream(path: &Path, max_bytes: u64) -> Result<Vec<u8>> {
            let meta = fs::symlink_metadata(path)
                .map_err(|_| RuntimeError::Integrity("remote output artifact missing".into()))?;
            if !meta.is_file() || meta.file_type().is_symlink() || meta.len() > max_bytes {
                return Err(RuntimeError::Integrity(
                    "remote output artifact unsafe or oversized".into(),
                ));
            }
            let bytes = fs::read(path)?;
            if bytes.len() as u64 > max_bytes || bytes.len() as u64 != meta.len() {
                return Err(RuntimeError::Integrity(
                    "remote output artifact changed during read".into(),
                ));
            }
            Ok(bytes)
        }
        let stdout = read_remote_stream(&remote_stdout, limits.max_output_bytes)?;
        let stderr = read_remote_stream(&remote_stderr, limits.max_output_bytes)?;
        if (stdout.len() as u64).saturating_add(stderr.len() as u64) > limits.max_output_bytes {
            return Err(RuntimeError::Integrity(
                "remote output budget exceeded".into(),
            ));
        }
        let field = |key: &str| record.get(key).and_then(Value::as_str).unwrap_or("");
        if field("protocol") != "voder-sandbox-receipt-v1"
            || field("provider") != self.provider
            || field("status") != "EXECUTED"
            || field("runId") != run_id
            || field("actionId") != action_id
            || (self.provider == "modal"
                && field("image") != std::env::var("VODER_MODAL_IMAGE").unwrap_or_default())
            || field("stdoutSha256") != sha256(&stdout)?
            || field("stderrSha256") != sha256(&stderr)?
        {
            return Err(RuntimeError::Integrity(
                "receipt identity or byte digest mismatch".into(),
            ));
        }
        let exit_code = record
            .get("exitCode")
            .and_then(Value::as_u64)
            .ok_or_else(|| RuntimeError::Integrity("missing remote exit code".into()))?;
        if exit_code > 255 {
            return Err(RuntimeError::Integrity(
                "remote exit code out of range".into(),
            ));
        }
        let bool_field = |name: &str| match record.get(name) {
            Some(Value::Bool(value)) => Ok(*value),
            _ => Err(RuntimeError::Integrity(format!(
                "invalid receipt boolean: {name}"
            ))),
        };
        let limited = bool_field("outputLimited")?;
        let timed_out = bool_field("timeout")?;
        Ok(ProcessResult {
            exit: Some(exit_code as i32),
            timed_out,
            output_limited: limited,
            stdout,
            stderr,
            elapsed: transport.elapsed,
        })
    }
}
