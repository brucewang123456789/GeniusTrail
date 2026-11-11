//! Bounded one-shot model Provider transport, not a tool runner or VGR authority.
//! The trusted CLI is selected by the daemon operator, never by an untrusted request.
//! An intent is journaled BEFORE invoking this external (billable) side effect.
use crate::{DaemonError, Result};
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::{
    env,
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};
use voder_protocol::json::{self, Value};

const MAX_RECEIPT_BYTES: u64 = 256 * 1024;
const MAX_STDERR_BYTES: u64 = 64 * 1024;
const TIMEOUT: Duration = Duration::from_secs(200);

#[derive(Debug)]
pub struct ModelProposal {
    pub bytes: Vec<u8>,
}

#[derive(Debug)]
pub struct ProviderBridge {
    node: PathBuf,
    cli: PathBuf,
}
impl ProviderBridge {
    pub fn from_env() -> Result<Option<Self>> {
        let node = env::var_os("VODER_PROVIDER_NODE");
        let cli = env::var_os("VODER_PROVIDER_CLI");
        if node.is_none() && cli.is_none() {
            return Ok(None);
        }
        let (Some(node), Some(cli)) = (node, cli) else {
            return Err(DaemonError::Invalid(
                "incomplete model bridge configuration",
            ));
        };
        Self::new(PathBuf::from(node), PathBuf::from(cli)).map(Some)
    }
    pub fn new(node: PathBuf, cli: PathBuf) -> Result<Self> {
        for p in [&node, &cli] {
            if !p.is_absolute() || !p.is_file() || fs::symlink_metadata(p)?.file_type().is_symlink()
            {
                return Err(DaemonError::Invalid(
                    "model bridge must be trusted absolute regular files",
                ));
            }
        }
        Ok(Self { node, cli })
    }
    pub fn invoke(
        &self,
        private_state: &Path,
        run_id: &str,
        instruction: &str,
    ) -> Result<ModelProposal> {
        if !run_id.starts_with("r-")
            || run_id.len() < 3
            || !run_id[2..].bytes().all(|b| b.is_ascii_digit())
            || instruction.len() > 32 * 1024
        {
            return Err(DaemonError::Invalid("invalid provider invocation"));
        }
        let home = private_state.join("model-requests");
        fs::create_dir_all(&home)?;
        let dir = home.join(run_id);
        fs::create_dir(&dir)?; // create_new semantics; never repeat ambiguous model requests
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&home, fs::Permissions::from_mode(0o700))?;
            fs::set_permissions(&dir, fs::Permissions::from_mode(0o700))?;
        }
        let request = dir.join("request.json");
        let output = dir.join("result.json");
        let stderr = dir.join("transport.stderr");
        let mut file_opts = OpenOptions::new();
        file_opts.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            file_opts.mode(0o600);
        }
        let mut input = file_opts.open(&request)?;
        let body = Value::object([
            (
                "protocol".into(),
                Value::String("voder-provider-bridge-v1".into()),
            ),
            ("runId".into(), Value::String(run_id.into())),
            ("instruction".into(), Value::String(instruction.into())),
        ]);
        input.write_all(body.encode().as_bytes())?;
        input.sync_all()?;
        let err_file = file_opts.open(&stderr)?;
        let mut child_cmd = Command::new(&self.node);
        child_cmd
            .arg(&self.cli)
            .arg(&request)
            .arg(&output)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::from(err_file))
            .env_remove("NODE_OPTIONS");
        #[cfg(unix)]
        {
            child_cmd.process_group(0);
        }
        let mut child = child_cmd.spawn()?;
        let started = Instant::now();
        loop {
            if let Some(status) = child.try_wait()? {
                if !status.success() {
                    return Err(DaemonError::Invalid(
                        "model bridge failed; see private diagnostics",
                    ));
                }
                break;
            }
            let err_bytes = fs::metadata(&stderr).map(|m| m.len()).unwrap_or(0);
            let out_bytes = fs::metadata(&output).map(|m| m.len()).unwrap_or(0);
            if started.elapsed() >= TIMEOUT
                || err_bytes > MAX_STDERR_BYTES
                || out_bytes > MAX_RECEIPT_BYTES
            {
                #[cfg(unix)]
                {
                    unsafe extern "C" {
                        fn kill(pid: i32, sig: i32) -> i32;
                    }
                    // SAFETY: the child was started in a new process group with its own PID.
                    unsafe {
                        let _ = kill(-(child.id() as i32), 9);
                    }
                }
                let _ = child.kill();
                let _ = child.wait();
                return Err(DaemonError::Invalid(
                    "model bridge time or output limit exceeded",
                ));
            }
            thread::sleep(Duration::from_millis(20));
        }
        if !output.is_file()
            || fs::symlink_metadata(&output)?.file_type().is_symlink()
            || fs::metadata(&output)?.len() > MAX_RECEIPT_BYTES
        {
            return Err(DaemonError::Invalid(
                "model bridge did not produce bounded result",
            ));
        }
        let bytes = fs::read(&output)?;
        let parsed = json::parse(&bytes)
            .map_err(|_| DaemonError::Invalid("malformed model bridge receipt"))?;
        let text = |key: &str| parsed.get(key).and_then(Value::as_str).unwrap_or("");
        if text("protocol") != "voder-provider-bridge-v1"
            || text("runId") != run_id
            || text("providerId").is_empty()
            || text("model").is_empty()
            || parsed.get("deliveryAllowed") != Some(&Value::Bool(false))
            || parsed.get("executionAuthorized") != Some(&Value::Bool(false))
            || parsed.get("proposalText").and_then(Value::as_str).is_none()
            || !matches!(parsed.get("proposedToolCalls"), Some(Value::Array(_)))
        {
            return Err(DaemonError::Invalid(
                "untrusted model receipt failed contract",
            ));
        }
        // Full validation belongs to the TS bridge; model output remains untrusted and inert.
        Ok(ModelProposal { bytes })
    }
}
