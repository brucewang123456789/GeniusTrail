//! Time/size bounded subprocess capture; no shell interpolation and no detached execution.
use crate::{Result, RuntimeError};
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::{
    fs::{self, File, OpenOptions},
    path::Path,
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

#[derive(Debug, Clone, Copy)]
pub struct ProcessLimits {
    pub timeout: Duration,
    pub max_output_bytes: u64,
    pub memory_bytes: u64,
    pub cpu_seconds: u64,
}
impl ProcessLimits {
    pub fn new(
        timeout: Duration,
        max_output_bytes: u64,
        memory_bytes: u64,
        cpu_seconds: u64,
    ) -> Result<Self> {
        if timeout.is_zero()
            || timeout > Duration::from_secs(3600)
            || max_output_bytes == 0
            || max_output_bytes > 16 * 1024 * 1024
            || memory_bytes < 64 * 1024 * 1024
            || cpu_seconds == 0
        {
            return Err(RuntimeError::Invalid("unsafe resource limit".into()));
        }
        Ok(Self {
            timeout,
            max_output_bytes,
            memory_bytes,
            cpu_seconds,
        })
    }
}
#[derive(Debug)]
pub struct ProcessResult {
    pub(crate) exit: Option<i32>,
    pub(crate) timed_out: bool,
    pub(crate) output_limited: bool,
    pub(crate) stdout: Vec<u8>,
    pub(crate) stderr: Vec<u8>,
    pub(crate) elapsed: Duration,
}

impl ProcessResult {
    pub fn exit_code(&self) -> Option<i32> {
        self.exit
    }
    pub fn timed_out(&self) -> bool {
        self.timed_out
    }
    pub fn output_limited(&self) -> bool {
        self.output_limited
    }
    pub fn elapsed(&self) -> Duration {
        self.elapsed
    }
}

#[cfg(unix)]
unsafe extern "C" {
    fn kill(pid: i32, sig: i32) -> i32;
}

#[cfg(unix)]
fn kill_tree(pid: u32) {
    // process_group(0) created a new group with child pid
    // SAFETY: sending SIGKILL to a known positive child process group; no unrelated group ids.
    unsafe {
        let _ = kill(-(pid as i32), 9);
    }
}

/// Runs a *constructed* command. Only the sandbox adapter may pass commands carrying model input.
pub(crate) fn run(
    cmd: Command,
    capture_dir: &Path,
    limits: ProcessLimits,
) -> Result<ProcessResult> {
    run_with_transport_deadline(cmd, capture_dir, limits, Some(limits.timeout))
}

/// Sandbox-plugin transport is a client operation, not the sandboxed program.
/// The remote program remains limited by the signed task policy and Modal's
/// own limits. Waiting for its receipt must not share that deadline: workspace
/// upload, service scheduling and verification can outlive the remote command.
/// Caller cancellation continues to be handled by the trusted host transport.
pub(crate) fn run_until_completion(
    cmd: Command,
    capture_dir: &Path,
    limits: ProcessLimits,
) -> Result<ProcessResult> {
    run_with_transport_deadline(cmd, capture_dir, limits, None)
}

fn run_with_transport_deadline(
    mut cmd: Command,
    capture_dir: &Path,
    limits: ProcessLimits,
    transport_deadline: Option<Duration>,
) -> Result<ProcessResult> {
    fs::create_dir_all(capture_dir)?;
    if fs::symlink_metadata(capture_dir)?.file_type().is_symlink() {
        return Err(RuntimeError::UnsafePath("log directory symlink".into()));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(capture_dir, fs::Permissions::from_mode(0o700))?;
    }
    let stdout_file = capture_dir.join("stdout.raw");
    let stderr_file = capture_dir.join("stderr.raw");
    let mut output_opts = OpenOptions::new();
    output_opts.create_new(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        output_opts.mode(0o600);
    }
    let stdout = output_opts.open(&stdout_file)?;
    let stderr = output_opts.open(&stderr_file)?;
    cmd.stdout(Stdio::from(stdout));
    cmd.stderr(Stdio::from(stderr));
    cmd.stdin(Stdio::null());
    #[cfg(unix)]
    {
        cmd.process_group(0);
    }
    let start = Instant::now();
    let mut child = cmd.spawn()?;
    let pid = child.id();
    let (status, timed_out, output_limited) = loop {
        let out_len = fs::metadata(&stdout_file)?.len();
        let err_len = fs::metadata(&stderr_file)?.len();
        let too_big = out_len.saturating_add(err_len) > limits.max_output_bytes;
        let timed_out = transport_deadline.is_some_and(|deadline| start.elapsed() >= deadline);
        if too_big || timed_out {
            #[cfg(unix)]
            kill_tree(pid);
            let _ = child.kill();
            break (child.wait()?, timed_out, too_big);
        }
        if let Some(status) = child.try_wait()? {
            break (status, false, false);
        }
        thread::sleep(Duration::from_millis(20));
    };
    fn read_bounded(p: &Path, max: u64) -> Result<Vec<u8>> {
        use std::io::Read;
        let mut data = Vec::new();
        File::open(p)?.take(max).read_to_end(&mut data)?;
        Ok(data)
    }
    let stdout = read_bounded(&stdout_file, limits.max_output_bytes)?;
    let stderr = read_bounded(
        &stderr_file,
        limits.max_output_bytes.saturating_sub(stdout.len() as u64),
    )?;
    Ok(ProcessResult {
        exit: status.code(),
        timed_out,
        output_limited,
        stdout,
        stderr,
        elapsed: start.elapsed(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tmp(label: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!(
            "voder-pipeline-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&p).unwrap();
        p
    }
    #[test]
    fn child_process_group_is_terminated_on_timeout() {
        let dir = tmp("timeout");
        let mut cmd = Command::new("/bin/sh");
        cmd.args(["-c", "sleep 5 & wait"]);
        let result = run(
            cmd,
            &dir,
            ProcessLimits::new(Duration::from_millis(150), 1024, 128 * 1024 * 1024, 2).unwrap(),
        )
        .unwrap();
        assert!(result.timed_out);
        assert_ne!(result.exit, Some(0));
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn plugin_transport_waits_beyond_remote_program_limit() {
        let dir = tmp("receipt-wait");
        let mut cmd = Command::new("/bin/sh");
        cmd.args(["-c", "sleep 0.35; printf receipt-ready"]);
        let result = run_until_completion(
            cmd,
            &dir,
            ProcessLimits::new(Duration::from_millis(100), 1024, 128 * 1024 * 1024, 2).unwrap(),
        ).unwrap();
        assert_eq!(result.exit, Some(0));
        assert!(!result.timed_out);
        assert_eq!(result.stdout, b"receipt-ready");
        assert!(result.elapsed >= Duration::from_millis(300));
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn huge_output_is_truncated_and_the_process_is_killed() {
        let dir = tmp("output");
        let mut cmd = Command::new("/bin/sh");
        cmd.args([
            "-c",
            "while :; do echo 1234567890123456789012345678901234567890; done",
        ]);
        let result = run(
            cmd,
            &dir,
            ProcessLimits::new(Duration::from_secs(2), 4096, 128 * 1024 * 1024, 2).unwrap(),
        )
        .unwrap();
        assert!(result.output_limited);
        assert!(result.stdout.len() <= 4096);
        fs::remove_dir_all(dir).unwrap();
    }
}
