//! Bubblewrap backend, no host-shell fallback. Namespace probing is an actual execution.
//! Unsupported operations must return an error; seccomp profile enforcement remains a production gate.
use crate::{
    process,
    sandbox::{SandboxAvailability, SandboxBackend, SandboxPolicy},
    Result, RuntimeError,
};
use std::{path::Path, process::Command, time::Duration};
#[derive(Debug, Default)]
pub struct Bubblewrap;
impl Bubblewrap {
    fn command(&self, policy: &SandboxPolicy, program: &str, args: &[String]) -> Result<Command> {
        policy.validate()?;
        if program.is_empty() || !program.starts_with('/') || program.contains('\0') {
            return Err(RuntimeError::Invalid(
                "absolute executable required inside sandbox".into(),
            ));
        }
        let mut cmd = Command::new("/usr/bin/prlimit");
        cmd.arg(format!("--as={}", policy.memory_bytes));
        cmd.arg(format!("--cpu={}", policy.cpu_seconds));
        cmd.arg(format!("--fsize={}", policy.max_output_bytes));
        cmd.args([
            "--nofile=128",
            "--nproc=64",
            "--",
            "/usr/bin/bwrap",
            "--unshare-all",
            "--die-with-parent",
            "--new-session",
            "--clearenv",
        ]);
        // Root filesystem is empty except explicitly mounted runtime and workspace paths.
        for p in ["/usr", "/bin", "/lib", "/lib64"] {
            if Path::new(p).exists() {
                cmd.args(["--ro-bind", p, p]);
            }
        }
        cmd.args([
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--tmpfs",
            "/home",
            "--dir",
            "/workspace",
        ]);
        let workspace = policy.worktree.canonicalize()?;
        cmd.arg("--bind").arg(&workspace).arg("/workspace");
        // Git control metadata must never be writable by sandboxed code.
        if workspace.join(".git").exists() {
            cmd.arg("--ro-bind")
                .arg(workspace.join(".git"))
                .arg("/workspace/.git");
        }
        cmd.args([
            "--dir",
            "/opt",
            "--dir",
            "/opt/voder-tools",
            "--dir",
            "/home/voder",
            "--dir",
            "/home/voder/.cargo",
        ]);
        let mut path = String::new();
        for (i, src) in policy.toolchains.iter().enumerate() {
            let guest = format!("/opt/voder-tools/{i}");
            cmd.arg("--ro-bind").arg(src.canonicalize()?).arg(&guest);
            path.push_str(&format!("{guest}/bin:"));
        }
        path.push_str("/usr/bin:/bin");
        cmd.args([
            "--chdir",
            "/workspace",
            "--setenv",
            "HOME",
            "/home/voder",
            "--setenv",
            "TMPDIR",
            "/tmp",
            "--setenv",
            "PATH",
            &path,
            "--setenv",
            "CARGO_HOME",
            "/home/voder/.cargo",
            "--setenv",
            "CARGO_TARGET_DIR",
            "/tmp/voder-target",
            "--",
        ]);
        cmd.arg(program).args(args);
        cmd.env_clear();
        cmd.env("PATH", "/usr/bin:/bin");
        Ok(cmd)
    }
}
impl crate::sandbox::private::Sealed for Bubblewrap {}
impl SandboxBackend for Bubblewrap {
    fn probe(&self) -> SandboxAvailability {
        if !Command::new("/usr/bin/bwrap")
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success())
        {
            return SandboxAvailability::Unavailable("bwrap executable unavailable".into());
        }
        if !Command::new("/usr/bin/prlimit")
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success())
        {
            return SandboxAvailability::Unavailable("prlimit executable unavailable".into());
        }
        let mut cmd = Command::new("/usr/bin/bwrap");
        cmd.args([
            "--unshare-all",
            "--die-with-parent",
            "--new-session",
            "--clearenv",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
        ]);
        for p in ["/usr", "/bin", "/lib", "/lib64"] {
            if Path::new(p).exists() {
                cmd.args(["--ro-bind", p, p]);
            }
        }
        cmd.args(["--", "/bin/true"]);
        match cmd.output() {
            Ok(out) if out.status.success() => SandboxAvailability::Available,
            Ok(out) => SandboxAvailability::Unavailable(format!(
                "namespace probe failed: {}",
                String::from_utf8_lossy(&out.stderr)
            )),
            Err(e) => SandboxAvailability::Unavailable(format!("namespace probe error: {e}")),
        }
    }
    fn execute(
        &self,
        policy: &SandboxPolicy,
        program: &str,
        args: &[String],
        capture_dir: &Path,
    ) -> Result<process::ProcessResult> {
        if self.probe() != SandboxAvailability::Available {
            return Err(RuntimeError::CapabilityUnavailable(
                "verified OS sandbox unavailable",
            ));
        }
        let cmd = self.command(policy, program, args)?;
        process::run(cmd, capture_dir, policy.limits()?)
    }
}
/// Standalone capability check; does not attempt an unsafe fallback.
pub fn capability_gate() -> Result<()> {
    if Bubblewrap.probe() != SandboxAvailability::Available {
        return Err(RuntimeError::CapabilityUnavailable(
            "bwrap OS namespaces unavailable",
        ));
    }
    // TODO: enforce seccomp filters before high-threat production approval.
    Ok(())
}
#[allow(dead_code)]
const DEFAULT_SANDBOX_TIMEOUT: Duration = Duration::from_secs(30);
