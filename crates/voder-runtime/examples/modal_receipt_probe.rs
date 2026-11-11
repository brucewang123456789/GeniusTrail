//! Real, authenticated Modal roundtrip via the production Rust PluginSandbox.
//! Run only with explicit Modal credentials; this creates a billable sandbox.
use std::{
    env, fs,
    path::PathBuf,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use voder_runtime::{
    process::ProcessLimits,
    sandbox::{SandboxAvailability, SandboxBackend, SandboxPolicy},
    sandbox_plugin::PluginSandbox,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let node = PathBuf::from(env::var("VODER_NODE_BINARY")?);
    let cli = PathBuf::from(env::var("VODER_MODAL_CLI")?);
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let base = env::temp_dir().join(format!(
        "voder-vgr5-modal-probe-{}-{stamp}",
        std::process::id()
    ));
    fs::create_dir(&base)?;
    let outcome = (|| -> Result<(), Box<dyn std::error::Error>> {
        let project = base.join("workspace");
        let capture = base.join("capture");
        fs::create_dir(&project)?;
        fs::create_dir(&capture)?;
        fs::write(project.join("input.txt"), b"VODER_NATIVE_MODAL_VERIFIED\n")?;
        let limits =
            ProcessLimits::new(Duration::from_secs(120), 32 * 1024, 512 * 1024 * 1024, 120)?;
        let backend = PluginSandbox::new(node, cli, "modal")?;
        if !matches!(backend.probe(), SandboxAvailability::Available) {
            return Err("Modal backend missing credentials/app/pinned image".into());
        }
        let policy = SandboxPolicy::new(project, limits);
        let result = backend.execute(
            &policy,
            "/usr/bin/cat",
            &["input.txt".to_string()],
            &capture,
        )?;
        if result.exit_code() != Some(0) || result.timed_out() || result.output_limited() {
            return Err("Modal native validated execution did not complete successfully".into());
        }
        println!("VODER_NATIVE_MODAL_RECEIPT_IDENTITY_GATE=PASS");
        Ok(())
    })();
    // Never delete the workspace when verification failed: retain evidence.
    if outcome.is_ok() {
        fs::remove_dir_all(base)?;
    }
    outcome
}
