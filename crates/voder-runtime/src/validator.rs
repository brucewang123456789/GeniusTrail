//! A frozen, trusted validator registry. Cannot convert textual PASS into an admitted proof.
use crate::{
    artifact_store::{sha256, ArtifactStore},
    sandbox::{SandboxBackend, SandboxPolicy},
    Result, RuntimeError,
};
use std::{collections::BTreeMap, path::Path};
use voder_core::{
    contract::FrozenContract,
    types::{ValidatorId, ValidatorVersion},
};
#[derive(Debug, Clone)]
pub struct ValidatorCommand {
    pub id: ValidatorId,
    pub version: ValidatorVersion,
    pub program: String,
    pub args: Vec<String>,
}
#[derive(Debug, Default)]
pub struct ValidatorRegistry {
    commands: BTreeMap<ValidatorId, ValidatorCommand>,
}
#[derive(Debug, Clone)]
pub struct ValidatorObservation {
    pub passed: bool,
    pub stdout_digest: String,
    pub stderr_digest: String,
    pub validator: ValidatorId,
    pub version: ValidatorVersion,
    pub executable_digest: String,
}
impl ValidatorRegistry {
    pub fn register(&mut self, cmd: ValidatorCommand) -> Result<()> {
        if !cmd.program.starts_with('/') || cmd.args.len() > 64 {
            return Err(RuntimeError::Invalid(
                "untrusted validator definition".into(),
            ));
        }
        if self.commands.contains_key(&cmd.id) {
            return Err(RuntimeError::Invalid("duplicate validator".into()));
        }
        self.commands.insert(cmd.id.clone(), cmd);
        Ok(())
    }
    pub fn run(
        &self,
        contract: &FrozenContract,
        id: &ValidatorId,
        policy: &SandboxPolicy,
        backend: &impl SandboxBackend,
        store: &ArtifactStore,
        capture_dir: &Path,
    ) -> Result<ValidatorObservation> {
        let configured = self
            .commands
            .get(id)
            .ok_or(RuntimeError::Unauthorized("unknown validator"))?;
        if !contract
            .validators()
            .iter()
            .any(|v| v.id == configured.id && v.version == configured.version)
        {
            return Err(RuntimeError::Unauthorized(
                "validator is not bound to frozen contract/version",
            ));
        }
        let result = backend.execute(policy, &configured.program, &configured.args, capture_dir)?;
        let stdout_digest = store.save(&result.stdout)?;
        let stderr_digest = store.save(&result.stderr)?;
        let executable_digest = sha256(
            format!(
                "{}\0{}\0{:?}",
                configured.program, configured.version, configured.args
            )
            .as_bytes(),
        )?;
        Ok(ValidatorObservation {
            passed: !result.timed_out && !result.output_limited && result.exit == Some(0),
            stdout_digest,
            stderr_digest,
            validator: configured.id.clone(),
            version: configured.version.clone(),
            executable_digest,
        })
    }
}
