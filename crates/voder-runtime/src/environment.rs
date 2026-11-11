//! Detect missing toolchains before execution; never silently install unpinned packages.
use crate::{paths, Result, RuntimeError};
use std::{path::Path, process::Command};
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    Rust,
    Python,
    Node,
}
#[derive(Debug, Clone)]
pub struct EnvironmentPlan {
    pub language: Language,
    pub required: Vec<String>,
    pub command: String,
    pub args: Vec<String>,
}
impl EnvironmentPlan {
    pub fn inspect(project: &Path, language: Language) -> Result<Self> {
        let (required, command, args) = match language {
            Language::Rust => (
                vec!["Cargo.toml"],
                "/usr/bin/env",
                vec!["cargo", "test", "--offline", "--all-targets"],
            ),
            Language::Python => (
                vec!["pyproject.toml"],
                "/usr/bin/python3",
                vec!["-I", "-m", "pytest", "-q"],
            ),
            Language::Node => (
                vec!["package.json", "pnpm-lock.yaml"],
                "/usr/bin/node",
                vec!["--test"],
            ),
        };
        let missing: Vec<String> = required
            .into_iter()
            .filter(|f| paths::checked_existing(project, f).is_err())
            .map(str::to_string)
            .collect();
        if !missing.is_empty() {
            return Err(RuntimeError::Invalid(format!(
                "missing required project files: {}",
                missing.join(", ")
            )));
        }
        Ok(Self {
            language,
            required: match language {
                Language::Rust => vec!["Cargo.toml".into()],
                Language::Python => vec!["pyproject.toml".into()],
                Language::Node => vec!["package.json".into(), "pnpm-lock.yaml".into()],
            },
            command: command.into(),
            args: args.into_iter().map(str::to_string).collect(),
        })
    }
    pub fn check_host_runtime(&self) -> Result<()> {
        let command = match self.language {
            Language::Rust => "cargo",
            Language::Python => "python3",
            Language::Node => "node",
        };
        if !Command::new(command)
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success())
        {
            return Err(RuntimeError::CapabilityUnavailable(
                "required language toolchain missing",
            ));
        }
        Ok(())
    }
}
