//! Trusted tool IDs and side effects. Model descriptions cannot change classification.
use crate::{Result, RuntimeError};
use std::collections::BTreeMap;
use voder_core::{action_commit::PreparedAction, types::ToolEffect};
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolKind {
    ReadFile,
    Patch,
    SandboxedCommand,
    Validate,
}
#[derive(Debug, Clone)]
pub struct ToolSpec {
    pub kind: ToolKind,
    pub effect: ToolEffect,
    pub sandbox_required: bool,
}
#[derive(Debug, Default)]
pub struct ToolRegistry {
    tools: BTreeMap<String, ToolSpec>,
}
impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn insert(&mut self, id: &str, kind: ToolKind, effect: ToolEffect) -> Result<()> {
        if id.is_empty()
            || id.len() > 128
            || !id
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-')
        {
            return Err(RuntimeError::Invalid("invalid tool id".into()));
        }
        if kind == ToolKind::ReadFile && effect != ToolEffect::Read {
            return Err(RuntimeError::Unauthorized("read-file effect mismatch"));
        }
        if matches!(kind, ToolKind::Patch | ToolKind::SandboxedCommand)
            && effect != ToolEffect::Write
        {
            return Err(RuntimeError::Unauthorized("write effect mismatch"));
        }
        if self.tools.contains_key(id) {
            return Err(RuntimeError::Invalid("duplicate tool id".into()));
        }
        self.tools.insert(
            id.into(),
            ToolSpec {
                kind,
                effect,
                sandbox_required: matches!(kind, ToolKind::SandboxedCommand | ToolKind::Validate),
            },
        );
        Ok(())
    }
    pub fn authorize<'a>(
        &'a self,
        action: &PreparedAction,
        kind: ToolKind,
    ) -> Result<&'a ToolSpec> {
        let spec = self
            .tools
            .get(action.intent().descriptor.tool_id.as_str())
            .ok_or(RuntimeError::Unauthorized("tool not registered"))?;
        if spec.kind != kind || spec.effect != action.effect() {
            return Err(RuntimeError::Unauthorized("effect or tool kind mismatch"));
        }
        Ok(spec)
    }
}
