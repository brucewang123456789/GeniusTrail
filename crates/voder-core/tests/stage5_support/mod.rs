#![allow(dead_code)]
use voder_core::action_commit::{ActionIntent, ExpectedChange, MutationKind};
use voder_core::budget::{BudgetAmount, BudgetLedger};
use voder_core::contract::{
    ContractDraft, FrozenContract, InputKind, RepoScope, RequiredInput, Requirement, TaskAnchor,
    ToolRule, ValidatorBinding,
};
use voder_core::decision::{ChoiceEstimate, DecisionCandidate};
use voder_core::evidence::EvidenceLedger;
use voder_core::obligation::{ObligationRegistry, ObligationSpec};
use voder_core::retry::{ActionDescriptor, RetryLedger};
use voder_core::types::*;

pub fn id<T: TryFrom<&'static str>>(text: &'static str) -> T
where
    <T as TryFrom<&'static str>>::Error: std::fmt::Debug,
{
    T::try_from(text).unwrap()
}

pub fn world() -> WorkspaceIdentity {
    WorkspaceIdentity {
        revision: id("r1"),
        digest: id("w1"),
    }
}
pub fn contract(require_file: bool, approval: bool) -> FrozenContract {
    FrozenContract::compile(ContractDraft {
        anchor: TaskAnchor {
            task_id: id("t1"),
            identity: "task-1".into(),
            goal: "Fix tests".into(),
            version: Version::new(1).unwrap(),
            accepted_facts: vec![],
            environment_constraints: vec![],
            evaluation_boundary: vec![],
        },
        identity: ContractIdentity {
            contract_id: id("c1"),
            version: Version::new(1).unwrap(),
            digest: id("c1digest"),
            toolchain: id("rust1"),
            environment: id("env1"),
        },
        requirements: vec![Requirement {
            id: id("req"),
            description: "tests pass".into(),
            hard: true,
            severity: Severity::Critical,
        }],
        validators: vec![ValidatorBinding {
            id: id("validator"),
            version: id("v1"),
            requirement_ids: vec![id("req")],
        }],
        trusted_sources: vec![id("runner")],
        tools: vec![
            ToolRule {
                id: id("reader"),
                effect: ToolEffect::Read,
                requires_approval: false,
            },
            ToolRule {
                id: id("writer"),
                effect: ToolEffect::Write,
                requires_approval: approval,
            },
        ],
        read_scopes: vec![RepoScope::parse("src").unwrap()],
        write_scopes: vec![RepoScope::parse("src").unwrap()],
        forbidden_effects: vec![],
        required_inputs: if require_file {
            vec![RequiredInput {
                id: id("file"),
                kind: InputKind::File,
                description: "src/main.rs".into(),
                why_required: "missing source".into(),
            }]
        } else {
            vec![]
        },
        sampling_rules: vec![],
    })
    .unwrap()
}
pub fn obligations(contract: &FrozenContract) -> ObligationRegistry {
    ObligationRegistry::compile(
        contract,
        vec![ObligationSpec {
            id: id("ob"),
            claim_id: "all-tests-pass".into(),
            requirement_id: Some(id("req")),
            threatened_judgement: "delivery".into(),
            expected_observation_type: "cargo-test".into(),
            admission_rule_id: "rule1".into(),
            scope: RepoScope::parse("src").unwrap(),
            priority: 10,
            stop_condition: "tests pass".into(),
            source: id("runner"),
            critical: true,
            closure_relation: EvidenceRelation::Supports,
            dependencies: vec![],
        }],
    )
    .unwrap()
}
pub fn budget() -> BudgetLedger {
    BudgetLedger::new(BudgetAmount {
        steps: 20,
        tool_calls: 20,
        tokens: 10_000,
        model_calls: 20,
        risk_microunits: 1_000_000,
        ..BudgetAmount::default()
    })
}
pub fn cost() -> BudgetAmount {
    BudgetAmount {
        steps: 1,
        tool_calls: 1,
        tokens: 100,
        model_calls: 1,
        risk_microunits: 100,
        ..BudgetAmount::default()
    }
}
pub fn choice_value(value: u8) -> ChoiceEstimate {
    ChoiceEstimate {
        closure: value,
        information: value,
        progress: value,
        reversibility: 70,
        token_cost: 5,
        latency_cost: 5,
        risk: 5,
        context_cost: 5,
    }
}
pub fn candidate(
    contract: &FrozenContract,
    action_id: &'static str,
    write: bool,
) -> DecisionCandidate {
    let tool = if write { "writer" } else { "reader" };
    DecisionCandidate {
        intent: ActionIntent {
            run_id: id("run"),
            action_id: id(action_id),
            descriptor: ActionDescriptor {
                kind: if write { "patch" } else { "inspect" }.into(),
                target_obligation_id: Some(id("ob")),
                hypothesis_id: None,
                tool_id: id(tool),
                parameter_key: id(action_id),
                scope: RepoScope::parse("src").unwrap(),
            },
            contract: contract.identity().clone(),
            workspace: world(),
            reservation_id: id(action_id),
            expected_changes: if write {
                vec![ExpectedChange {
                    path: RepoScope::parse("src/main.rs").unwrap(),
                    kind: MutationKind::Modify,
                }]
            } else {
                vec![]
            },
            required_validators: if write { vec![id("validator")] } else { vec![] },
            rollback_ref: if write {
                Some("snapshot-1".into())
            } else {
                None
            },
        },
        estimate: choice_value(50),
        max_cost: cost(),
        sample_ordinal: None,
    }
}
pub fn engine(
    contract: &FrozenContract,
) -> (
    ObligationRegistry,
    BudgetLedger,
    RetryLedger,
    EvidenceLedger,
) {
    (
        obligations(contract),
        budget(),
        RetryLedger::new(),
        EvidenceLedger::new(),
    )
}
