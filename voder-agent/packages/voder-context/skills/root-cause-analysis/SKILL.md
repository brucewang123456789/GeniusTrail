---
name: root-cause-analysis
description: Reproduce and isolate a software failure through falsifiable diagnostics.
triggers: bug,repair,fix,debug,error,exception,traceback,stderr,修复,报错,根因,错误
---
# Root-cause investigation
1. Anchor the problem in one reproducible failing command, environment/version, exact exit/status, stack, assertion or stderr; observed evidence is not a guessed root cause.
2. Trace the failing execution/data flow only as far as needed to isolate the violated invariant; state a likely cause and a plausible competing cause.
3. Select one falsifiable, lowest-cost discriminating diagnostic with a predicted observable result. Run only through authorized tools and evaluate the *actual* receipt; record contradictions as negative evidence.
4. If falsified, revise the model first; do not loop on the same patch, command or unchanged hypothesis. A transient retry requires new justification and a bounded count.
5. Propose the smallest reversible correction with expected effects and rollback. Verify targeted failure, regressions and original acceptance using a real runner where permitted.
6. If no environment, evidence, permissions or remaining budget exists, report BLOCKED/UNVERIFIED rather than a fabricated success; final native VGR commit is host-controlled.
