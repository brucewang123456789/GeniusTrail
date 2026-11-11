---
name: scientific-validation
description: Apply scientific and numerical evidence checks when code makes mathematical or physical claims.
triggers: math,mathematics,physics,numerical,simulation,scientific,statistics,quantitative,数学,物理,数值,统计,仿真,模拟
---
# Quantitative software evidence
1. Pose the actual quantity, units, system, assumptions and decision. Reject a computable proxy when it does not answer the question.
2. Name model applicability and the dominant sensitivity or uncertainty; propose a concrete result that would refute the model. Do not confuse exact arithmetic with correct assumptions.
3. Prefer executable calculation and independent cross-check where stakes warrant. Check dimensions, sign, scaling, limiting/edge cases, conservation or invariants as relevant; include discretization, truncation, floating-point and sampling errors.
4. Distinguish measured from fitted, simulated from proved, precision from accuracy and point estimates from justified uncertainty ranges. Never present unsupported digits as evidence.
5. If verification contradicts the model, revise assumptions before recomputing. Report RESOLVED, BOUNDED or UNRESOLVED evidence honestly; never bypass the executable task's native VGR commit requirement.
