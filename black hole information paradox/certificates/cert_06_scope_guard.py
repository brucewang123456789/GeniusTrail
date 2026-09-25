"""
CERTIFICATE 6 - Scope guard.
Fails loudly if any scope label in results/theorem_status.json has been weakened.
The purpose is to make overclaiming a build error rather than an editorial slip.
"""
import json, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
st = json.loads((ROOT / "results" / "theorem_status.json").read_text())
REQUIRED = {
    ("black_hole_information_paradox",): "NOT_SOLVED",
    ("interacting_or_evaporating_dynamics",): "NOT_ADDRESSED",
    ("initial_to_out_faithfulness",): "NOT_PROVED",
    ("scope",): "FREE_ASYMPTOTIC_RADIATIVE_DATA_ONLY",
    ("theorems", "T1", "status"): "PROVED",
    ("theorems", "T1", "mathematical_novelty"): "NOT_CLAIMED_NEAR_FOLKLORE",
    ("theorems", "T4", "status"): "PROVED_AS_DICHOTOMY",
    ("theorems", "T4", "hypothesis_M1"): "CONTESTED_IN_LITERATURE_NOT_ADJUDICATED",
    ("theorems", "T8", "status"): "PROVED",
    ("theorems", "T9", "exponent_alpha"): "IMPORTED_NOT_DERIVED",
    ("world_priority",): "NOT_CLAIMED",
}
bad = []
for path, want in REQUIRED.items():
    cur = st
    for k in path:
        cur = cur.get(k, "<MISSING>") if isinstance(cur, dict) else "<MISSING>"
    if cur != want:
        bad.append(f"  {'.'.join(path)}: got {cur!r}, require {want!r}")
if bad:
    print("SCOPE GUARD FAIL:"); [print(b) for b in bad]; sys.exit(1)
print("CERTIFICATE 6 : SCOPE GUARD PASS - all scope labels intact")
