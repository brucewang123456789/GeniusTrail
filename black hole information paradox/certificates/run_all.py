"""Run every certificate in order. Exit code 0 iff all pass."""
import subprocess, sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent
S = ["cert_01_sector_connectivity.py", "cert_02_spanning_tree_bound.py",
     "cert_03_fock_commutant.py", "cert_04_memory_dichotomy.py",
     "cert_05_spectral_gap.py", "cert_06_scope_guard.py"]
fail = 0
for s in S:
    print(f"\n{'='*72}\n RUN {s}\n{'='*72}")
    r = subprocess.run([sys.executable, str(HERE / s)], capture_output=True, text=True)
    print(r.stdout.strip())
    if r.stderr.strip(): print(r.stderr.strip(), file=sys.stderr)
    if r.returncode != 0 or "FAIL" in r.stdout:
        fail += 1; print(">>> FAILED")
print(f"\n{'='*72}\nSUITE: {len(S)-fail}/{len(S)} PASS")
sys.exit(1 if fail else 0)
