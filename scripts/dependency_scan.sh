# File: scripts/dependency_scan.sh
# Purpose: Run Semgrep CI rules and fail on any ERROR (severity ≥ high)

set -eo pipefail

# Install semgrep if not present
if ! command -v semgrep &>/dev/null; then
  pip install --user semgrep
  export PATH="$HOME/.local/bin:$PATH"
fi

# Run semgrep with CI rulepack, fail on high-severity findings
semgrep --config p/ci --severity ERROR --error --timeout 120 .

echo "✅ No high-severity findings."
