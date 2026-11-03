"""Stage 10 - Promotion-gated deployment.

* lifecycle report says promoted=false  -> nothing is deployed (the safe default).
* promoted=true                         -> build the bundle, then docker build + kubectl rollout.
* SKIP_DEPLOYMENT=1 in the environment  -> build the bundle only (used by CI, which has no cluster).
"""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIFECYCLE_REPORT = PROJECT_ROOT / "reports" / "lifecycle_report.json"
PREPARE_SCRIPT = PROJECT_ROOT / "src" / "mlops_pipeline" / "11_prepare_deployment.py"
DEPLOY_SCRIPT = PROJECT_ROOT / "deployment" / "deploy_production.py"
DEPLOY_REPORT = PROJECT_ROOT / "reports" / "deployment_automation_report.json"


def save_report(status, **extra):
    DEPLOY_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with DEPLOY_REPORT.open("w", encoding="utf-8") as f:
        json.dump({"status": status, "timestamp": datetime.now(timezone.utc).isoformat(), **extra}, f, indent=4)


def run_script(script_path):
    return subprocess.run([sys.executable, str(script_path)], cwd=PROJECT_ROOT).returncode == 0


def main():
    print("=" * 70)
    print("DEPLOY IF MODEL WAS PROMOTED")
    print("=" * 70)

    if not LIFECYCLE_REPORT.exists():
        print(f"[ERROR] Lifecycle report not found: {LIFECYCLE_REPORT}")
        return False

    with LIFECYCLE_REPORT.open("r", encoding="utf-8") as f:
        report = json.load(f)

    promoted = bool(report.get("promoted", False))
    candidate_version = report.get("candidate_version")
    production_version = report.get("production_version_after")

    print(f"[INFO] Candidate version : {candidate_version}")
    print(f"[INFO] Promoted          : {promoted}")
    print(f"[INFO] Production version: {production_version}")

    if not promoted:
        print("[INFO] Model was not promoted. Deployment skipped.")
        save_report("skipped_not_promoted", candidate_version=candidate_version, production_version=production_version)
        return True

    print(f"[INFO] Model version {candidate_version} was promoted. Preparing deployment bundle...")
    if not run_script(PREPARE_SCRIPT):
        print("[ERROR] Deployment preparation failed.")
        save_report("failed_prepare", candidate_version=candidate_version)
        return False
    print("[SUCCESS] Deployment bundle prepared.")

    if os.getenv("SKIP_DEPLOYMENT", "").strip().lower() in {"1", "true", "yes"}:
        print("[INFO] SKIP_DEPLOYMENT is set. Docker/Kubernetes steps skipped.")
        save_report("bundle_prepared_only", candidate_version=candidate_version, production_version=production_version)
        return True

    print("[INFO] Starting Production deployment...")
    if not run_script(DEPLOY_SCRIPT):
        print("[ERROR] Production deployment failed.")
        save_report("failed_deploy", candidate_version=candidate_version)
        return False

    print("[SUCCESS] Production deployment completed.")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
