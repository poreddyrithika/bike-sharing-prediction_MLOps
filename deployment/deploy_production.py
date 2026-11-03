"""Package the Production model as a Docker image and roll it out on Kubernetes.

    python deployment/deploy_production.py            # real deployment (needs docker + kubectl)
    python deployment/deploy_production.py --dry-run  # validate the bundle and print the commands only
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT_DIR = PROJECT_ROOT / "deployment"
METADATA_PATH = DEPLOYMENT_DIR / "model_metadata.json"
K8S_DEPLOYMENT = PROJECT_ROOT / "k8s" / "deployment.yaml"
K8S_SERVICE = PROJECT_ROOT / "k8s" / "service.yaml"
REPORT_PATH = PROJECT_ROOT / "reports" / "deployment_automation_report.json"


def run(command, dry_run, check=True):
    print(f"[INFO] {'(dry-run) ' if dry_run else ''}Running: {' '.join(command)}")
    if dry_run:
        return 0
    code = subprocess.run(command, cwd=PROJECT_ROOT).returncode
    if check and code != 0:
        raise RuntimeError(f"Command failed with exit code {code}: {' '.join(command)}")
    return code


def deploy(dry_run):
    print("=" * 70 + "\nPRODUCTION DEPLOYMENT\n" + "=" * 70)
    for required in (METADATA_PATH, DEPLOYMENT_DIR / "model.pkl", DEPLOYMENT_DIR / "preprocessor.pkl"):
        if not required.exists():
            raise FileNotFoundError(f"Deployment artifact not found: {required}")

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if metadata["model_stage"] != "Production":
        raise RuntimeError(f"Deployment blocked. Model stage is '{metadata['model_stage']}', not 'Production'.")

    with open(PROJECT_ROOT / "config.yaml", encoding="utf-8") as f:
        image_name = yaml.safe_load(f)["deployment"]["image_name"]
    k8s = yaml.safe_load(K8S_DEPLOYMENT.read_text(encoding="utf-8"))
    deployment_name = k8s["metadata"]["name"]
    container = k8s["spec"]["template"]["spec"]["containers"][0]["name"]
    tag = f"{image_name}:model-v{metadata['model_version']}"

    print(f"[INFO] Model {metadata['model_name']} v{metadata['model_version']} | image {tag} | k8s {deployment_name}/{container}")

    exists = dry_run or subprocess.run(["docker", "image", "inspect", tag], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    if exists and not dry_run:
        print(f"[INFO] Image {tag} already exists. Reusing it.")
    else:
        run(["docker", "build", "-t", tag, "."], dry_run)

    present = dry_run or subprocess.run(["kubectl", "get", f"deployment/{deployment_name}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    if not present:
        print("[INFO] Deployment not found in the cluster. Applying manifests first.")
        run(["kubectl", "apply", "-f", str(K8S_DEPLOYMENT)], dry_run)
        run(["kubectl", "apply", "-f", str(K8S_SERVICE)], dry_run)
    run(["kubectl", "set", "image", f"deployment/{deployment_name}", f"{container}={tag}"], dry_run)
    run(["kubectl", "rollout", "status", f"deployment/{deployment_name}", "--timeout=120s"], dry_run)

    report = {"status": "dry_run_ok" if dry_run else "success", "model_name": metadata["model_name"],
              "model_version": metadata["model_version"], "run_id": metadata["run_id"], "docker_image": tag,
              "kubernetes_deployment": deployment_name, "kubernetes_container": container}
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=4), encoding="utf-8")
    print(f"[SUCCESS] Deployment {'dry run' if dry_run else 'rollout'} complete. Report: {REPORT_PATH}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        sys.exit(0 if deploy(args.dry_run) else 1)
    except Exception as exc:
        print(f"[ERROR] Production deployment failed: {exc}")
        sys.exit(1)
