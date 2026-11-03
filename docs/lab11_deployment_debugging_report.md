# Lab 11 - Deployment debugging guide
| Symptom | Cause / fix |
|---|---|
| `docker: not found` / permission denied from Python | Docker Desktop not running, or the venv python has a low integrity level (check from a normal PowerShell) |
| Pod `ImagePullBackOff` | image not built locally; keep `imagePullPolicy: IfNotPresent` and build the exact tag |
| Pod not Ready | `/health` is 503: the bundle in the image is missing/corrupt; `kubectl logs <pod>` |
| `InconsistentVersionWarning` | scikit-learn in the image differs from training; rebuild after `11_prepare_deployment.py` |
| Wrong model served | `kubectl get deployment bike-api -o jsonpath="{.spec.template.spec.containers[0].image}"` must show `model-vN` |
