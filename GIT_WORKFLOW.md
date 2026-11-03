# Git workflow (Lab 3) - template
Your own commit hashes/history are the evidence for this lab; fill them in from `git log --oneline --graph`.
Suggested flow: `git init`; commit the Lab 1-2 baseline; branch `feature/modular-scripts` (split preprocess/train/evaluate) and merge;
create two conflicting edits to one line (e.g. how to scale `windspeed`) on two branches, merge, resolve; use `git bisect` / `git revert` for a faulty commit.
Never commit `venv/`, `mlflow.db`, tokens (see `.gitignore`).
