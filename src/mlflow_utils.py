"""Shared MLflow helpers used by the lab scripts and the mlops_pipeline stages.

Why this file exists
--------------------
Recent MLflow releases save scikit-learn models with the `skops` format, which
refuses to serialise tree-based models (GradientBoosting, RandomForest, ...)
unless the type ``sklearn.tree._tree.Tree`` is explicitly declared as trusted.
Without that, ``mlflow.sklearn.log_model`` fails with
"The saved sklearn model references untrusted types".
Older MLflow releases do not know the ``skops_trusted_types`` argument at all,
so ``log_sklearn_model`` falls back to a plain call when the argument is rejected.
"""
import os

import mlflow
import mlflow.sklearn

# Reviewed and trusted: node storage of scikit-learn tree ensembles trained by this project.
SKOPS_TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]


def configure_mlflow(config=None):
    """Use MLFLOW_TRACKING_URI, else config['mlflow']['tracking_uri'], else MLflow's local default."""
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri and config:
        uri = (config.get("mlflow") or {}).get("tracking_uri")
    if uri:
        mlflow.set_tracking_uri(uri)
    return mlflow.get_tracking_uri()


def log_sklearn_model(model, name, registered_model_name=None):
    """Log (and optionally register) a scikit-learn model in an MLflow-version tolerant way."""
    kwargs = {"sk_model": model, "name": name}
    if registered_model_name:
        kwargs["registered_model_name"] = registered_model_name
    try:
        return mlflow.sklearn.log_model(skops_trusted_types=SKOPS_TRUSTED_TYPES, **kwargs)
    except TypeError:
        # MLflow version without the skops_trusted_types argument.
        return mlflow.sklearn.log_model(**kwargs)
