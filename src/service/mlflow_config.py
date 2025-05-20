import os
import mlflow
from mlflow.tracking import MlflowClient

def configure_mlflow():
    """
    Configure global MLflow tracking URI + autologging.
    """
    tracking_uri = os.environ.get("MLFLOW_SERVER_URI")
    if not tracking_uri:
        raise ValueError("MLFLOW_SERVER_URI is not set in environment variables.")
    
    mlflow.set_tracking_uri(tracking_uri)
    
    # Set autolog once globally for sklearn
    mlflow.sklearn.autolog(
        log_models=True,
        log_input_examples=False,
        log_model_signatures=True
    )

def get_mlflow_client() -> MlflowClient:
    """
    Return a properly configured MLflow client.
    Assumes configure_mlflow() has already been called.
    """
    return MlflowClient()
