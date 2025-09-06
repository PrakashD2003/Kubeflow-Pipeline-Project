import os
import logging
from datetime import datetime
import pickle
from sklearn.ensemble import RandomForestClassifier
import dagshub
import mlflow
import argparse
import json
import yaml
from mlflow.tracking import MlflowClient
from typing import Dict, Any, Optional

# Ensure that a directory named 'logs' exist in our root folder (if not it creates one)(for storing log file)
log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

# Logging Configuration
logger = logging.getLogger('Model_Pusher')
logger.setLevel('DEBUG')

# Creating Handlers
console_handler = logging.StreamHandler()
file_log_path = os.path.join(log_dir,'Model_Pusher.log')
file_handler = logging.FileHandler(file_log_path,encoding='utf-8')

# Setting Log Levels for Handlers
console_handler.setLevel('DEBUG')
file_handler.setLevel('DEBUG')

# Creating a Formatter and attaching it to handelers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Adding handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


logger.info("\n" + " "*52 + "="*60)
logger.info(f"NEW RUN STARTED AT {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("="*60 + "\n")

# Function to Load Parameters from params.yaml
def load_params(param_path:str) ->dict:
    try:
        logger.debug("Loading Params From: %s",param_path)
        with open(param_path,'r') as file:
            params = yaml.safe_load(file)
        logger.info("Params Loaded Successfully From: %s",param_path)
        return params
    except FileNotFoundError:
        logger.debug('File not found: %s',param_path)
        raise
    except yaml.YAMLError as e:
        logger.debug('Yaml error: %s',e)
        raise
    except Exception as e:
        logger.debug('Unexpected error occured while loadind parameters: %s',e)
        raise

# Function for Loadind Trained Model
def load_model(model_dir: str) -> RandomForestClassifier:
    """Load a trained model from a directory (expects model.pkl inside)."""
    try:
        model_path = os.path.join(model_dir, "model.pkl")  # or whatever name you saved it as
        logger.debug("Loading Model From: %s", model_path)
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.info("Model Loaded Successfully.")
        return model
    except FileNotFoundError:
        logger.debug("File not found: %s", model_path)
        raise
    except Exception as e:
        logger.debug("Unexpected error while loading the model: %s", e)
        raise

# Function to Load the Evaluation Metrics as Json File
def load_metrics(path: str) -> Dict[str, Any]:
    """Loads the Evaluation Metrics from a JSON file in a given output directory path from Kubeflow."""
    try:
        # Define full file path
        file_path = os.path.join(path, "metrics.json")

        # Ensure the directory exists
        os.makedirs(path, exist_ok=True)

        logger.debug("Loading evaluation metrics from file: %s", file_path)
        with open(file_path, 'r') as file:
            metrics = json.load(file)

        logger.info("Evaluation metrics successfully loaded from %s", file_path)
        return metrics

    except Exception as e:
        logger.error("Error while loading metrics: %s",e)
        raise

# Function to Register Model in MLflow
def register_model(self, run_id:str, artifact_path: str, model_name:str, stage:str):
    """
    Register a logged MLflow model under a given name and transition it to a stage.

    Args:
        run_id:     MLflow run ID containing the logged model artifact.
        model_name: Registered model name in the MLflow Model Registry.
        stage:      Stage to transition the new version to (e.g. "Staging" or "Production").

    Returns:
        The new model version as a string.

    Raises:
        DetailedException: If registration or transition fails.
    """

    try:
        result = mlflow.register_model(model_uri=f"runs:/{run_id}/{artifact_path}", name=model_name)

        mlflow.tracking.MlflowClient().transition_model_version_stage(name=model_name,
                                                                        version=result.version,
                                                                        stage=stage,
                                                                        archive_existing_versions=True
                                                                        )
        return result.version
    except Exception as e:
        logger.debug("Unexpected error while registering the model: %s",e)

# Function to setup DagsHub authentication using provided credentials
def setup_dagshub_auth(dagshub_username: str, dagshub_token: str):
    """Setup DagsHub authentication using provided credentials."""
    try:
        if not dagshub_username or not dagshub_token:
            logger.error("DagsHub credentials are required but not provided.")
            return False
            
        # Set DagsHub authentication
        dagshub.auth.add_app_token(dagshub_token)
        logger.info("DagsHub authentication configured successfully for user: %s", dagshub_username)
        return True
        
    except Exception as e:
        logger.error("Failed to setup DagsHub authentication: %s", e)
        return False

# Function to get production model metrics from MLflow registry
def get_production_model_metrics(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve metrics from the current production model in MLflow registry.
    
    Args:
        model_name: Name of the model in MLflow registry
        
    Returns:
        Dictionary containing production model metrics or None if no production model exists
    """
    try:
        client = MlflowClient()
        
        # Get the latest production model version
        latest_versions = client.get_latest_versions(model_name, stages=["Production"])
        
        if not latest_versions:
            logger.info("No production model found for %s", model_name)
            return None
            
        production_version = latest_versions[0]
        logger.info("Found production model version: %s", production_version.version)
        
        # Get run details for the production model
        run = client.get_run(production_version.run_id)
        
        # Extract metrics from the run
        metrics = run.data.metrics
        logger.info("Retrieved production model metrics: %s", list(metrics.keys()))
        
        return metrics
        
    except Exception as e:
        logger.error("Failed to get production model metrics: %s", e)
        return None

# Function to compare models and determine if new model should be promoted
def should_promote_model(new_metrics: Dict[str, Any], production_metrics: Optional[Dict[str, Any]], 
                        threshold: float, metric_name: str = 'accuracy') -> bool:
    """
    Compare new model metrics with production model and determine if promotion is warranted.
    
    Args:
        new_metrics: Metrics from the newly trained model
        production_metrics: Metrics from current production model (None if no production model)
        threshold: Minimum improvement threshold required for promotion
        metric_name: Primary metric to compare (default: 'accuracy')
        
    Returns:
        True if new model should be promoted to production, False otherwise
    """
    try:
        # If no production model exists, promote the new model
        if production_metrics is None:
            logger.info("No production model exists. Promoting new model to production.")
            return True
            
        # Check if the primary metric exists in both models
        if metric_name not in new_metrics:
            logger.error("Primary metric '%s' not found in new model metrics", metric_name)
            return False
            
        if metric_name not in production_metrics:
            logger.error("Primary metric '%s' not found in production model metrics", metric_name)
            return False
            
        new_score = new_metrics[metric_name]
        production_score = production_metrics[metric_name]
        
        # Calculate improvement
        improvement = new_score - production_score
        improvement_percentage = (improvement / production_score) * 100
        
        logger.info("Model comparison results:")
        logger.info("  Production model %s: %.4f", metric_name, production_score)
        logger.info("  New model %s: %.4f", metric_name, new_score)
        logger.info("  Improvement: %.4f (%.2f%%)", improvement, improvement_percentage)
        logger.info("  Required threshold: %.2f%%", threshold * 100)
        
        # Check if improvement meets threshold
        if improvement_percentage >= threshold * 100:
            logger.info("New model meets improvement threshold. Promoting to production.")
            return True
        else:
            logger.info("New model does not meet improvement threshold. Keeping current production model.")
            return False
            
    except Exception as e:
        logger.error("Error during model comparison: %s", e)
        return False

# Function to retire old production model
def retire_production_model(model_name: str):
    """
    Retire the current production model by archiving it.
    
    Args:
        model_name: Name of the model in MLflow registry
    """
    try:
        client = MlflowClient()
        
        # Get current production versions
        production_versions = client.get_latest_versions(model_name, stages=["Production"])
        
        for version in production_versions:
            logger.info("Retiring production model version: %s", version.version)
            client.transition_model_version_stage(
                name=model_name,
                version=version.version,
                stage="Archived"
            )
            
        logger.info("Successfully retired production model")
        
    except Exception as e:
        logger.error("Failed to retire production model: %s", e)
        raise

def main(repo_owner_name: str, repo_name: str, model_name: str, stage: str, 
         param_path: str, model_path: str, metrics_path: str,
         dagshub_username: str, dagshub_token: str):
    try:
        # Setup DagsHub authentication using provided credentials
        if not setup_dagshub_auth(dagshub_username, dagshub_token):
            logger.error("Failed to setup DagsHub authentication. Exiting.")
            return
        
        # Loading Parameters From params.yaml
        params = load_params(param_path)
        
        # Loading Trained Model
        model = load_model(model_path)
        
        # Loading Evaluation Metrics
        metrics = load_metrics(metrics_path)
        
        # Initialize DagsHub MLflow tracking
        dagshub.init(repo_owner=repo_owner_name, repo_name=repo_name, mlflow=True)
        
        # Get model comparison parameters from params
        comparison_threshold = params.get('model_comparison', {}).get('improvement_threshold', 0.05)  # 5% default
        primary_metric = params.get('model_comparison', {}).get('primary_metric', 'accuracy')
        
        logger.info("Model comparison configuration:")
        logger.info("  Improvement threshold: %.2f%%", comparison_threshold * 100)
        logger.info("  Primary metric: %s", primary_metric)
        
        # Get current production model metrics
        production_metrics = get_production_model_metrics(model_name)
        
        # Determine if new model should be promoted to production
        should_promote = should_promote_model(metrics, production_metrics, comparison_threshold, primary_metric)
        
        # Log model to MLflow
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")
            
            # Register model based on comparison results
            if should_promote:
                logger.info("Promoting new model to production")
                
                # Retire current production model if it exists
                if production_metrics is not None:
                    retire_production_model(model_name)
                
                # Register new model as production
                mlflow.register_model(
                    model_uri=f"runs:/{mlflow.active_run().info.run_id}/model", 
                    name=model_name
                )
                
                # Transition to production stage
                client = MlflowClient()
                latest_version = client.get_latest_versions(model_name)[0]
                client.transition_model_version_stage(
                    name=model_name,
                    version=latest_version.version,
                    stage="Production"
                )
                
                logger.info("New model successfully promoted to production")
            else:
                logger.info("Registering new model as staging (not promoted to production)")
                mlflow.register_model(
                    model_uri=f"runs:/{mlflow.active_run().info.run_id}/model", 
                    name=model_name
                )
                
                # Transition to staging stage
                client = MlflowClient()
                latest_version = client.get_latest_versions(model_name)[0]
                client.transition_model_version_stage(
                    name=model_name,
                    version=latest_version.version,
                    stage="Staging"
                )
                
                logger.info("New model registered as staging")
                
    except Exception as e:
        logger.error("Failed to complete the model pushing process: %s", e)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_owner_name", type=str, help="Name of the repository owner")
    parser.add_argument("repo_name", type=str, help="Name of the repository")
    parser.add_argument("model_name", type=str, help="Name of the model")
    parser.add_argument("stage", type=str, help="Stage to transition the new version to")
    parser.add_argument("param_path", type=str, help="Path to load parameters")
    parser.add_argument("model_path", type=str, help="Path to load trained Model")
    parser.add_argument("metrics_path", type=str, help="Path to load evaluation metrics")
    parser.add_argument("dagshub_username", type=str, help="DagsHub username for authentication")
    parser.add_argument("dagshub_token", type=str, help="DagsHub token for authentication")
    args = parser.parse_args()
    main(repo_owner_name=args.repo_owner_name, repo_name=args.repo_name, model_name=args.model_name, 
         stage=args.stage, param_path=args.param_path, model_path=args.model_path, 
         metrics_path=args.metrics_path, dagshub_username=args.dagshub_username, 
         dagshub_token=args.dagshub_token)