import time
import joblib
import logging
import pandas as pd
from dotenv import load_dotenv
from typing import Literal, Any, Optional, Tuple, Dict, List
import mlflow
from mlflow.exceptions import MlflowException
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    classification_report
)
from src.enums import Feature, PlayHand

from src.repository.model_data import load_model_data
from src.service.mlflow_config import configure_mlflow, get_mlflow_client

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Explicitly set the logger level to INFO

load_dotenv()

# Initialize MLflow
configure_mlflow()

# ------------------------------------------------------------------------------
models = {}

all_algorithms = Literal[
    'LogisticRegression',
    'RandomForest',
    'SVM',
    'GradientBoosting',
    'MLP',
    'DecisionTree',
    'ExtraTrees',
    'Bagging',
]

client = None

def create_pairwise_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a balanced dataset with pairwise comparisons
    """
    records = []
    for _, row in df.iterrows():
        # Record 1 : original order (winner in position 1, loser in position 2)
        record_1 = {
            Feature.SERIES.name: row['tournament_series'],
            Feature.SURFACE.name: row['tournament_surface'],
            Feature.COURT.name: row['tournament_court'],
            Feature.DIFF_RANKING.name: row['diff_rank'],
            Feature.MEAN_RANKING.name: row['mean_rank'],
            Feature.DIFF_PLAY_HAND.name: row['diff_play_hand'],
            Feature.DIFF_BACK_HAND.name: row['diff_back_hand'],
            Feature.DIFF_HEIGHT.name: row['diff_height_cm'],
            Feature.MEAN_HEIGHT.name: row['mean_height_cm'],
            Feature.DIFF_WEIGHT.name: row['diff_weight_kg'],
            Feature.MEAN_WEIGHT.name: row['mean_weight_kg'],
            Feature.DIFF_AGE.name: row['diff_age'],
            Feature.DIFF_NB_PRO_YEARS.name: row['diff_nb_pro_years'],
            'target': 1 # Player in first position won
        }

        # Record 2 : invert players
        record_2 = record_1.copy()
        record_2[Feature.DIFF_RANKING.name] *= -1
        record_2[Feature.DIFF_PLAY_HAND.name] *= -1
        record_2[Feature.DIFF_BACK_HAND.name] *= -1
        record_2[Feature.DIFF_HEIGHT.name] *= -1
        record_2[Feature.DIFF_WEIGHT.name] *= -1
        record_2[Feature.DIFF_AGE.name] *= -1
        record_2[Feature.DIFF_NB_PRO_YEARS.name] *= -1
        record_2['target'] = 0 # Player in first position lost

        records.append(record_1)
        records.append(record_2)
    
    return pd.DataFrame(records)

def create_pipeline(algo: all_algorithms = 'MLP') -> Pipeline:
    """
    Creates a machine learning pipeline with SimpleImputer, StandardScaler, OneHotEncoder and LogisticRegression.

    Returns:
        Pipeline: A scikit-learn pipeline object.
    """
    # Define the features, numerical and categorical
    cat_features = [f.name for f in Feature.get_features_by_type('category')]
    num_features = [f.name for f in Feature.get_features_by_type('number')]

    # Pipeline for numerical variables
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Pipeline for categorical variables
    cat_transformer = OneHotEncoder(handle_unknown='ignore')

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features)
        ]
    )

    # Choose the classifier based on the algorithm
    if algo == 'LogisticRegression':
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(solver='lbfgs', max_iter=100000)
    elif algo == 'RandomForest':
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier()
    elif algo == 'SVM':
        from sklearn.svm import SVC
        classifier = SVC(probability=False)
    elif algo == 'KMeans':
        from sklearn.cluster import KMeans
        classifier = KMeans(n_clusters=2)
    elif algo == 'GaussianNB':
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB()
    elif algo == 'GradientBoosting':
        from sklearn.ensemble import GradientBoostingClassifier
        classifier = GradientBoostingClassifier()
    elif algo == 'AdaBoost':
        from sklearn.ensemble import AdaBoostClassifier
        classifier = AdaBoostClassifier()
    elif algo == 'MLP':
        from sklearn.neural_network import MLPClassifier
        classifier = MLPClassifier(max_iter=1000)
    elif algo == 'DecisionTree':
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier()
    elif algo == 'KNeighbors':
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier()
    elif algo == 'ExtraTrees':
        from sklearn.ensemble import ExtraTreesClassifier
        classifier = ExtraTreesClassifier()
    elif algo == 'Bagging':
        from sklearn.ensemble import BaggingClassifier
        classifier = BaggingClassifier()
    elif algo == 'Voting':
        from sklearn.ensemble import VotingClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        classifier = VotingClassifier(estimators=[
            ('rf', RandomForestClassifier()),
            ('svc', SVC(probability=True)),
        ], voting='soft')
    elif algo == 'Stacking':
        from sklearn.ensemble import StackingClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        classifier = StackingClassifier(estimators=[
            ('rf', RandomForestClassifier()),
            ('lr', LogisticRegression())
        ], final_estimator=LogisticRegression())
    elif algo == 'NearestCentroid':
        from sklearn.neighbors import NearestCentroid
        classifier = NearestCentroid()
    elif algo == 'QuadraticDiscriminantAnalysis':
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        classifier = QuadraticDiscriminantAnalysis()
    elif algo == 'GaussianProcess':
        from sklearn.gaussian_process import GaussianProcessClassifier
        classifier = GaussianProcessClassifier()
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    # Full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    return pipeline

def train_model_from_scratch(output_path: str = '/data/model.pkl') -> Pipeline:
    """
    Train a model from scratch
    """
    # Load data
    data = load_model_data()

    # Train the model
    pipeline = create_and_train_model(data, evaluate=True)

    # Save the model
    joblib.dump(pipeline, output_path)

    return pipeline

def create_and_train_model(data: pd.DataFrame,
                           evaluate: bool = False,
                           algo: all_algorithms = 'MLP') -> Pipeline:
    """
    Create and train a model on the given data
    """
    if evaluate:
        test_size = 0.2
    else:
        test_size = 0.0
    
    # Split the data
    X_train, X_test, y_train, y_test = preprocess_data(df=data, test_size=test_size)

    # Train the model
    pipeline = create_pipeline(algo)
    pipeline = train_model(pipeline, X_train, y_train)

    if evaluate:
        evaluation_results = evaluate_model(pipeline, X_test, y_test)
        logger.info(f"Evaluation results for {algo}:")
        logger.info(f"F1 Score: {evaluation_results['f1_score']}\n")
        logger.info(f"Confusion Matrix:\n{evaluation_results['confusion_matrix']}\n")
        logger.info(f"ROC AUC: {evaluation_results['roc_auc']}\n")
        logger.info(f"Classification Report:\n{evaluation_results['classification_report']}\n")

    return pipeline

def train_model(
        pipeline: Pipeline,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame) -> Pipeline:
    """
    Train the pipeline
    """
    pipeline.fit(X_train, y_train)
    return pipeline

def preprocess_data(df: pd.DataFrame, test_size: float = 0.2) -> Tuple:
    """
    Split the dataframe into X (features) and y (target).

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        Tuple: Split data (X_train, X_test, y_train, y_test).
    """
    # Format data for the model
    df_model = create_pairwise_data(df)

    features = [f.name for f in Feature.get_all_features()]
    X = df_model[features]
    y = df_model['target']

    # X_odd: keep the odd rows
    # X_even: keep the even rows
    X_odd = X.iloc[::2, :]
    X_even = X.iloc[1::2, :]
    y_odd = y.iloc[::2]
    y_even = y.iloc[1::2]

    # Split the data
    if test_size > 0:
        X_odd_train, X_odd_test, X_even_train, X_even_test, y_odd_train, y_odd_test, y_even_train, y_even_test = train_test_split(X_odd, X_even, y_odd, y_even, test_size=test_size, stratify=y_odd, random_state=42)
        X_train = pd.concat([X_odd_train, X_even_train], axis=0)
        X_test = pd.concat([X_odd_test, X_even_test], axis=0)
        y_train = pd.concat([y_odd_train, y_even_train], axis=0)
        y_test = pd.concat([y_odd_test, y_even_test], axis=0)

        return X_train, X_test, y_train, y_test
    else:
        return X, pd.DataFrame(), y, pd.DataFrame()

def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
    """
    Evaluates the model
    """
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
    report = classification_report(y_test, y_pred)

    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "confusion_matrix": cm,
        "roc_auc": roc_auc,
        "classification_report": report
    }

def predict(
    pipeline: Pipeline,
    series: str,
    surface: str,
    court: str,
    p1_rank: int,
    p1_play_hand: PlayHand,
    p1_back_hand: int,
    p1_height: int,
    p1_weight: int,
    p1_year_of_birth: int,
    p1_pro_year: int,
    p2_rank: int,
    p2_play_hand: PlayHand,
    p2_back_hand: int,
    p2_height: int,
    p2_weight: int,
    p2_year_of_birth: int,
    p2_pro_year: int,
) -> Dict[str, Any]:
    # Built a DataFrame with the new match
    new_match = pd.DataFrame([{
        Feature.SERIES.name: series,
        Feature.SURFACE.name: surface,
        Feature.COURT.name: court,
        Feature.DIFF_RANKING.name: p1_rank - p2_rank,
        Feature.MEAN_RANKING.name: (p1_rank + p2_rank) / 2,
        Feature.DIFF_PLAY_HAND.name: p1_play_hand - p2_play_hand,
        Feature.DIFF_BACK_HAND.name: p1_back_hand - p2_back_hand,
        Feature.DIFF_HEIGHT.name: p1_height - p2_height,
        Feature.MEAN_HEIGHT.name: (p1_height + p2_height) / 2,
        Feature.DIFF_WEIGHT.name: p1_weight - p2_weight,
        Feature.MEAN_WEIGHT.name: (p1_weight + p2_weight) / 2,
        Feature.DIFF_AGE.name: p1_year_of_birth - p2_year_of_birth,
        Feature.DIFF_NB_PRO_YEARS.name: p1_pro_year - p2_pro_year
    }])

    # Use the pipeline to make a prediction
    prediction = pipeline.predict(new_match)[0]
    proba = pipeline.predict_proba(new_match)[0]

    # Print the result
    logger.info("\n--- 📊 Result ---")
    logger.info(f"🏆 Win probability : {proba[1]:.2f}")
    logger.info(f"❌ Lose probability : {proba[0]:.2f}")
    logger.info(f"🎾 Prediction : {'Victory' if prediction == 1 else 'Loss'}")

    return {"result": prediction.item(), "prob": [p.item() for p in proba]}

def get_training_dataset(model_name: str, alias: str = "prod") -> Optional[pd.DataFrame]:
    """
    Get the dataset path in the MLflow registry.
    """
    if not model_name:
        raise ValueError("Model name is required.")
    
    client = get_mlflow_client()

    try:
        model_version = client.get_model_version_by_alias(
            name=model_name,
            alias=alias
        )

        # Download the dataset
        data_file = mlflow.artifacts.download_artifacts(run_id=model_version.run_id, artifact_path="datasets/data.csv")

        return pd.read_csv(data_file)
    except Exception as e:
        logger.exception(f"Model {model_name} not found: {e}")
    
    logger.info(f"No artifacts found for model {model_name}.")
    
    return None

def run_experiment(
        artifact_path: str = None,
        algo: all_algorithms = 'LogisticRegression',
        registered_model_name: Optional[str] = None,
        experiment_name: str = 'Tennis Prediction',
        ):
    """
    Run the entire ML experiment pipeline.

    Args:
        artifact_path (str): Path to store the model artifact.
        algo (str): Algorithm to use for training.
        registered_model_name (str): Name to register the model under in MLflow.
        experiment_name (str): Name of the MLflow experiment.
    """
    if not artifact_path:
        artifact_path = 'model'
    
    if not registered_model_name:
        registered_model_name = algo
    
    client = get_mlflow_client()

    # Set experiment's info -> ensures creation of the experiment if it doesn't exist
    mlflow.set_experiment(experiment_name)

    # Get our experiment info
    experiment = client.get_experiment_by_name(experiment_name)

    # Start timing
    start_time = time.time()

    # Create pipeline
    pipe = create_pipeline(algo=algo)
    
    logger.warning("Load data from database.")
    df = load_model_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Call mlflow autolog
    mlflow.sklearn.autolog(log_models=True, log_input_examples=False, log_model_signatures=True)

    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
        # Train model
        train_model(pipe, X_train, y_train)

        evaluation_results = evaluate_model(pipe, X_test, y_test)
        logger.info(f"Evaluation results for {algo}:")
        logger.info(f"F1 Score: {evaluation_results['f1_score']}\n")
        logger.info(f"Confusion Matrix:\n{evaluation_results['confusion_matrix']}\n")
        logger.info(f"ROC AUC: {evaluation_results['roc_auc']}\n")
        logger.info(f"Classification Report:\n{evaluation_results['classification_report']}\n")

        model_info = mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name,
        )

        alias_name = "latest_model"
        client.set_registered_model_alias(
            name=registered_model_name,
            version=model_info.registered_model_version,
            alias=alias_name
        )
        logger.info(f"Alias '{alias_name}' set on model '{registered_model_name}' version {model_info.registered_model_version}")

        # Save datasets to CSV
        logger.info("Saving datasets to CSV...")

        df_filename = 'data.csv'
        df.to_csv(df_filename, index=False)
        client.log_artifact(run_id=run.info.run_id, local_path=df_filename, artifact_path="datasets")

    # Print timing
    logger.info(f"...Training Done! --- Total training time: {time.time() - start_time} seconds")

def list_registered_models(alias_filter: Optional[List[str]] = None) -> List[Dict]:
    client = get_mlflow_client()
    # Should be:
    #   results = client.search_registered_models()
    # but this is not working from inside the container
    # so we need to use the store client to get the registered models
    results = client._get_registry_client().store.search_registered_models()

    output = []
    for res in results:
        for mv in res.latest_versions:
            try:
                model_version = client.get_model_version(name=mv.name, version=mv.version)
                aliases = list(model_version.aliases or [])
            except Exception as e:
                logger.error(e)
                aliases = []
            
            if alias_filter:
                matched_aliases = list(set(aliases).intersection(alias_filter))
                if not matched_aliases:
                    continue
            else:
                matched_aliases = aliases

            output.append({
                "name": mv.name,
                "run_id": mv.run_id,
                "version": mv.version,
                "aliases": matched_aliases
            })

    return output

def deploy_model(
        model_name: str,
        model_version: str,
) -> None:
    """
    Deploy a model to production
    """
    client = get_mlflow_client()
    try:
        client.set_registered_model_alias(
            name=model_name,
            version=model_version,
            alias="prod"
        )
        logger.info(f"Model '{model_name}' version {model_version} deployed to production.")
    except MlflowException as e:
        logger.error(f"Failed to deploy model '{model_name}' version {model_version} to production: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while deploying model '{model_name}' version {model_version} to production: {e}")
        raise

def undeploy_model(model_name: str) -> None:
    """
    Undeploy a model from production
    """
    client = get_mlflow_client()

    # Will raise an error if the alias does not exist
    client.get_model_version_by_alias(
        name=model_name,
        alias="prod"
    )

    client.delete_registered_model_alias(
        name=model_name,
        alias="prod"
    )
    logger.info(f"Model {model_name} undeployed from production.")

def load_model(name: str, alias: Optional[str] = None) -> Pipeline:
    """
    Load a model from MLflow using an alias or default to the latest version.

    Args:
        name (str): Registered model name.
        alias (Optional[str]): Alias to identify the version (e.g., 'prod').

    Returns:
        Pipeline: The loaded sklearn pipeline.
    """
    client = get_mlflow_client()

    # Construct the model URI
    if alias:
        model_key = f"{name}@{alias}"
    else:
        # Get the latest version
        model_info = client.get_registered_model(name)
        latest_version = model_info.latest_versions[0].version
        model_key = f"{name}/{latest_version}"
    
    if model_key in models:
        return models[model_key]

    model_uri = f"models:/{model_key}"

    # Load the model
    pipeline = mlflow.sklearn.load_model(model_uri=model_uri)

    logger.info(f"Model '{model_key}' loaded from URI: {model_uri}")

    models[model_key] = pipeline  # Cache the loaded model

    return pipeline