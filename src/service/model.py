import os
import time
import joblib
import logging
import pandas as pd
from dotenv import load_dotenv
from typing import Literal, Any, Optional, Tuple, Dict, List
import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
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
from src.enums import Feature

from src.repository.model_data import load_model_data

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Explicitly set the logger level to INFO

load_dotenv()

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
            Feature.DIFF_RANKING.name: row['winner_rank'] - row['loser_rank'],
            Feature.DIFF_POINTS.name: row['winner_points'] - row['loser_points'],
            Feature.P1_PLAY_HAND.name: row['w_play_hand'],
            Feature.P1_BACK_HAND.name: row['w_back_hand'],
            Feature.P2_PLAY_HAND.name: row['l_play_hand'],
            Feature.P2_BACK_HAND.name: row['l_back_hand'],
            Feature.DIFF_HEIGHT.name: row['w_height_cm'] - row['l_height_cm'],
            Feature.DIFF_WEIGHT.name: row['w_weight_kg'] - row['l_weight_kg'],
            Feature.DIFF_AGE.name: row['w_year_of_birth'] - row['l_year_of_birth'],
            Feature.DIFF_PRO_AGE.name: row['w_pro_year'] - row['l_pro_year'],
            'target': 1 # Player in first position won
        }

        # Record 2 : invert players
        record_2 = record_1.copy()
        record_2[Feature.DIFF_RANKING.name] *= -1
        record_2[Feature.DIFF_POINTS.name] *= -1
        record_2[Feature.DIFF_HEIGHT.name] *= -1
        record_2[Feature.DIFF_WEIGHT.name] *= -1
        record_2[Feature.DIFF_AGE.name] *= -1
        record_2[Feature.DIFF_PRO_AGE.name] *= -1
        record_2[Feature.P1_PLAY_HAND.name] = record_1[Feature.P2_PLAY_HAND.name]
        record_2[Feature.P1_BACK_HAND.name] = record_1[Feature.P2_BACK_HAND.name]
        record_2[Feature.P2_PLAY_HAND.name] = record_1[Feature.P1_PLAY_HAND.name]
        record_2[Feature.P2_BACK_HAND.name] = record_1[Feature.P1_BACK_HAND.name]
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
    cat_features = [f.name for f in Feature.get_features_by_type('category') if f not in [Feature.DIFF_POINTS]]
    num_features = [f.name for f in Feature.get_features_by_type('number') if f not in [Feature.DIFF_POINTS]]

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

    features = [f.name for f in Feature.get_all_features() if f not in [Feature.DIFF_POINTS]]
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
    p1_play_hand: str,
    p1_back_hand: int,
    p1_height: int,
    p1_weight: int,
    p1_year_of_birth: int,
    p1_pro_year: int,
    p2_rank: int,
    p2_play_hand: str,
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
        Feature.P1_PLAY_HAND.name: p1_play_hand,
        Feature.P1_BACK_HAND.name: p1_back_hand,
        Feature.P2_PLAY_HAND.name: p2_play_hand,
        Feature.P2_BACK_HAND.name: p2_back_hand,
        Feature.DIFF_HEIGHT.name: p1_height - p2_height,
        Feature.DIFF_WEIGHT.name: p1_weight - p2_weight,
        Feature.DIFF_AGE.name: p1_year_of_birth - p2_year_of_birth,
        Feature.DIFF_PRO_AGE.name: p1_pro_year - p2_pro_year
    }])

    # Use the pipeline to make a prediction
    prediction = pipeline.predict(new_match)[0]
    proba = pipeline.predict_proba(new_match)[0]

    # Print the result
    logging.info("\n--- 📊 Result ---")
    logging.info(f"🏆 Win probability : {proba[1]:.2f}")
    logging.info(f"❌ Lose probability : {proba[0]:.2f}")
    logging.info(f"🎾 Prediction : {'Victory' if prediction == 1 else 'Loss'}")

    return {"result": prediction.item(), "prob": [p.item() for p in proba]}

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
    
    # Set tracking URI to your mlflow application
    mlflow.set_tracking_uri(os.environ["MLFLOW_SERVER_URI"])

    # Start timing
    start_time = time.time()

    # Load and preprocess data
    df = load_model_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Create pipeline
    pipe = create_pipeline(algo=algo)

    # Set experiment's info 
    mlflow.set_experiment(experiment_name)

    # Get our experiment info
    experiment = mlflow.get_experiment_by_name(experiment_name)

    # Call mlflow autolog
    mlflow.sklearn.autolog(log_models=True, log_input_examples=False, log_model_signatures=False)

    with mlflow.start_run(experiment_id=experiment.experiment_id):
        # Train model
        train_model(pipe, X_train, y_train)

        evaluation_results = evaluate_model(pipe, X_test, y_test)
        logger.info(f"Evaluation results for {algo}:")
        logger.info(f"F1 Score: {evaluation_results['f1_score']}\n")
        logger.info(f"Confusion Matrix:\n{evaluation_results['confusion_matrix']}\n")
        logger.info(f"ROC AUC: {evaluation_results['roc_auc']}\n")
        logger.info(f"Classification Report:\n{evaluation_results['classification_report']}\n")
            
        signature = infer_signature(X_test, pipe.predict(X_test))

        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name,
            signature=signature
        )

    # Print timing
    logging.info(f"...Training Done! --- Total training time: {time.time() - start_time} seconds")

def list_registered_models() -> List[Dict]:
    """
    List all the registered models
    """
    # Set tracking URI to your Heroku application
    tracking_uri = os.environ.get("MLFLOW_SERVER_URI")
    if tracking_uri is None:
        raise ValueError("MLFLOW_SERVER_URI environment variable is not set.")

    client = MlflowClient(tracking_uri=tracking_uri)
    # Should be:
    #   results = client.search_registered_models()
    # but this is not working from inside the container
    # so we need to use the store client to get the registered models
    results = client._get_registry_client().store.search_registered_models()

    output = []
    for res in results:
        for mv in res.latest_versions:
            output.append({"name": mv.name, "run_id": mv.run_id, "version": mv.version})
    
    return output

def load_model(name: str, version: str = 'latest') -> Pipeline:
    """
    Load a model from MLflow
    """
    if name in models.keys():
        return models[name]
    
    mlflow.set_tracking_uri(os.environ["MLFLOW_SERVER_URI"])
    client = MlflowClient()

    model_info = client.get_registered_model(name)

    # Load the model
    pipeline = mlflow.sklearn.load_model(model_uri=model_info.latest_versions[0].source)

    logging.info(f'Model {name} loaded')

    models[name] = pipeline

    return pipeline