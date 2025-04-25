import pandas as pd
from sklearn.pipeline import Pipeline

from src.service.model import create_pairwise_data, create_pipeline

def test_create_pairwise_data(simple_match: pd.DataFrame, simple_match_pairwise_data: pd.DataFrame):
    result = create_pairwise_data(simple_match)

    assert set(result.columns) == set(simple_match_pairwise_data.columns), "Columns are different"
    assert simple_match_pairwise_data.equals(result), "Dataframes are different"

def test_create_pairwise_data_empty(simple_match_empty: pd.DataFrame):
    result = create_pairwise_data(simple_match_empty)

    assert result.empty, "Dataframe is not empty"

def test_create_pipeline():
    pipeline = create_pipeline()
    assert pipeline is not None, "Pipeline is None"
    assert isinstance(pipeline, Pipeline), "Pipeline is not a Pipeline"
    assert len(pipeline.named_steps) == 2, "Pipeline has wrong number of steps"
    assert 'preprocessor' in pipeline.named_steps, "Preprocessor is missing"
    assert 'classifier' in pipeline.named_steps, "Classifier is missing"
