import pandas as pd
from typing import Dict, Optional, List
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataSummaryPreset, DataDriftPreset
from evidently.ui.workspace import CloudWorkspace
from src.repository.model_data import load_model_data
from src.service.model import get_training_dataset

import logging

logger = logging.getLogger(__name__)

class DataChecker:
    schemas: Dict[str, DataDefinition] = {
        "raw": DataDefinition(
            id_column="match_id",
            datetime_columns=["date"],
            numerical_columns=["winner_rank", "loser_rank", "winner_points", "loser_points", "w_height_cm", "w_weight_kg",
                            "w_year_of_birth", "w_pro_year", "l_height_cm", "l_weight_kg", "l_year_of_birth", "l_pro_year"],
            categorical_columns=["tournament_name", "tournament_series", "tournament_surface", "tournament_court",
                            "tournament_location", "winner_name", "w_first_name", "w_last_name", "w_play_hand",
                            "w_back_hand", "loser_name", "l_first_name", "l_last_name", "l_play_hand", "l_back_hand"],
        ),
        "cleaned": DataDefinition(
            numerical_columns=["diff_rank", "mean_rank",
                               "diff_height_cm", "mean_height_cm",
                               "diff_weight_kg", "mean_weight_kg",
                               "diff_nb_pro_years", "diff_age",
                               "diff_play_hand", "diff_back_hand"],
            categorical_columns=["tournament_series", "tournament_surface", "tournament_court",],
        )
    }

    def __init__(self, api_key: str, project_id: str):
        self._api_key = api_key
        self._project_id = project_id
        
        self.workspace = CloudWorkspace(
            token=api_key,
            url="https://app.evidently.cloud"
        )

        if not self.workspace:
            raise ValueError("CloudWorkspace not found. Please check your API key.")
        
        self.project = self.workspace.get_project(project_id=project_id)

        if not self.project:
            raise ValueError("Project not found. Please check your project ID.")

    def check_data(self, df: pd.DataFrame, ref_df: pd.DataFrame, tags: Optional[List[str]] = None) -> str:
        eval_data = Dataset.from_pandas(
            data=df,
            data_definition=self.schemas['cleaned']
        )
        ref_data = Dataset.from_pandas(
            data=ref_df,
            data_definition=self.schemas['cleaned']
        )

        report = Report(
            [
                DataSummaryPreset(),
                DataDriftPreset(),
            ],
            include_tests=True,
        )
        
        # Run the report
        logger.info("Running the report...")
        my_eval = report.run(current_data=eval_data, reference_data=ref_data, tags=tags)
        
        # Save the evaluation to the workspace
        logger.info("Saving the evaluation to the workspace...")
        snapshot_id = self.workspace.add_run(self.project.id, my_eval, include_data=False)

        logger.info(f"Evaluation saved with snapshot ID: {snapshot_id}")

        return snapshot_id

def check_model_data(
        model_name: str,
        checker: DataChecker,
) -> str:
    """
    Check the model data using Evidently.
    """
    # Get the newest data from the database
    df = load_model_data()
    
    # Get the training dataset
    ref_df = get_training_dataset(model_name=model_name)
    
    # Check the data
    return checker.check_data(df, ref_df, tags=[model_name])