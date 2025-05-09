import pandas as pd
from typing import Optional
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataSummaryPreset, DataDriftPreset
from evidently.ui.workspace import CloudWorkspace

class DataChecker:
    def __init__(self, api_key: str, project_id: str, ref_dataset_id: str):
        self._api_key = api_key
        self._project_id = project_id
        self._ref_dataset_id = ref_dataset_id
        
        self.workspace = CloudWorkspace(
            token=api_key,
            url="https://app.evidently.cloud"
        )

        if not self.workspace:
            raise ValueError("CloudWorkspace not found. Please check your API key.")
        
        self.project = self.workspace.get_project(project_id=project_id)

        if not self.project:
            raise ValueError("Project not found. Please check your project ID.")

        self.ref_dataset = self.workspace.load_dataset(ref_dataset_id)

        if not self.ref_dataset:
            raise ValueError("Reference dataset not found. Please check your reference dataset ID.")

    def check_data(self, df: pd.DataFrame, schema: Optional[DataDefinition] = None) -> str:
        if not schema:
            schema = DataDefinition(
                id_column="match_id",
                datetime_columns=["date"],
                numerical_columns=["winner_rank", "loser_rank", "winner_points", "loser_points", "w_height_cm", "w_weight_kg",
                                "w_year_of_birth", "w_pro_year", "l_height_cm", "l_weight_kg", "l_year_of_birth", "l_pro_year"],
                categorical_columns=["tournament_name", "tournament_series", "tournament_surface", "tournament_court",
                                "tournament_location", "winner_name", "w_first_name", "w_last_name", "w_play_hand",
                                "w_back_hand", "loser_name", "l_first_name", "l_last_name", "l_play_hand", "l_back_hand"],
            )

        eval_data = Dataset.from_pandas(
            data=df,
            data_definition=schema
        )

        report = Report(
            [
                DataSummaryPreset(),
                DataDriftPreset(),
            ],
            include_tests=True,
        )
        
        # Run the report
        my_eval = report.run(eval_data, self.ref_dataset)
        
        # Save the evaluation to the workspace
        return self.workspace.add_run(self.project.id, my_eval, include_data=False)
        