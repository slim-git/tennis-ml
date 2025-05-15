from typing import Optional
from src.repository.common import get_connection
import pandas as pd

def load_model_data(limit: Optional[int] = None) -> pd.DataFrame:
    """
    Load model data from Postgres
    """
    with get_connection() as conn:
        query = "SELECT * FROM ml.model_data_m_view"

        if limit is not None:
            query += f" LIMIT {limit}"
        
        data = pd.read_sql_query(query, conn)

    # Remove rows with null values
    data.dropna(inplace=True)

    # Remove rows where 'victory_type' is not 'Completed'
    data = data[data['victory_type'] == 'Completed']
    
    # Remove rows where 'tournament_series' is not in the specified list
    data = data[data['tournament_series'].isin(['Grand Slam', 'Masters 1000', 'Masters', 'Masters Cup', 'ATP500', 'ATP250'])]

    return data