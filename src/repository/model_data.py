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
    
    # Clean the data
    data = _clean_data(data)

    return data

def _clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the data by removing unnecessary columns and rows with null values
    and filtering based on specific criteria.
    """
    # Drop unnecessary columns
    columns_to_drop = [
        'match_id', 'date',
        'tournament_name', 'tournament_location',
        'winner_name', 'loser_name',
        'min_winner', 'min_loser',
        'avg_winner', 'avg_loser',
        'max_winner', 'max_loser'
    ]
    data.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    # Remove rows where 'victory_type' is not 'Completed'
    data = data[data['victory_type'] == 'Completed']
    data.drop(columns='victory_type', inplace=True, errors='ignore')
    
    # Remove rows where 'tournament_series' is not in the specified list
    data = data[data['tournament_series'].isin(['Grand Slam', 'Masters 1000', 'Masters', 'Masters Cup', 'ATP500', 'ATP250'])]

    # Remove rows with null values
    data.dropna(inplace=True)

    # Reset the index
    data.reset_index(drop=True, inplace=True)

    return data