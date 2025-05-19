import pytest
import pandas as pd

# ----------------------------------------------------------------
# Fixtures for Match Data
# ----------------------------------------------------------------
@pytest.fixture
def simple_match():
    return pd.DataFrame({
        'tournament_series': ['Masters',],
        'tournament_surface': ['Hard',],
        'tournament_court': ['Outdoor',],
        'diff_rank': [-63],
        'mean_rank': [32.5],
        'diff_play_hand': [0],
        'diff_back_hand': [-1],
        'diff_height_cm': [2],
        'mean_height_cm': [184],
        'diff_weight_kg': [6],
        'mean_weight_kg': [83],
        'diff_nb_pro_years': [-3],
        'diff_year_of_birth': [-1],
    })

@pytest.fixture
def simple_match_pairwise_data(simple_match: pd.DataFrame):
    series = simple_match['tournament_series'].values[0]
    surface = simple_match['tournament_surface'].values[0]
    court = simple_match['tournament_court'].values[0]
    diff_ranking = simple_match['diff_rank'].values[0]
    mean_ranking = simple_match['mean_rank'].values[0]
    diff_play_hand = simple_match['diff_play_hand'].values[0]
    diff_back_hand = simple_match['diff_back_hand'].values[0]
    diff_height = simple_match['diff_height_cm'].values[0]
    mean_height = simple_match['mean_height_cm'].values[0]
    diff_weight = simple_match['diff_weight_kg'].values[0]
    mean_weight = simple_match['mean_weight_kg'].values[0]
    diff_nb_pro_years = simple_match['diff_nb_pro_years'].values[0]
    diff_age = simple_match['diff_year_of_birth'].values[0]

    return pd.DataFrame({
        'Series': [series, series],
        'Surface': [surface, surface],
        'Court': [court, court],

        'diffNbProYears': [diff_nb_pro_years, -diff_nb_pro_years],
        'diffAge': [diff_age, -diff_age],

        'diffPlayHand': [diff_play_hand, -diff_play_hand],
        'diffBackHand': [diff_back_hand, -diff_back_hand],

        'diffRanking': [diff_ranking, -diff_ranking],
        'diffHeight': [diff_height, -diff_height],
        'diffWeight': [diff_weight, -diff_weight],

        'meanRanking': [mean_ranking, mean_ranking],
        'meanHeight': [mean_height, mean_height],
        'meanWeight': [mean_weight, mean_weight],

        'target': [1, 0]
    })

@pytest.fixture
def simple_match_empty():
    return pd.DataFrame({
        'Series': [],
        'Surface': [],
        'Court': [],

        'diffNbProYears': [],
        'diffAge': [],

        'diffPlayHand': [],
        'diffBackHand': [],

        'diffRanking': [],
        'diffHeight': [],
        'diffWeight': [],

        'meanRanking': [],
        'meanHeight': [],
        'meanWeight': [],

        'target': []
    })
