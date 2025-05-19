"""new_model_inputs

Revision ID: 5f6a51af75db
Revises: 2902890ae133
Create Date: 2025-05-15 15:13:46.768232

"""
from typing import Sequence, Union
from importlib import import_module
import os
import sys

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '5f6a51af75db'
down_revision: Union[str, None] = '2902890ae133'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

sys.path.append(os.path.join(os.path.dirname(__file__)))
previous_migration = import_module('2025_05_08_1642-2902890ae133_add_model_data_m_view')

def upgrade() -> None:
    """Upgrade schema."""
    previous_migration.downgrade()

    sql_view = """
        CREATE MATERIALIZED VIEW IF NOT EXISTS ml.model_data_m_view
        TABLESPACE pg_default
        AS
        WITH data_odds AS (
            SELECT 
                o.fk_match_id AS match_id,
                AVG(o.winner) AS avg_winner,
                AVG(o.loser) AS avg_loser,
                MIN(o.winner) AS min_winner,
                MIN(o.loser) AS min_loser,
                MAX(o.winner) AS max_winner,
                MAX(o.loser) AS max_loser
            FROM data.odds o
            WHERE o.bookmaker NOT LIKE 'Avg%'
            GROUP BY o.fk_match_id
        ),
        data AS (
            SELECT 
                m.id AS match_id,
                DATE_TRUNC('day', m.date) AS date,
                m.tournament_name,
                m.tournament_series,
                m.tournament_surface,
                m.tournament_court,

                -- Match details
                m.comment AS victory_type,
                m.winner_rank,
                m.loser_rank,
                m.winner_rank - m.loser_rank AS diff_rank,
                (m.winner_rank + m.loser_rank) / 2.0 AS mean_rank,

                -- Caracteristics of the winner
                winner.name AS winner_name,
                w_caracs.height_cm AS w_height_cm,
                w_caracs.weight_kg AS w_weight_kg,
                -- Age at the time of the match
                EXTRACT(YEAR FROM AGE(m.date::date, w_caracs.date_of_birth::date)) AS w_age,
                -- Experience at the time of the match
                EXTRACT(YEAR FROM m.date::date) - w_caracs.pro_year AS w_nb_pro_years,
                w_caracs.play_hand AS w_play_hand,
                w_caracs.back_hand AS w_back_hand,

                -- Caracteristics of the loser
                loser.name AS loser_name,
                l_caracs.height_cm AS l_height_cm,
                l_caracs.weight_kg AS l_weight_kg,
                -- Age at the time of the match
                EXTRACT(YEAR FROM AGE(m.date::date, l_caracs.date_of_birth::date)) AS l_age,
                -- Experience at the time of the match
                EXTRACT(YEAR FROM m.date::date) - l_caracs.pro_year AS l_nb_pro_years,
                l_caracs.play_hand AS l_play_hand,
                l_caracs.back_hand AS l_back_hand,

                -- Bookmaker odds
                d_o.min_winner,
                d_o.min_loser,
                d_o.avg_winner,
                d_o.avg_loser,
                d_o.max_winner,
                d_o.max_loser
            FROM data.match m
            JOIN data.player winner ON winner.id = m.fk_winner_id
            JOIN data.player loser ON loser.id = m.fk_loser_id
            LEFT JOIN data.caracteristics w_caracs ON w_caracs.fk_player_id = winner.id
            LEFT JOIN data.caracteristics l_caracs ON l_caracs.fk_player_id = loser.id
            LEFT JOIN data_odds d_o ON d_o.match_id = m.id
            WHERE
                m.tournament_series IN ('Grand Slam', 'Masters 1000', 'Masters', 'Masters Cup', 'ATP500', 'ATP250')
                AND m.comment = 'Completed'
        )

        -- Final selection of columns
        SELECT 
            match_id,
            date,
            victory_type,
            winner_name,
            loser_name,
            
            tournament_name,
            tournament_series,
            tournament_surface,
            tournament_court,

            diff_rank,
            mean_rank,

            w_height_cm - l_height_cm AS diff_height_cm,
            (w_height_cm + l_height_cm) / 2.0 AS mean_height_cm,

            w_weight_kg - l_weight_kg AS diff_weight_kg,
            (w_weight_kg + l_weight_kg) / 2.0 AS mean_weight_kg,

            w_nb_pro_years - l_nb_pro_years AS diff_nb_pro_years,
            w_age - l_age AS diff_age,

            CASE 
                WHEN w_play_hand = l_play_hand THEN 0
                WHEN w_play_hand = 'R' THEN 1
                ELSE -1 
            END AS diff_play_hand,

            w_back_hand - l_back_hand AS diff_back_hand,

            min_winner,
            min_loser,
            avg_winner,
            avg_loser,
            max_winner,
            max_loser

        FROM data
        WITH DATA;

        ALTER TABLE IF EXISTS ml.model_data_m_view
            OWNER TO tennis_admin;
    """
    sql_index = """
        CREATE UNIQUE INDEX idx_unique_model_data_m_view_match_id
            ON ml.model_data_m_view USING btree
            (match_id)
            TABLESPACE pg_default;
    """
    op.execute(sa.text(sql_view))
    op.execute(sa.text(sql_index))
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    sql_view = """
        DROP MATERIALIZED VIEW IF EXISTS ml.model_data_m_view;
    """
    sql_index = """
        DROP INDEX IF EXISTS ml.idx_unique_model_data_m_view_match_id;
    """

    op.execute(sa.text(sql_view))
    op.execute(sa.text(sql_index))

    previous_migration.upgrade()
    # ### end Alembic commands ###
