"""add model_data_m_view

Revision ID: 2902890ae133
Revises: 0e832792e8c4
Create Date: 2025-05-08 16:42:24.585635

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2902890ae133'
down_revision: Union[str, None] = '0e832792e8c4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    sql_view = """
        CREATE MATERIALIZED VIEW IF NOT EXISTS ml.model_data_m_view
        TABLESPACE pg_default
        AS
        WITH data_odds AS (
                SELECT o.fk_match_id AS match_id,
                    avg(o.winner) AS avg_winner,
                    avg(o.loser) AS avg_loser,
                    min(o.winner) AS min_winner,
                    min(o.loser) AS min_loser,
                    max(o.winner) AS max_winner,
                    max(o.loser) AS max_loser
                FROM data.odds o
                WHERE o.bookmaker::text !~~ 'Avg%'::text
                GROUP BY o.fk_match_id
                )
        SELECT m.id AS match_id,
            date_trunc('DAY'::text, m.date) AS date,
            m.comment,
            m.winner_rank,
            m.loser_rank,
            m.winner_points,
            m.loser_points,
            m.tournament_name,
            m.tournament_series,
            m.tournament_surface,
            m.tournament_court,
            m.tournament_location,
            winner.name AS winner_name,
            w_caracs.first_name AS w_first_name,
            w_caracs.last_name AS w_last_name,
            w_caracs.play_hand AS w_play_hand,
            w_caracs.back_hand AS w_back_hand,
            w_caracs.height_cm AS w_height_cm,
            w_caracs.weight_kg AS w_weight_kg,
            date_part('YEAR'::text, w_caracs.date_of_birth::date)::integer AS w_year_of_birth,
            w_caracs.pro_year AS w_pro_year,
            loser.name AS loser_name,
            l_caracs.first_name AS l_first_name,
            l_caracs.last_name AS l_last_name,
            l_caracs.play_hand AS l_play_hand,
            l_caracs.back_hand AS l_back_hand,
            l_caracs.height_cm AS l_height_cm,
            l_caracs.weight_kg AS l_weight_kg,
            date_part('YEAR'::text, l_caracs.date_of_birth::date)::integer AS l_year_of_birth,
            l_caracs.pro_year AS l_pro_year,
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
        ORDER BY m.id DESC
        WITH DATA;

        ALTER TABLE IF EXISTS ml.model_data_m_view
            OWNER TO tennis_admin;
    """

    sql_index = """
        CREATE INDEX idx_unique_model_data_m_view_match_id
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
    # ### end Alembic commands ###
