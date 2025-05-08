"""init_schema

Revision ID: 0e832792e8c4
Revises: 
Create Date: 2025-05-08 16:40:24.585635

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0e832792e8c4'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    sql = """
        CREATE SCHEMA IF NOT EXISTS ml
            AUTHORIZATION tennis_admin;
    """
    op.execute(sa.text(sql))
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    sql = "DROP SCHEMA IF EXISTS ml CASCADE;"

    op.execute(sa.text(sql))
    # ### end Alembic commands ###
