"""keep model corrections separate from the transcription

The LLM no longer writes `corrected_text`: that column is the user's ground
truth and must stay free of machine output, or every fine-tune would learn from
unreviewed text. Model proposals now live in `suggested_text` / `suggested_by`
and only move into `corrected_text` when the user accepts them.

Lines written by the old behaviour (`corrected_by = 'llm'`) are therefore moved
back: the transcription becomes the raw prediction again and the model text
becomes what it always was — a suggestion.

Revision ID: c3d4e5f6a7b8
Revises: f2b3c4d5e6a7
Create Date: 2026-09-23 17:10:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c3d4e5f6a7b8"
down_revision: Union[str, Sequence[str], None] = "f2b3c4d5e6a7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("htr_lines", sa.Column("suggested_text", sa.Text(), nullable=True))
    op.add_column("htr_lines", sa.Column("suggested_by", sa.String(length=32), nullable=True))

    # untangle the old behaviour: automatic text becomes a reviewable proposal
    op.execute(
        """
        UPDATE htr_lines
           SET suggested_text = corrected_text,
               suggested_by = corrected_by,
               corrected_text = NULL,
               corrected_by = NULL
         WHERE corrected_by = 'llm'
        """
    )


def downgrade() -> None:
    op.execute(
        """
        UPDATE htr_lines
           SET corrected_text = suggested_text,
               corrected_by = 'llm',
               suggested_text = NULL,
               suggested_by = NULL
         WHERE suggested_by IS NOT NULL
        """
    )
    op.drop_column("htr_lines", "suggested_by")
    op.drop_column("htr_lines", "suggested_text")
