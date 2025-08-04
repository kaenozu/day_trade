"""add_enum_types"

Revision ID: ec374a75826d
Revises: 647bd8b0ed7f
Create Date: 2025-08-04 09:17:16.990060

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'ec374a75826d'
down_revision = '647bd8b0ed7f'
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table('alerts', schema=None) as batch_op:
        batch_op.alter_column('alert_type',
                   existing_type=sa.VARCHAR(length=20),
                   type_=sa.Enum('PRICE_ABOVE', 'PRICE_BELOW', 'CHANGE_PERCENT_UP', 'CHANGE_PERCENT_DOWN', 'VOLUME_SPIKE', 'RSI_OVERBOUGHT', 'RSI_OVERSOLD', 'MACD_SIGNAL', 'BOLLINGER_BREAKOUT', 'PATTERN_DETECTED', 'CUSTOM_CONDITION', name='alerttype'),
                   existing_nullable=False)

    with op.batch_alter_table('trades', schema=None) as batch_op:
        batch_op.alter_column('trade_type',
                   existing_type=sa.VARCHAR(length=10),
                   type_=sa.Enum('BUY', 'SELL', name='tradetype'),
                   existing_nullable=False)


def downgrade() -> None:
    with op.batch_alter_table('trades', schema=None) as batch_op:
        batch_op.alter_column('trade_type',
                   existing_type=sa.Enum('BUY', 'SELL', name='tradetype'),
                   type_=sa.VARCHAR(length=10),
                   existing_nullable=False)

    with op.batch_alter_table('alerts', schema=None) as batch_op:
        batch_op.alter_column('alert_type',
                   existing_type=sa.Enum('PRICE_ABOVE', 'PRICE_BELOW', 'CHANGE_PERCENT_UP', 'CHANGE_PERCENT_DOWN', 'VOLUME_SPIKE', 'RSI_OVERBOUGHT', 'RSI_OVERSOLD', 'MACD_SIGNAL', 'BOLLINGER_BREAKOUT', 'PATTERN_DETECTED', 'CUSTOM_CONDITION', name='alerttype'),
                   type_=sa.VARCHAR(length=20),
                   existing_nullable=False)
