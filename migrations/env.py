"""Alembic environment. Configured programmatically by ``probity.persistence.db.run_migrations``.

There is intentionally no ``alembic.ini``; callers pass ``script_location`` and either an open
connection (``config.attributes["connection"]``) or ``sqlalchemy.url``.
"""

from __future__ import annotations

from alembic import context

from probity.persistence.db import create_engine_for
from probity.persistence.schema import metadata

config = context.config


def _run(connection: object, *, manage_transaction: bool) -> None:
    # When the caller already holds BEGIN IMMEDIATE (run_migrations), do not emit another
    # BEGIN: our connect listener would try BEGIN DEFERRED inside the write lock.
    context.configure(
        connection=connection,
        target_metadata=metadata,
        render_as_batch=True,
        compare_type=True,
        transaction_per_migration=False,
        transactional_ddl=False,
    )
    if manage_transaction:
        with context.begin_transaction():
            context.run_migrations()
        return
    context.run_migrations()


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=metadata,
        literal_binds=True,
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connection = config.attributes.get("connection")
    if connection is not None:
        _run(connection, manage_transaction=False)
        return
    url = config.get_main_option("sqlalchemy.url")
    if not url:
        raise RuntimeError("sqlalchemy.url or an open connection is required")
    engine = create_engine_for(url)
    try:
        with engine.connect() as conn:
            _run(conn, manage_transaction=True)
            if conn.in_transaction():
                conn.commit()
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
