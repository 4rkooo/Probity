"""SQLite engine construction, programmatic Alembic migrations, and transaction helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import Connection, Engine, create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.pool import StaticPool

from probity.config import REPO_ROOT

MIGRATIONS_DIR = REPO_ROOT / "migrations"
BUSY_TIMEOUT_MS = 5000

Clock = Callable[[], datetime]

_BEGIN_OPTION = "probity_sqlite_begin"


def utc_clock() -> datetime:
    return datetime.now(UTC)


def to_epoch_us(moment: datetime) -> int:
    if moment.tzinfo is None:
        raise ValueError("naive datetime is not allowed")
    delta = moment.astimezone(UTC) - datetime(1970, 1, 1, tzinfo=UTC)
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


def _is_memory(database: str | None) -> bool:
    return not database or database == ":memory:" or database.startswith("file::memory:")


def database_path(url: str) -> str:
    """Filesystem path of a SQLite URL (``:memory:`` for in-memory); never includes credentials."""
    database = make_url(url).database
    return ":memory:" if _is_memory(database) else str(Path(database or "").resolve())


def create_engine_for(url: str, *, echo: bool = False) -> Engine:
    """SQLite engine with WAL, ``synchronous=FULL``, foreign keys, and a 5 s busy timeout.

    The driver is put in autocommit mode so SQLAlchemy's ``begin`` event controls the exact
    ``BEGIN`` statement; ``write_transaction`` uses ``BEGIN IMMEDIATE`` to take the write lock
    up front, which is what makes lease claims atomic across processes.
    """
    parsed = make_url(url)
    if parsed.get_backend_name() != "sqlite":
        raise ValueError("Probity persistence supports SQLite only")
    kwargs: dict[str, Any] = {
        "echo": echo,
        "connect_args": {"check_same_thread": False, "timeout": BUSY_TIMEOUT_MS / 1000},
    }
    if _is_memory(parsed.database):
        kwargs["poolclass"] = StaticPool
    else:
        Path(parsed.database or "").expanduser().resolve().parent.mkdir(
            parents=True, exist_ok=True
        )
    engine = create_engine(url, **kwargs)
    memory = _is_memory(parsed.database)

    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_conn: Any, _record: Any) -> None:
        dbapi_conn.isolation_level = None
        cursor = dbapi_conn.cursor()
        try:
            if not memory:
                cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=FULL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        finally:
            cursor.close()

    @event.listens_for(engine, "begin")
    def _on_begin(conn: Connection) -> None:
        mode = conn.get_execution_options().get(_BEGIN_OPTION, "DEFERRED")
        conn.exec_driver_sql(f"BEGIN {mode}")

    return engine


@contextmanager
def write_transaction(engine: Engine) -> Iterator[Connection]:
    """``BEGIN IMMEDIATE`` transaction: commits on success, rolls back on any exception."""
    with engine.connect() as conn:
        conn = conn.execution_options(**{_BEGIN_OPTION: "IMMEDIATE"})
        with conn.begin():
            yield conn


@contextmanager
def read_transaction(engine: Engine) -> Iterator[Connection]:
    """Deferred (snapshot) transaction for consistent multi-statement reads."""
    with engine.connect() as conn, conn.begin():
        yield conn


def alembic_config(url: str) -> Any:
    from alembic.config import Config

    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return cfg


def run_migrations(url: str, revision: str = "head", *, engine: Engine | None = None) -> None:
    """Upgrade the database to ``revision`` (programmatic Alembic config; no alembic.ini).

    The upgrade runs inside one ``BEGIN IMMEDIATE`` transaction, so concurrent API/worker
    startups serialize and the second sees the already-applied version. Pass ``engine`` to
    keep an in-memory database alive after the upgrade (tests); production uses ``url`` only.
    """
    from alembic import command

    cfg = alembic_config(url)
    owns = engine is None
    engine = engine if engine is not None else create_engine_for(url)
    try:
        with write_transaction(engine) as conn:
            cfg.attributes["connection"] = conn
            command.upgrade(cfg, revision)
    finally:
        if owns:
            engine.dispose()
