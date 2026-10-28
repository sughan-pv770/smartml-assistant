"""
smartml.core.loaders
======================
Smart data ingestion from a variety of file formats and SQL databases.

Supported formats (detected automatically from the file extension):
    * CSV / TSV (``.csv``, ``.tsv``)
    * Excel (``.xlsx``, ``.xls``)
    * JSON (``.json``)
    * Parquet (``.parquet``)
    * Feather (``.feather``, ``.ftr``)
    * ORC (``.orc``) — requires ``pyarrow``
    * SQLite / SQL (via :func:`from_sql`)

Examples::

    import smartml as sml

    df = sml.load("data/titanic.csv")
    df = sml.load("sales.parquet")
    df = sml.load("report.xlsx", sheet_name="Sheet2")
    df = sml.from_sql("SELECT * FROM users LIMIT 1000",
                       connection="sqlite:///mydb.sqlite3")
"""

from __future__ import annotations

import os
from typing import Any, Optional

import pandas as pd


# --------------------------------------------------------------------------- #
# Supported extension → loader mapping
# --------------------------------------------------------------------------- #

_LOADERS: dict[str, Any] = {
    ".csv": pd.read_csv,
    ".tsv": lambda p, **kw: pd.read_csv(p, sep="\t", **kw),
    ".xlsx": pd.read_excel,
    ".xls": pd.read_excel,
    ".json": pd.read_json,
    ".parquet": pd.read_parquet,
    ".feather": pd.read_feather,
    ".ftr": pd.read_feather,
    ".orc": pd.read_orc,
}


def load(path: str, **kwargs: Any) -> pd.DataFrame:
    """Load a dataset from a file into a pandas DataFrame.

    The file format is inferred from the extension. Any extra keyword
    arguments are forwarded directly to the underlying pandas reader.

    Args:
        path: Path to the data file.
        **kwargs: Extra arguments forwarded to the pandas reader
            (e.g. ``sheet_name`` for Excel, ``sep`` for CSV).

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is not recognised.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: '{path}'")

    ext = os.path.splitext(path)[-1].lower()
    reader = _LOADERS.get(ext)
    if reader is None:
        supported = ", ".join(sorted(_LOADERS))
        raise ValueError(
            f"Unsupported file format '{ext}'. "
            f"Supported formats: {supported}"
        )

    try:
        df = reader(path, **kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load '{path}' as {ext}: {exc}"
        ) from exc

    if not isinstance(df, pd.DataFrame):
        raise RuntimeError(
            f"Expected a DataFrame from loading '{path}', got {type(df)}."
        )

    return df


def from_sql(
    query: str,
    connection: Any,
    **kwargs: Any,
) -> pd.DataFrame:
    """Load data from a SQL database via a SELECT query.

    Args:
        query: A SQL SELECT statement.
        connection: A SQLAlchemy engine / connection string, or any
            DBAPI2-compatible connection object accepted by
            ``pandas.read_sql``.
        **kwargs: Extra arguments forwarded to ``pandas.read_sql``.

    Returns:
        Query result as a DataFrame.

    Raises:
        RuntimeError: If the query fails.

    Examples::

        df = from_sql(
            "SELECT age, salary FROM employees WHERE active = 1",
            connection="sqlite:///hr.sqlite3",
        )
    """
    # Security guard: Ensure query is strictly a read operation (SELECT or WITH)
    trimmed = query.strip()
    first_token = trimmed.split()[0].upper() if trimmed else ""
    if first_token not in ("SELECT", "WITH"):
        raise PermissionError(
            f"Security Violation: from_sql only permits read queries (SELECT / WITH). "
            f"Received disallowed command starting with '{first_token}'."
        )

    try:
        # Accept both connection strings (SQLAlchemy) and raw connections
        if isinstance(connection, str):
            try:
                from sqlalchemy import create_engine  # type: ignore
                engine = create_engine(connection)
                df = pd.read_sql(query, con=engine, **kwargs)
            except ImportError:
                raise RuntimeError(
                    "SQLAlchemy is required for connection-string SQL loading. "
                    "Install it with: pip install sqlalchemy"
                )
        else:
            df = pd.read_sql(query, con=connection, **kwargs)
    except Exception as exc:
        raise RuntimeError(f"SQL query failed: {exc}") from exc

    return df


def sniff_format(path: str) -> str:
    """Return the detected format name for a given file path.

    Args:
        path: File path to inspect.

    Returns:
        A short format name string (e.g. ``"CSV"``, ``"Parquet"``).

    Raises:
        ValueError: If the extension is not recognised.
    """
    ext = os.path.splitext(path)[-1].lower()
    if ext not in _LOADERS:
        raise ValueError(f"Unrecognised file extension: '{ext}'")
    names = {
        ".csv": "CSV", ".tsv": "TSV",
        ".xlsx": "Excel", ".xls": "Excel (legacy)",
        ".json": "JSON",
        ".parquet": "Parquet", ".feather": "Feather", ".ftr": "Feather",
        ".orc": "ORC",
    }
    return names.get(ext, ext.upper().lstrip("."))
