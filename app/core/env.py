"""Load the project's local ``.env`` file into the process environment.

``app.core.config.Settings`` reads ``.env`` for its own fields only; it does not export them.
The Anthropic configuration (``ANTHROPIC_API_KEY``, ``ANTHROPIC_MODEL``, ``LLM_TEMPERATURE``,
``LLM_MAX_TOKENS``) is read from ``os.environ`` by ``app.services.ai_service`` at import
time, so ``.env`` must be loaded before that module is imported. ``app.main`` calls
``load_local_env()`` first, which makes ``uvicorn app.main:app --reload`` work without
exporting variables in the shell.

Variables that are already set in the environment always take precedence over ``.env``.
Values are never printed or logged.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"


def load_local_env(env_file: str | Path | None = None) -> bool:
    """Load ``env_file`` (default: ``<project root>/.env``) without overriding existing variables.

    Returns ``True`` when a file was found and loaded, ``False`` otherwise.
    """
    from dotenv import load_dotenv

    path = Path(env_file) if env_file is not None else DEFAULT_ENV_FILE
    if not path.is_file():
        return False
    load_dotenv(dotenv_path=path, override=False)
    return True
