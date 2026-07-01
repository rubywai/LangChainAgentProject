import os
import re
from pathlib import Path

from dotenv import load_dotenv


SECRET_PATTERNS = [
    (re.compile(r"(mongodb(?:\+srv)?://)([^:@/\s]+):([^@/\s]+)@"), r"\1***:***@"),
    (re.compile(r"\bsk-[A-Za-z0-9._-]+\b"), "***"),
    (re.compile(r"\btvly-[A-Za-z0-9._-]+\b"), "***"),
]


def load_project_env(script_file: str) -> Path:
    env_path = Path(script_file).resolve().parent / ".env"
    load_dotenv(dotenv_path=env_path)
    return env_path


def require_env(name: str, env_path: Path | None = None, example: str | None = None) -> str:
    value = os.getenv(name)
    if value:
        return value

    message = f"{name} is required but not set."
    if env_path is not None:
        message += f" Checked: {env_path}"
    if example:
        message += f" Example: {example}"
    raise EnvironmentError(message)


def sanitize_error_message(error: object) -> str:
    message = str(error)
    for pattern, replacement in SECRET_PATTERNS:
        message = pattern.sub(replacement, message)
    return message
