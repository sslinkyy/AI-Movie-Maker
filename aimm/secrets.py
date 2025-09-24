"""API key storage helpers."""
from __future__ import annotations

import contextlib
from typing import Optional

import keyring

from .config import APP_NAME


def get_api_key(service: str) -> Optional[str]:
    with contextlib.suppress(Exception):
        return keyring.get_password(APP_NAME, service)
    return None


def set_api_key(service: str, value: str) -> None:
    keyring.set_password(APP_NAME, service, value)
