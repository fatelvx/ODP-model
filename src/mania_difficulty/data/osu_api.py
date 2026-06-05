from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import requests


OSU_BASE_URL = "https://osu.ppy.sh/api/v2"
OSU_TOKEN_URL = "https://osu.ppy.sh/oauth/token"


@dataclass
class OsuApiClient:
    access_token: str
    base_url: str = OSU_BASE_URL
    sleep_seconds: float = 0.0

    @classmethod
    def from_client_credentials(
        cls,
        client_id: str | None = None,
        client_secret: str | None = None,
        *,
        sleep_seconds: float = 0.0,
    ) -> "OsuApiClient":
        client_id = client_id or os.getenv("OSU_CLIENT_ID")
        client_secret = client_secret or os.getenv("OSU_CLIENT_SECRET")
        if not client_id or not client_secret:
            raise RuntimeError(
                "Missing osu! API credentials. Set OSU_CLIENT_ID and OSU_CLIENT_SECRET."
            )

        response = requests.post(
            OSU_TOKEN_URL,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "grant_type": "client_credentials",
                "scope": "public",
            },
            headers={"Accept": "application/json"},
            timeout=30,
        )
        response.raise_for_status()
        token = response.json()["access_token"]
        return cls(access_token=token, sleep_seconds=sleep_seconds)

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if self.sleep_seconds:
            time.sleep(self.sleep_seconds)

        response = requests.get(
            f"{self.base_url}/{path.lstrip('/')}",
            params=params,
            headers={
                "Authorization": f"Bearer {self.access_token}",
                "Accept": "application/json",
            },
            timeout=45,
        )
        response.raise_for_status()
        return response.json()
