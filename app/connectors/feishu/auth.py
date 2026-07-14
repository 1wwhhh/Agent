from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from app.connectors.feishu.exceptions import FeishuApiError, FeishuAuthError, FeishuConfigError

TENANT_ACCESS_TOKEN_URL = "/auth/v3/tenant_access_token/internal"


class _TenantTokenResponse(BaseModel):
    model_config = ConfigDict(extra="allow", validate_assignment=True, str_strip_whitespace=True)

    code: int = Field(default=0)
    msg: str | None = Field(default=None)
    tenant_access_token: str | None = Field(default=None)
    expire: int | str | None = Field(default=None)


class FeishuAuthClient:
    def __init__(
        self,
        *,
        app_id: str | None = None,
        app_secret: str | None = None,
        base_url: str = "https://open.feishu.cn/open-apis",
        timeout: float = 20.0,
        refresh_margin_seconds: int = 60,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.app_id = self._resolve_env("FEISHU_APP_ID", app_id)
        self.app_secret = self._resolve_env("FEISHU_APP_SECRET", app_secret)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.refresh_margin_seconds = max(0, int(refresh_margin_seconds))
        self._http_client = http_client
        self._owns_client = http_client is None
        self._lock = asyncio.Lock()
        self._tenant_access_token: str | None = None
        self._expires_at_monotonic: float = 0.0
        self._closed = False

    @staticmethod
    def _resolve_env(name: str, override: str | None) -> str:
        value = override if override is not None else os.getenv(name)
        if value is None or not str(value).strip():
            raise FeishuConfigError(f"{name} is required")
        return str(value).strip()

    async def __aenter__(self) -> "FeishuAuthClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_client and self._http_client is not None:
            await self._http_client.aclose()

    async def get_tenant_access_token(self, *, force_refresh: bool = False) -> str:
        if not force_refresh and self._tenant_access_token and time.monotonic() < self._expires_at_monotonic:
            return self._tenant_access_token

        async with self._lock:
            if not force_refresh and self._tenant_access_token and time.monotonic() < self._expires_at_monotonic:
                return self._tenant_access_token

            token_payload = await self._fetch_tenant_access_token()
            token = token_payload.tenant_access_token
            if not token:
                raise FeishuAuthError("tenant_access_token is missing in Feishu auth response")

            expire_value = token_payload.expire
            try:
                expire_seconds = int(expire_value) if expire_value is not None else 0
            except (TypeError, ValueError) as exc:
                raise FeishuAuthError(f"invalid expire value in Feishu auth response: {expire_value!r}") from exc

            lifetime_seconds = max(0, expire_seconds - self.refresh_margin_seconds)
            self._tenant_access_token = token
            self._expires_at_monotonic = time.monotonic() + lifetime_seconds
            return token

    async def _fetch_tenant_access_token(self) -> _TenantTokenResponse:
        url = f"{self.base_url}{TENANT_ACCESS_TOKEN_URL}"
        payload = {"app_id": self.app_id, "app_secret": self.app_secret}

        try:
            client = await self._get_http_client()
            response = await client.post(url, json=payload)
        except httpx.TimeoutException as exc:
            raise FeishuAuthError(f"tenant_access_token request timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise FeishuAuthError(f"tenant_access_token request failed: {exc}") from exc

        response_data = self._safe_json(response)
        token_payload = _TenantTokenResponse.model_validate(response_data)
        if response.status_code >= 400 or token_payload.code != 0:
            raise FeishuApiError(
                token_payload.msg or "failed to obtain tenant_access_token",
                path=TENANT_ACCESS_TOKEN_URL,
                status_code=response.status_code,
                code=token_payload.code,
                response_body=response_data,
            )
        return token_payload

    async def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=self.timeout)
        return self._http_client

    def _safe_json(self, response: httpx.Response) -> dict[str, Any]:
        try:
            data = response.json()
        except ValueError as exc:
            raise FeishuAuthError(f"tenant_access_token response is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise FeishuAuthError("tenant_access_token response must be a JSON object")
        return data
