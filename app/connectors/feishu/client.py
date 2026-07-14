from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx

from app.connectors.feishu.auth import FeishuAuthClient
from app.connectors.feishu.exceptions import FeishuApiError, FeishuDownloadError, FeishuTransportError


class FeishuClient:
    def __init__(
        self,
        *,
        auth_client: FeishuAuthClient | None = None,
        base_url: str = "https://open.feishu.cn/open-apis",
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_backoff_seconds: float = 0.5,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.auth_client = auth_client or FeishuAuthClient(timeout=timeout)
        self._owns_auth_client = auth_client is None
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max(1, int(max_retries))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        self._http_client = http_client or httpx.AsyncClient(timeout=httpx.Timeout(timeout))
        self._owns_client = http_client is None
        self._closed = False

    async def __aenter__(self) -> "FeishuClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_client:
            await self._http_client.aclose()
        if self._owns_auth_client:
            await self.auth_client.aclose()

    async def get(self, path: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return await self._request_json("GET", path, params=params)

    async def post(self, path: str, *, json_body: dict[str, Any] | None = None) -> dict[str, Any]:
        return await self._request_json("POST", path, json_body=json_body)

    async def download_to_path(
        self,
        path: str,
        save_path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> None:
        destination = Path(save_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_path = destination.with_name(f"{destination.name}.part")
        if temp_path.exists():
            temp_path.unlink()

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            download_succeeded = False
            try:
                token = await self.auth_client.get_tenant_access_token(force_refresh=attempt > 1)
                headers = {"Authorization": f"Bearer {token}"}
                url = self._build_url(path)
                async with self._http_client.stream("GET", url, params=params, headers=headers) as response:
                    if response.status_code == 401 and attempt < self.max_retries:
                        last_error = FeishuDownloadError(
                            f"download unauthorized for {path}; refreshing token and retrying"
                        )
                        await response.aclose()
                        continue
                    if response.status_code in {429} or 500 <= response.status_code < 600:
                        body = await self._read_error_body(response)
                        last_error = FeishuTransportError(
                            f"download request failed with HTTP {response.status_code}: {body}"
                        )
                        if attempt < self.max_retries:
                            await self._sleep_backoff(attempt)
                            continue
                        raise last_error
                    if response.status_code >= 400:
                        body = await self._read_error_body(response)
                        raise FeishuDownloadError(
                            f"download request failed with HTTP {response.status_code}: {body}"
                        )
                    try:
                        with temp_path.open("wb") as handle:
                            async for chunk in response.aiter_bytes():
                                if chunk:
                                    handle.write(chunk)
                        temp_path.replace(destination)
                        download_succeeded = True
                        return
                    except Exception as exc:
                        last_error = FeishuDownloadError(f"failed to write download to '{save_path}': {exc}")
                        if attempt < self.max_retries:
                            await self._sleep_backoff(attempt)
                            continue
                        raise last_error from exc
            except httpx.TimeoutException as exc:
                last_error = FeishuTransportError(f"download timed out for {path}: {exc}")
            except httpx.HTTPError as exc:
                last_error = FeishuTransportError(f"download request failed for {path}: {exc}")
            finally:
                if not download_succeeded and temp_path.exists():
                    try:
                        temp_path.unlink()
                    except OSError:
                        pass

            if attempt < self.max_retries and last_error is not None:
                await self._sleep_backoff(attempt)
                continue

        if last_error is not None:
            raise last_error
        raise FeishuDownloadError(f"download failed for {path}")

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                token = await self.auth_client.get_tenant_access_token(force_refresh=attempt > 1)
                headers = {"Authorization": f"Bearer {token}"}
                response = await self._http_client.request(
                    method,
                    self._build_url(path),
                    params=params,
                    json=json_body,
                    headers=headers,
                )
                if response.status_code == 401 and attempt < self.max_retries:
                    last_error = FeishuApiError(
                        "unauthorized",
                        path=path,
                        status_code=response.status_code,
                        response_body=self._safe_response_preview(response),
                    )
                    await self._sleep_backoff(attempt)
                    continue
                if response.status_code in {429} or 500 <= response.status_code < 600:
                    last_error = FeishuTransportError(
                        f"HTTP {response.status_code} calling {path}: {self._safe_response_preview(response)}"
                    )
                    if attempt < self.max_retries:
                        await self._sleep_backoff(attempt)
                        continue
                    raise last_error
                if response.status_code >= 400:
                    data = self._safe_response_json(response, path)
                    raise FeishuApiError(
                        self._extract_message(data) or f"HTTP {response.status_code} calling {path}",
                        path=path,
                        status_code=response.status_code,
                        code=self._extract_code(data),
                        response_body=data,
                    )

                data = self._safe_response_json(response, path)
                code = self._extract_code(data)
                if code not in (None, 0):
                    raise FeishuApiError(
                        self._extract_message(data) or f"Feishu API error code {code}",
                        path=path,
                        status_code=response.status_code,
                        code=code,
                        response_body=data,
                    )

                return data
            except httpx.TimeoutException as exc:
                last_error = FeishuTransportError(f"timeout calling {path}: {exc}")
            except httpx.HTTPError as exc:
                last_error = FeishuTransportError(f"request failed calling {path}: {exc}")

            if attempt < self.max_retries and last_error is not None:
                await self._sleep_backoff(attempt)
                continue

        if last_error is not None:
            raise last_error
        raise FeishuTransportError(f"request failed calling {path}")

    async def _sleep_backoff(self, attempt: int) -> None:
        await asyncio.sleep(self.retry_backoff_seconds * attempt)

    def _build_url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def _safe_response_json(self, response: httpx.Response, path: str) -> dict[str, Any]:
        try:
            data = response.json()
        except ValueError as exc:
            raise FeishuApiError(
                f"response from {path} is not valid JSON",
                path=path,
                status_code=response.status_code,
            ) from exc
        if not isinstance(data, dict):
            raise FeishuApiError(
                f"response from {path} must be a JSON object",
                path=path,
                status_code=response.status_code,
                response_body=data,
            )
        return data

    def _extract_code(self, data: dict[str, Any]) -> int | None:
        value = data.get("code")
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _extract_message(self, data: dict[str, Any]) -> str | None:
        message = data.get("msg")
        if message is None:
            return None
        return str(message)

    def _describe_body(self, data: dict[str, Any]) -> str:
        try:
            return json.dumps(data, ensure_ascii=True)
        except Exception:
            return repr(data)

    def _safe_response_preview(self, response: httpx.Response) -> str:
        try:
            body = response.text
        except Exception:
            body = ""
        if body:
            return body[:500]
        return f"HTTP {response.status_code}"

    async def _read_error_body(self, response: httpx.Response) -> str:
        try:
            payload = await response.aread()
        except Exception:
            return ""
        if not payload:
            return ""
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            return f"{len(payload)} bytes"
        return text[:500]
