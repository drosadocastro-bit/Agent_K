from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from typing import Any

from agent_k.openclaw_capture import OpenClawCapture
from agent_k.openclaw_models import OpenClawCaptureResult
from agent_k.profiles import CALIBRATED_DEFAULT, AgentProfile


class ManatuabonBridgeError(RuntimeError):
    """Raised when the Manatuabon bridge is unreachable or returns an error."""


class ManatuabonBridgeRunner:
    """Black-box evaluator for a running Manatuabon bridge.

    This runner does not instrument Manatuabon internals. It queries the
    existing HTTP bridge, resolves cited memory summaries, and captures a
    minimal OpenClaw trace so Agent K can score the returned answer.
    """

    provider = "manatuabon"
    profile: AgentProfile = CALIBRATED_DEFAULT

    def __init__(
        self,
        capture: OpenClawCapture,
        *,
        host: str = "http://127.0.0.1:7777",
        timeout: float = 120.0,
    ) -> None:
        self.capture = capture
        self.host = host.rstrip("/")
        self.timeout = timeout

    def run_prompt(self, prompt: str, *, session_id: str = "manatuabon-bridge-eval") -> OpenClawCaptureResult:
        query_payload = self._post_json(
            "/query",
            {"prompt": prompt},
        )
        memories_payload = self._get_json("/memories")
        source_ids = tuple(int(value) for value in query_payload.get("sources", ()))
        context_items = self._resolve_memory_summaries(memories_payload, source_ids)

        self.capture.before_agent_start(
            session_id,
            system_instructions=(
                "Answer the user's question using the memory context.",
                "Cite every factual claim with exact memory references in the form [Memory #ID].",
                "If the memory bank is insufficient, say that explicitly instead of guessing.",
            ),
            user_prompt=prompt,
            context_items=context_items,
            allowed_tools=(),
            agent_name="manatuabon-bridge",
            prompt_metadata={
                "bridge_url": self.host,
                "source_ids": list(source_ids),
                "mode": "bridge_query",
            },
        )

        return self.capture.agent_end(
            session_id,
            output=query_payload.get("answer", ""),
            confidence=query_payload.get("confidence"),
            output_metadata={
                "provider": self.provider,
                "bridge_url": self.host,
                "sources": list(source_ids),
                "confidence_details": query_payload.get("confidence_details", {}),
            },
            profile=self.profile,
        )

    def _get_json(self, path: str) -> dict[str, Any] | list[Any]:
        request = urllib.request.Request(
            f"{self.host}{path}",
            headers={"Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ManatuabonBridgeError(
                f"GET {self.host}{path} failed with HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise ManatuabonBridgeError(f"Could not reach Manatuabon at {self.host}{path}: {exc}") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise ManatuabonBridgeError(
                f"GET {self.host}{path} timed out after {self.timeout:.0f}s. "
                "If Manatuabon is running but the model is slow, retry with a larger --timeout."
            ) from exc

        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise ManatuabonBridgeError(f"Manatuabon returned non-JSON response from {self.host}{path}") from exc

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.host}{path}",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ManatuabonBridgeError(
                f"POST {self.host}{path} failed with HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise ManatuabonBridgeError(f"Could not reach Manatuabon at {self.host}{path}: {exc}") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise ManatuabonBridgeError(
                f"POST {self.host}{path} timed out after {self.timeout:.0f}s. "
                "If Manatuabon is running but the model is slow, retry with a larger --timeout."
            ) from exc

        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise ManatuabonBridgeError(f"Manatuabon returned non-JSON response from {self.host}{path}") from exc

    @staticmethod
    def _resolve_memory_summaries(
        memories_payload: dict[str, Any] | list[Any],
        source_ids: tuple[int, ...],
    ) -> tuple[str, ...]:
        if not isinstance(memories_payload, list) or not source_ids:
            return ()

        memory_by_id = {
            int(item.get("id")): item
            for item in memories_payload
            if isinstance(item, dict) and item.get("id") is not None
        }
        context_items: list[str] = []
        for source_id in source_ids:
            memory_item = memory_by_id.get(source_id)
            if not memory_item:
                continue
            summary = str(memory_item.get("summary") or "").strip()
            if summary:
                context_items.append(f"Memory #{source_id}: {summary}")
        return tuple(context_items)
