"""Vertex AI transport (gcloud-authenticated) mirroring the OpenRouter contract.

Uses the Vertex OpenAI-compatible chat-completions endpoint, so requests and
responses have the same shape the rest of the pipeline already parses. Not
wired into any generation/validation entry point: pass ``call_structured``
from this module as the ``caller`` argument (see ``structured_caller_for``)
when a run should use a Vertex-backed profile.

Auth comes from the gcloud CLI (``gcloud auth print-access-token``), not an
API-key environment variable; the profile's ``project`` is also sent as the
quota project, which user-credential calls to aiplatform require.
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any

from .io import MVPError, sha256_object
from .openrouter import (
    OpenRouterResponseError,
    _parse_content,
    provider_response_schema,
)


class VertexResponseError(OpenRouterResponseError):
    """A successful HTTP response whose structured content could not be parsed.

    Subclasses OpenRouterResponseError so existing except-clauses that record
    provenance keep working when this transport is injected as the caller.
    """


_TOKEN_CACHE: dict[str, Any] = {"token": None, "expires_at": 0.0}
_TOKEN_TTL_SECONDS = 45 * 60


def _access_token() -> str:
    now = time.monotonic()
    if _TOKEN_CACHE["token"] and now < _TOKEN_CACHE["expires_at"]:
        return str(_TOKEN_CACHE["token"])
    try:
        result = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except FileNotFoundError as exc:
        raise MVPError("gcloud CLI not found; Vertex profiles require it") from exc
    except subprocess.CalledProcessError as exc:
        raise MVPError(
            f"gcloud auth print-access-token failed: {exc.stderr.strip()[:500]}"
        ) from exc
    token = result.stdout.strip()
    if not token:
        raise MVPError("gcloud auth print-access-token returned an empty token")
    _TOKEN_CACHE["token"] = token
    _TOKEN_CACHE["expires_at"] = now + _TOKEN_TTL_SECONDS
    return token


def build_request_body(
    profile: dict[str, Any],
    *,
    system: str,
    user: str,
    response_schema: dict[str, Any],
    schema_name: str,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": profile["model"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": int(profile["max_output_tokens"]),
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": provider_response_schema(response_schema),
            },
        },
    }
    for parameter in ("temperature", "top_p"):
        if parameter in profile:
            body[parameter] = profile[parameter]
    if "reasoning_effort" in profile:
        body["reasoning_effort"] = str(profile["reasoning_effort"])
    return body


def call_structured(
    profile: dict[str, Any],
    *,
    system: str,
    user: str,
    response_schema: dict[str, Any],
    schema_name: str,
) -> dict[str, Any]:
    """Call one pinned Vertex model route and return parsed output plus provenance."""
    if profile.get("provider") != "vertex":
        raise MVPError(
            f"vertex.call_structured requires provider=vertex; got {profile.get('provider')!r}"
        )
    body = build_request_body(
        profile,
        system=system,
        user=user,
        response_schema=response_schema,
        schema_name=schema_name,
    )
    encoded = json.dumps(body).encode("utf-8")
    total_attempts = int(profile["transport_attempts"])
    last_error: Exception | None = None
    for transport_attempt in range(1, total_attempts + 1):
        headers = {
            "Authorization": f"Bearer {_access_token()}",
            "x-goog-user-project": str(profile["project"]),
            "Content-Type": "application/json",
        }
        request = urllib.request.Request(
            str(profile["base_url"]), data=encoded, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(
                request, timeout=float(profile["timeout_seconds"])
            ) as stream:
                raw = json.loads(stream.read().decode("utf-8"))
                response_headers = {
                    key.casefold(): value
                    for key, value in stream.headers.items()
                    if key.casefold() in {"x-request-id", "server-timing"}
                }
            if not isinstance(raw, dict):
                raise MVPError("Vertex response is not a JSON object")
            provenance = {
                "profile_id": profile["profile_id"],
                "requested_model": profile["model"],
                "canonical_slug": profile["canonical_slug"],
                "returned_model": raw.get("model"),
                "response_id": raw.get("id"),
                "provider": "vertex",
                "project": profile["project"],
                "location": profile["location"],
                "usage": raw.get("usage"),
                "response_headers": response_headers,
                "request_sha256": sha256_object(body),
                "transport_attempt": transport_attempt,
            }
            try:
                parsed = _parse_content(raw, role=str(profile["role"]))
            except MVPError as exc:
                raise VertexResponseError(
                    str(exc), raw_response=raw, provenance=provenance
                ) from exc
            return {
                "parsed": parsed,
                "raw_response": raw,
                "provenance": provenance,
            }
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            last_error = MVPError(f"Vertex HTTP {exc.code}: {details[:1000]}")
            if exc.code == 401:
                _TOKEN_CACHE["token"] = None
            retryable = exc.code in (401, 429) or 500 <= exc.code <= 599
            if not retryable:
                raise last_error from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
        if transport_attempt < total_attempts:
            time.sleep(min(2 ** (transport_attempt - 1), 4))
    raise MVPError(
        f"Vertex transport failed after {total_attempts} attempts: {last_error}"
    ) from last_error


def structured_caller_for(profile: dict[str, Any]):
    """Return the transport matching a profile's provider.

    Lets a run select the right caller from the profile alone, e.g.
    ``run_generation(..., caller=structured_caller_for(generator_profile))``.
    """
    from . import openrouter

    provider = profile.get("provider")
    if provider == "openrouter":
        return openrouter.call_structured
    if provider == "vertex":
        return call_structured
    raise MVPError(f"No transport for provider {provider!r}")
