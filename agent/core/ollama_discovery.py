"""Discovery helpers for a locally-running Ollama daemon.

Used by the ``/model ollama`` picker to enumerate installed models without
making the user type exact tags. Everything here is best-effort: if the
daemon isn't running or the response shape changes, callers get an empty
list (or ``None`` on metadata) and the picker shows a clear error.

No API key. ``OLLAMA_HOST`` is respected (matches the Ollama client itself).

Tool-call capability note: this file does NOT filter the list by
capability — that's a user decision per the "warn and allow" policy. The
picker annotates risky models in its own rendering.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

from agent.core.llm_params import _ollama_native_base

logger = logging.getLogger(__name__)

# Short timeout because this runs synchronously on the REPL thread when
# the user types ``/model ollama`` — we don't want the CLI to hang if the
# daemon is down or unreachable. 3s is plenty for a local request; raise
# if your Ollama host is remote and slow.
_LIST_TIMEOUT = 3.0


@dataclass
class OllamaModel:
    """One installed-locally Ollama model, as returned by /api/tags.

    ``name`` is the tag the user installed it under (e.g. ``qwen2.5:14b``
    or ``llama3.2:latest``). This is what goes into the litellm id as
    ``ollama_chat/<name>``.

    ``context_length`` and ``parameter_size`` come from the /api/show
    endpoint when we have time to fetch them; /api/tags alone only gives
    us name + size-on-disk + family.
    """
    name: str
    size_bytes: int
    family: Optional[str] = None
    parameter_size: Optional[str] = None  # e.g. "7B", "14B", "70B"
    quantization: Optional[str] = None    # e.g. "Q4_K_M"


class OllamaUnavailable(RuntimeError):
    """The local Ollama daemon couldn't be reached.

    Raised by ``list_local_models`` on connection refused / DNS failure /
    timeout so the picker can render a clear 'Ollama isn't running'
    message instead of an empty list.
    """


def list_local_models(base_url: Optional[str] = None) -> list[OllamaModel]:
    """Return installed-locally models by hitting ``/api/tags``.

    ``base_url`` defaults to OLLAMA_HOST / localhost:11434. Raises
    ``OllamaUnavailable`` if the daemon isn't reachable; returns ``[]``
    (never None) if it is but has no models.
    """
    base = (base_url or _ollama_native_base()).rstrip("/")
    url = f"{base}/api/tags"

    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=_LIST_TIMEOUT) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise OllamaUnavailable(
            f"Couldn't reach Ollama at {base}: {e}. "
            "Is the daemon running? Try `ollama serve` or `brew services start ollama`."
        ) from e
    except (json.JSONDecodeError, ValueError) as e:
        # Daemon responded but not with JSON we can parse. Treat as
        # unavailable so the picker shows a clear error instead of
        # a confusing empty list.
        raise OllamaUnavailable(
            f"Ollama at {base} returned an unparseable response: {e}"
        ) from e

    raw_models = payload.get("models") or []
    out: list[OllamaModel] = []
    for m in raw_models:
        # /api/tags shape (as of ollama 0.3+):
        #   { name, size, digest, modified_at,
        #     details: { family, parameter_size, quantization_level, ... } }
        details = m.get("details") or {}
        try:
            size = int(m.get("size", 0))
        except (TypeError, ValueError):
            size = 0
        out.append(OllamaModel(
            name=str(m.get("name", "")).strip(),
            size_bytes=size,
            family=details.get("family"),
            parameter_size=details.get("parameter_size"),
            quantization=details.get("quantization_level"),
        ))
    # Filter out models with blank names just in case the response has
    # garbage entries. Sort by size descending — bigger models first,
    # since they're almost always the better choice for tool-calling.
    out = [m for m in out if m.name]
    out.sort(key=lambda m: m.size_bytes, reverse=True)
    return out


# A small list of known tool-call-capable model families. We use this
# only for UX annotation — NOT as a block-list. The README documents
# that anything smaller than ~8B typically fails at this agent's tool
# schemas. These are the families where tool calling is known to work
# reliably per Ollama's own docs (https://ollama.com/search?c=tools).
_GOOD_TOOLCALL_FAMILIES = {
    "llama",        # llama3.1+, llama3.2 tool-use variants
    "qwen2",        # qwen2 / qwen2.5 all sizes
    "qwen3",        # qwen3 / qwen3-thinking
    "mistral",      # mistral-nemo, mistral-small, mistral-large
    "mixtral",
    "command-r",
    "firefunction",
    "gpt-oss",      # OpenAI's open weights
    "granite",      # IBM Granite 3+
    "gemma",        # gemma2 / gemma3 (tool variants only)
    "glm",
}


def looks_toolcall_capable(model: OllamaModel) -> bool:
    """Heuristic: does this model's family usually do structured tool calls?

    Used to annotate the picker, not to block. A "False" means we print
    a warning stripe next to the model — user can still pick it.
    """
    fam = (model.family or "").lower()
    if not fam:
        # Fall back to name prefix. Ollama's tag naming is consistent
        # enough that this works for most installs.
        head = model.name.split(":", 1)[0].split("/", 1)[-1].lower()
        fam = head.split("-", 1)[0]
    return any(fam.startswith(g) for g in _GOOD_TOOLCALL_FAMILIES)


def looks_too_small_for_agent(model: OllamaModel) -> bool:
    """Rough cutoff for "this is going to struggle with a tool-heavy agent".

    The floor is about 7B-of-real-parameters. Anything below that is
    going to either hallucinate tool arguments or get stuck in planning
    loops. We warn; we don't block.
    """
    ps = (model.parameter_size or "").upper().rstrip("B").strip()
    # Parameter_size strings from Ollama look like "7B", "13B", "70B",
    # "8x7B" for mixtral, "0.5B", "1.5B", "3B".
    try:
        # Handle the mixture-of-experts case ("8x7B") by taking the
        # per-expert size, which approximates activated params.
        if "X" in ps:
            ps = ps.split("X", 1)[1]
        n = float(ps)
    except (TypeError, ValueError):
        return False
    return n < 7.0


# ---- Cloud side ----------------------------------------------------------
# Ollama Cloud's available models are a small curated list we can hardcode
# rather than hit a discovery endpoint for. Updating this is a 1-line
# change when they add something. Source: https://ollama.com/cloud
# (captured April 2026; re-check before releases).
_OLLAMA_CLOUD_MODELS = [
    ("gpt-oss:120b-cloud", "OpenAI GPT-OSS 120B — strong tool-calling"),
    ("gpt-oss:20b-cloud",  "OpenAI GPT-OSS 20B  — lighter, still good at tools"),
    ("qwen3-coder:480b-cloud", "Qwen3 Coder 480B — coding-specialized"),
    ("deepseek-v3.1:671b-cloud", "DeepSeek V3.1 671B — huge MoE"),
    ("kimi-k2:1t-cloud", "Kimi K2 1T — massive MoE, agentic-tuned"),
]


def cloud_model_suggestions() -> list[tuple[str, str]]:
    """Return [(model_id, human_description)] for the ollama_cloud picker.

    Not fetched at runtime — Ollama Cloud doesn't expose a public catalog
    endpoint. Kept small and maintained by hand.
    """
    return list(_OLLAMA_CLOUD_MODELS)
