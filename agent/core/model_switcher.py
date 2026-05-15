"""Model-switching logic for the interactive CLI's ``/model`` command.

Split out of ``agent.main`` so the REPL dispatcher stays focused on input
parsing. Exposes:

* ``SUGGESTED_MODELS`` — the short list shown by ``/model`` with no arg.
* ``is_valid_model_id`` — loose format check on user input.
* ``probe_and_switch_model`` — async: checks routing, fires a 1-token
  probe to resolve the effort cascade, then commits the switch (or
  rejects it on hard error).

The probe's cascade lives in ``agent.core.effort_probe``; this module
glues it to CLI output + session state.
"""

from __future__ import annotations

import asyncio
import os

from litellm import acompletion

from agent.core.effort_probe import ProbeInconclusive, probe_effort
from agent.core.llm_params import _resolve_llm_params
from agent.core.local_models import (
    LOCAL_MODEL_PREFIXES,
    is_local_model_id,
    is_reserved_local_model_id,
)


# Suggested models shown by `/model` (not a gate). Users can paste any HF
# model id (e.g. "MiniMaxAI/MiniMax-M2.7") or an `anthropic/` / `openai/`
# prefix for direct API access. For HF ids, append ":fastest" /
# ":cheapest" / ":preferred" / ":<provider>" to override the default
# routing policy (auto = fastest with failover).
SUGGESTED_MODELS = [
    {"id": "openai/gpt-5.5", "label": "GPT-5.5"},
    {"id": "openai/gpt-5.4", "label": "GPT-5.4"},
    {"id": "anthropic/claude-opus-4-7", "label": "Claude Opus 4.7"},
    {"id": "anthropic/claude-opus-4-6", "label": "Claude Opus 4.6"},
    {
        "id": "bedrock/us.anthropic.claude-opus-4-6-v1",
        "label": "Claude Opus 4.6 via Bedrock",
    },
    {"id": "gemini/gemini-2.5-pro", "label": "Gemini 2.5 Pro"},
    {"id": "gemini/gemini-2.5-flash", "label": "Gemini 2.5 Flash"},
    {"id": "ollama_cloud/gpt-oss:120b-cloud", "label": "GPT-OSS 120B (Ollama Cloud)"},
    {"id": "ollama", "label": "Local Ollama — opens picker"},
    {"id": "MiniMaxAI/MiniMax-M2.7", "label": "MiniMax M2.7"},
    {"id": "moonshotai/Kimi-K2.6", "label": "Kimi K2.6"},
    {"id": "zai-org/GLM-5.1", "label": "GLM 5.1"},
    {"id": "deepseek-ai/DeepSeek-V4-Pro:deepinfra", "label": "DeepSeek V4 Pro"},
]

# Providers that silently ignore reasoning_effort on the wire. We skip
# the effort probe entirely for these so we don't cache a bogus
# "effective" level against an endpoint that would have accepted any
# value without enforcing it.
_NO_EFFORT_PROVIDERS = ("ollama_chat/", "ollama_cloud/")

# Providers that should bypass the HF router catalog lookup in
# ``_print_hf_routing_info``. The probe + LiteLLM adapter still cover
# "does this model exist" on the first real call.
_ROUTING_POLICIES = {"fastest", "cheapest", "preferred"}
_DIRECT_PREFIXES = (
    "anthropic/",
    "openai/",
    "bedrock/",
    "gemini/",
    "ollama_chat/",
    "ollama_cloud/",
    *LOCAL_MODEL_PREFIXES,
)
_LOCAL_PROBE_TIMEOUT = 15.0


def _has_gemini_key() -> bool:
    return bool(
        os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    )


def _has_ollama_cloud_key() -> bool:
    return bool(os.environ.get("OLLAMA_API_KEY"))


def _warn_if_ollama_down(model_id: str, console) -> None:
    """If the user picked an ``ollama_chat/<tag>`` but the daemon isn't
    reachable, warn now — cheaper than a late connection error.

    Also warns if the specific tag isn't currently installed. These are
    warnings only; we still let the switch proceed because the user may
    be about to start the daemon or pull the model.
    """
    from agent.core import ollama_discovery as disco

    tag = model_id.split("/", 1)[1]
    try:
        installed = {m.name for m in disco.list_local_models()}
    except disco.OllamaUnavailable as e:
        console.print(f"[bold red]Warning:[/bold red] {e}")
        return
    if tag not in installed:
        suggestions = ", ".join(sorted(installed)[:4]) or "(none installed)"
        console.print(
            f"[bold red]Warning:[/bold red] '{tag}' isn't installed locally. "
            f"Available: {suggestions}. Pull it with `ollama pull {tag}`."
        )


def is_valid_model_id(model_id: str) -> bool:
    """Loose format check — lets users pick any model id.

    Accepts:
      • anthropic/<model>
      • openai/<model>
      • ollama/<model>, vllm/<model>, lm_studio/<model>, llamacpp/<model>
      • bedrock/<model>
      • gemini/<model>
      • ollama_chat/<tag>
      • ollama_cloud/<tag>
      • <org>/<model>[:<tag>]            (HF router; tag = provider or policy)
      • huggingface/<org>/<model>[:<tag>] (same, accepts legacy prefix)

    Also accepts the bare sentinels ``ollama`` and ``ollama_cloud`` — the
    REPL uses those to open the interactive picker. The picker is
    triggered in ``agent.main``, not here, but we must not reject these
    as malformed.

    Actual availability is verified against the HF router catalog on
    switch, and by the provider on the probe's ping call.
    """
    if not model_id:
        return False
    if model_id in ("ollama", "ollama_cloud"):
        return True
    if is_local_model_id(model_id):
        return True
    if is_reserved_local_model_id(model_id):
        return False
    if any(model_id.startswith(prefix) for prefix in LOCAL_MODEL_PREFIXES):
        return False
    if "/" not in model_id:
        return False
    head = model_id.split(":", 1)[0]
    parts = head.split("/")
    return len(parts) >= 2 and all(parts)


def _print_hf_routing_info(model_id: str, console) -> bool:
    """Show HF router catalog info (providers, price, context, tool support)
    for an HF-router model id. Returns ``True`` to signal the caller can
    proceed with the switch, ``False`` to indicate a hard problem the user
    should notice before we fire the effort probe.

    Anthropic / OpenAI / Bedrock / Gemini / Ollama ids return ``True``
    without printing anything — the probe (or first real call) covers
    "does this model exist".
    """
    if model_id.startswith(_DIRECT_PREFIXES):
        # Native-Ollama health check: if the user picked an ``ollama_chat/``
        # model but the daemon isn't up, surface that now instead of
        # letting the first real request blow up with a connection error.
        if model_id.startswith("ollama_chat/"):
            _warn_if_ollama_down(model_id, console)
        if model_id.startswith("ollama_cloud/") and not _has_ollama_cloud_key():
            console.print(
                "[bold red]Warning:[/bold red] OLLAMA_API_KEY is not set. "
                "Ollama Cloud requests will fail with 401 until you export it."
            )
        if model_id.startswith("gemini/") and not _has_gemini_key():
            console.print(
                "[bold red]Warning:[/bold red] GEMINI_API_KEY (or GOOGLE_API_KEY) "
                "is not set. Gemini requests will fail until you export it."
            )
        return True

    from agent.core import hf_router_catalog as cat

    bare, _, tag = model_id.partition(":")
    info = cat.lookup(bare)
    if info is None:
        console.print(
            f"[bold red]Warning:[/bold red] '{bare}' isn't in the HF router "
            "catalog. Checking anyway — first call may fail."
        )
        suggestions = cat.fuzzy_suggest(bare)
        if suggestions:
            console.print(f"[dim]Did you mean: {', '.join(suggestions)}[/dim]")
        return True

    live = info.live_providers
    if not live:
        console.print(
            f"[bold red]Warning:[/bold red] '{bare}' has no live providers "
            "right now. First call will likely fail."
        )
        return True

    if tag and tag not in _ROUTING_POLICIES:
        matched = [p for p in live if p.provider == tag]
        if not matched:
            names = ", ".join(p.provider for p in live)
            console.print(
                f"[bold red]Warning:[/bold red] provider '{tag}' doesn't serve "
                f"'{bare}'. Live providers: {names}. Checking anyway."
            )

    if not info.any_supports_tools:
        console.print(
            f"[bold red]Warning:[/bold red] no provider for '{bare}' advertises "
            "tool-call support. This agent relies on tool calls — expect errors."
        )

    if tag in _ROUTING_POLICIES:
        policy = tag
    elif tag:
        policy = f"pinned to {tag}"
    else:
        policy = "auto (fastest)"
    console.print(f"  [dim]routing: {policy}[/dim]")
    for p in live:
        price = (
            f"${p.input_price:g}/${p.output_price:g} per M tok"
            if p.input_price is not None and p.output_price is not None
            else "price n/a"
        )
        ctx = f"{p.context_length:,} ctx" if p.context_length else "ctx n/a"
        tools = "tools" if p.supports_tools else "no tools"
        console.print(f"  [dim]{p.provider}: {price}, {ctx}, {tools}[/dim]")
    return True


def print_model_listing(config, console) -> None:
    """Render the default ``/model`` (no-arg) view: current + suggested."""
    current = config.model_name if config else ""
    console.print("[bold]Current model:[/bold]")
    console.print(f"  {current}")
    console.print("\n[bold]Suggested:[/bold]")
    for m in SUGGESTED_MODELS:
        marker = " [dim]<-- current[/dim]" if m["id"] == current else ""
        console.print(f"  {m['id']}  [dim]({m['label']})[/dim]{marker}")
    console.print(
        "\n[dim]Paste any HF model id (e.g. 'MiniMaxAI/MiniMax-M2.7').\n"
        "Add ':fastest', ':cheapest', ':preferred', or ':<provider>' to override routing.\n"
        "Direct APIs: 'anthropic/<model>', 'openai/<model>', 'gemini/<model>'.\n"
        "Use 'ollama/<model>', 'vllm/<model>', 'lm_studio/<model>', or "
        "'llamacpp/<model>' for local OpenAI-compatible endpoints.\n"
        "Ollama:\n"
        "  • '/model ollama'         — arrow-key picker of locally installed models\n"
        "  • '/model ollama_chat/qwen2.5:14b'  — pick a specific local tag\n"
        "  • '/model ollama_cloud'   — arrow-key picker for Ollama Cloud (needs OLLAMA_API_KEY)\n"
        "  • '/model ollama_cloud/gpt-oss:120b-cloud' — pick a specific cloud tag[/dim]"
    )


def print_invalid_id(arg: str, console) -> None:
    console.print(f"[bold red]Invalid model id format:[/bold red] {arg}")
    console.print(
        "[dim]Expected:\n"
        "  • <org>/<model>[:tag]    (HF router — paste from huggingface.co)\n"
        "  • anthropic/<model>\n"
        "  • openai/<model>\n"
        "  • gemini/<model>\n"
        "  • ollama_chat/<tag> | ollama_cloud/<tag>\n"
        "  • ollama/<model> | vllm/<model> | lm_studio/<model> | llamacpp/<model>[/dim]"
    )


async def _probe_local_model(model_id: str) -> None:
    params = _resolve_llm_params(model_id)
    await asyncio.wait_for(
        acompletion(
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            stream=False,
            **params,
        ),
        timeout=_LOCAL_PROBE_TIMEOUT,
    )


async def probe_and_switch_model(
    model_id: str,
    config,
    session,
    console,
    hf_token: str | None,
) -> None:
    """Validate model+effort with a 1-token ping, cache the effective effort,
    then commit the switch.

    Three visible outcomes:

    * ✓ ``effort: <level>`` — model accepted the preferred effort (or a
      fallback from the cascade; the note explains if so)
    * ✓ ``effort: off`` — model doesn't support thinking; we'll strip it
    * ✗ hard error (auth, model-not-found, quota) — we reject the switch
      and keep the current model so the user isn't stranded

    For non-local models, transient errors (5xx, timeout) complete the switch
    with a yellow warning; the next real call re-surfaces the error if it's
    persistent. Local models reject every probe error, including timeouts, and
    keep the current model.
    """
    if is_local_model_id(model_id):
        console.print(f"[dim]checking local model {model_id}...[/dim]")
        try:
            await _probe_local_model(model_id)
        except Exception as e:
            console.print(f"[bold red]Switch failed:[/bold red] {e}")
            console.print(f"[dim]Keeping current model: {config.model_name}[/dim]")
            return

        _commit_switch(model_id, config, session, effective=None, cache=True)
        console.print(
            f"[green]Model switched to {model_id}[/green] [dim](effort: off)[/dim]"
        )
        return

    preference = config.reasoning_effort
    if not _print_hf_routing_info(model_id, console):
        return

    # Ollama (native + cloud) endpoints silently accept unknown params,
    # which would cause the probe to cache a wrong "effective" level.
    # Commit with effort=None and skip the probe. ``cache=False`` so a
    # future /effort change can still try to re-send something (it won't
    # land either, but the control flow stays consistent).
    if model_id.startswith(_NO_EFFORT_PROVIDERS):
        _commit_switch(model_id, config, session, effective=None, cache=False)
        console.print(
            f"[green]Model switched to {model_id}[/green] "
            "[dim](effort: off — Ollama ignores effort params)[/dim]"
        )
        return

    if not preference:
        # Nothing to validate with a ping that we couldn't validate on the
        # first real call just as cheaply. Skip the probe entirely.
        _commit_switch(model_id, config, session, effective=None, cache=False)
        console.print(
            f"[green]Model switched to {model_id}[/green] [dim](effort: off)[/dim]"
        )
        return

    console.print(f"[dim]checking {model_id} (effort: {preference})...[/dim]")
    try:
        outcome = await probe_effort(model_id, preference, hf_token, session=session)
    except ProbeInconclusive as e:
        _commit_switch(model_id, config, session, effective=None, cache=False)
        console.print(
            f"[yellow]Model switched to {model_id}[/yellow] "
            f"[dim](couldn't validate: {e}; will verify on first message)[/dim]"
        )
        return
    except Exception as e:
        # Hard persistent error — auth, unknown model, quota. Don't switch.
        console.print(f"[bold red]Switch failed:[/bold red] {e}")
        console.print(f"[dim]Keeping current model: {config.model_name}[/dim]")
        return

    _commit_switch(
        model_id,
        config,
        session,
        effective=outcome.effective_effort,
        cache=True,
    )
    effort_label = outcome.effective_effort or "off"
    suffix = f" — {outcome.note}" if outcome.note else ""
    console.print(
        f"[green]Model switched to {model_id}[/green] "
        f"[dim](effort: {effort_label}{suffix}, {outcome.elapsed_ms}ms)[/dim]"
    )


async def handle_ollama_picker(
    kind: str,
    config,
    session,
    console,
    hf_token: str | None,
) -> None:
    """Open the interactive picker and, if a model is chosen, switch to it.

    ``kind`` is ``"local"`` (native daemon via /api/tags) or ``"cloud"``
    (curated Ollama Cloud list). Cancel / unavailable daemon prints a
    message and returns — current model is untouched.
    """
    from agent.core import ollama_picker

    if kind == "local":
        chosen = await ollama_picker.pick_local_model(console)
    elif kind == "cloud":
        chosen = await ollama_picker.pick_cloud_model(console)
    else:
        console.print(f"[red]Unknown Ollama picker kind: {kind}[/red]")
        return

    if not chosen:
        console.print("[dim]No model selected. Current model unchanged.[/dim]")
        return

    await probe_and_switch_model(chosen, config, session, console, hf_token)


def _commit_switch(model_id, config, session, effective, cache: bool) -> None:
    """Apply the switch to the session (or bare config if no session yet).

    ``effective`` is the probe's resolved effort; ``cache=True`` stores it
    in the session's per-model cache so real calls use the resolved level
    instead of re-probing. ``cache=False`` (inconclusive probe / effort
    off) leaves the cache untouched — next call falls back to preference.
    """
    if session is not None:
        session.update_model(model_id)
        if cache:
            session.model_effective_effort[model_id] = effective
        else:
            session.model_effective_effort.pop(model_id, None)
    else:
        config.model_name = model_id
