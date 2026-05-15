"""Interactive arrow-key picker for Ollama models.

Invoked from the ``/model ollama`` (bare, no model specified) path in the
REPL. Uses ``prompt_toolkit.shortcuts.radiolist_dialog`` — the same
library the REPL already depends on for input — so we don't add new
dependencies.

Two pickers live here:

* ``pick_local_model()`` — hits the local daemon via ``/api/tags``,
  renders installed models, returns the selected ``ollama_chat/<tag>``
  id (or ``None`` if cancelled / daemon down).

* ``pick_cloud_model()`` — shows a curated list from
  ``ollama_discovery.cloud_model_suggestions`` for Ollama Cloud, returns
  an ``ollama_cloud/<tag>`` id.

Both functions handle "ESC was pressed" by returning ``None``. The caller
is responsible for printing that cancellation message.
"""

from __future__ import annotations

from typing import Optional

from agent.core import ollama_discovery as disco


def _format_size(n: int) -> str:
    """Pretty-print a byte count as GB (anything else is irrelevant here)."""
    if n <= 0:
        return "n/a"
    gb = n / (1024 ** 3)
    if gb >= 10:
        return f"{gb:.0f} GB"
    return f"{gb:.1f} GB"


def _annotate_local(model: disco.OllamaModel) -> str:
    """Build the right-hand descriptor line for a local model entry."""
    bits: list[str] = []
    if model.parameter_size:
        bits.append(model.parameter_size)
    if model.quantization:
        bits.append(model.quantization)
    bits.append(_format_size(model.size_bytes))

    warn: list[str] = []
    if not disco.looks_toolcall_capable(model):
        warn.append("unknown tool-call")
    if disco.looks_too_small_for_agent(model):
        warn.append("small model")

    tail = f" — {', '.join(warn)}" if warn else ""
    return f"{' · '.join(bits)}{tail}"


async def _run_dialog_safely(dialog) -> Optional[str]:
    """Run a prompt_toolkit dialog from an async context.

    ``Dialog.run()`` wraps ``asyncio.run(run_async())``, which blows up
    inside the REPL's running event loop with
    "asyncio.run() cannot be called from a running event loop". We're
    always called from the /model handler, which is async — so we
    always have a running loop — so we always want ``run_async()``
    directly. Returns the selected value or ``None`` (cancel).
    """
    return await dialog.run_async()


def _filter_installed_locally(models: list[disco.OllamaModel]) -> list[disco.OllamaModel]:
    """Drop cloud-proxy entries from a /api/tags listing.

    Modern Ollama (0.5+) surfaces cloud-hosted models alongside local
    ones in /api/tags: they carry a ``:cloud`` suffix and zero bytes on
    disk. Those belong in the cloud picker, not the local one. We filter
    them out here so the local picker only shows models you can actually
    run on your hardware.
    """
    return [
        m for m in models
        if not m.name.endswith(":cloud") and m.size_bytes > 0
    ]


async def pick_local_model(console) -> Optional[str]:
    """Show an arrow-key picker of locally-installed Ollama models.

    Async because prompt_toolkit dialogs must be driven with
    ``run_async()`` when we're already inside an event loop (the REPL
    is). Returns an id like ``ollama_chat/qwen2.5:14b`` or ``None``
    on cancel / daemon down / no installed models.
    """
    try:
        models = disco.list_local_models()
    except disco.OllamaUnavailable as e:
        console.print(f"[bold red]Ollama unavailable:[/bold red] {e}")
        return None

    models = _filter_installed_locally(models)
    if not models:
        console.print(
            "[yellow]No local Ollama models found.[/yellow] "
            "Install one with e.g. `ollama pull qwen2.5:14b`, then try again. "
            "(Cloud-proxy entries — anything tagged `:cloud` — are filtered "
            "out of this picker; use `/model ollama_cloud` for those.)"
        )
        return None

    try:
        from prompt_toolkit.shortcuts import radiolist_dialog
    except ImportError as e:
        console.print(f"[bold red]prompt_toolkit dialogs unavailable:[/bold red] {e}")
        return None

    values = [
        (f"ollama_chat/{m.name}", f"{m.name:<30}  {_annotate_local(m)}")
        for m in models
    ]

    console.print(
        "[dim]Pick a local model. 'small model' or 'unknown tool-call' tags mean\n"
        "the model may struggle with this agent's tool-heavy workload. You can\n"
        "still pick them — you'll get a warning, not a block.[/dim]"
    )

    try:
        chosen = await _run_dialog_safely(radiolist_dialog(
            title="Local Ollama models",
            text="Use ↑/↓ to move, Space to select, Enter to confirm, Esc to cancel.",
            values=values,
        ))
    except Exception as e:
        # Dialog can still throw on terminals without full ANSI support
        # (stripped CI, some IDE terminals). Fall back to numeric input.
        console.print(
            f"[yellow]Arrow-key dialog failed ({e}); falling back to numeric input.[/yellow]"
        )
        return _numeric_fallback(values, console)

    return chosen  # None on cancel


async def pick_cloud_model(console) -> Optional[str]:
    """Show an arrow-key picker of curated Ollama Cloud models.

    Returns an id like ``ollama_cloud/gpt-oss:120b-cloud``, or ``None``
    on cancel.
    """
    suggestions = disco.cloud_model_suggestions()
    if not suggestions:
        console.print("[yellow]No Ollama Cloud models configured.[/yellow]")
        return None

    try:
        from prompt_toolkit.shortcuts import radiolist_dialog
    except ImportError as e:
        console.print(f"[bold red]prompt_toolkit dialogs unavailable:[/bold red] {e}")
        return None

    values = [
        (f"ollama_cloud/{tag}", f"{tag:<30}  {desc}")
        for tag, desc in suggestions
    ]

    try:
        chosen = await _run_dialog_safely(radiolist_dialog(
            title="Ollama Cloud models",
            text="Requires OLLAMA_API_KEY. Use ↑/↓, Enter to confirm, Esc to cancel.",
            values=values,
        ))
    except Exception as e:
        console.print(
            f"[yellow]Arrow-key dialog failed ({e}); falling back to numeric input.[/yellow]"
        )
        return _numeric_fallback(values, console)

    return chosen


def _numeric_fallback(values: list[tuple[str, str]], console) -> Optional[str]:
    """Degenerate-terminal fallback when radiolist_dialog fails.

    Prints a numbered list and reads a number from stdin. Better than
    leaving the user stranded with no way to pick. ``values`` is the
    same ``[(id, label)]`` shape ``radiolist_dialog`` expects.
    """
    for i, (_id, label) in enumerate(values, 1):
        console.print(f"  [dim]{i:>2}.[/dim] {label}")
    try:
        raw = input("Pick a number (blank to cancel): ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not raw:
        return None
    try:
        idx = int(raw) - 1
    except ValueError:
        console.print("[red]Not a number; cancelled.[/red]")
        return None
    if idx < 0 or idx >= len(values):
        console.print("[red]Out of range; cancelled.[/red]")
        return None
    return values[idx][0]
