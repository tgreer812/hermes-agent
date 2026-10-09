"""``/mymodels`` — cross-provider model listing.

Covers the three things the first live run got wrong: keyless local providers
(LM Studio) must appear even though they never authenticate, duplicate
``custom`` ids must collapse, and the current model / alias'd models must sort
ahead of the alphabetical catalog tail so they survive the display cap.
"""

from types import SimpleNamespace

import pytest

from hermes_cli.cli_commands_mixin import CLICommandsMixin
from hermes_cli.commands import resolve_command


def _mixin(provider: str, model: str, monkeypatch) -> type:
    """A mixin subclass with stubbed provider/model state and stubbed catalogs."""
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "model": {
                "provider": "lmstudio",
                "default": "qwen/qwen3.8-27b",
                "base_url": "http://127.0.0.1:1234/v1",
                "aliases": {"glm53": "openrouter/z-ai/glm-5.3", "glm53f": "openrouter/z-ai/glm-5.3-flash"},
            },
            "agent": {"reasoning_overrides": {"z-ai/glm-5.3": "low"}},
        },
        raising=False,
    )
    monkeypatch.setattr(
        "hermes_cli.models.list_available_providers",
        lambda: [
            {"id": "openrouter", "label": "OpenRouter", "aliases": [], "authenticated": True},
            {"id": "lmstudio", "label": "lmstudio", "aliases": [], "authenticated": False},
            {"id": "custom", "label": "Custom endpoint", "aliases": [], "authenticated": True},
            {"id": "custom", "label": "Custom endpoint", "aliases": [], "authenticated": True},
        ],
        raising=False,
    )
    monkeypatch.setattr(
        "agent.models_dev.list_provider_models",
        lambda pid, *, allow_network=True: (
            ["z-ai/glm-5.3", "z-ai/glm-5.3-flash", "anthropic/claude-fable-5.1"]
            if pid == "openrouter" else []
        ),
        raising=False,
    )
    monkeypatch.setattr(
        "hermes_cli.models_local.fetch_lmstudio_models",
        lambda base_url=None, **kw: ["qwen/qwen3.8-27b"],
        raising=False,
    )
    return type("StubCLI", (CLICommandsMixin,), {"provider": provider, "model": model})


def test_registry_entry_and_dispatch_name():
    cmd = resolve_command("mymodels")
    assert cmd is not None and cmd.name == "mymodels"
    # The slash dispatcher falls back to _handle_<name>_command by convention.
    assert callable(getattr(CLICommandsMixin, "_handle_mymodels_command", None))


def test_lists_all_authenticated_providers_and_marks_current(
        monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    Stub = _mixin("openrouter", "z-ai/glm-5.3", monkeypatch)
    Stub()._handle_mymodels_command("/mymodels")
    out = capsys.readouterr().out

    assert "openrouter" in out and "<-- current" in out
    assert "lmstudio" in out                       # keyless local provider still listed
    assert out.count("Custom endpoint (custom)") == 1       # duplicate custom ids collapse
    assert "[current" in out and "z-ai/glm-5.3" in out


def test_current_model_and_aliases_sort_ahead_of_cap(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    Stub = _mixin("openrouter", "z-ai/glm-5.3", monkeypatch)
    Stub()._handle_mymodels_command("/mymodels")
    out = capsys.readouterr().out
    lines = out.splitlines()

    openrouter_start = next(i for i, l in enumerate(lines) if "openrouter" in l)
    model_lines = [l for l in lines[openrouter_start + 1:] if l.startswith("    ")]
    assert "[current" in model_lines[0]           # current model is row one
    assert any("alias: glm53f" in l for l in model_lines)


def test_lmstudio_current_provider_probes_local_endpoint(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    Stub = _mixin("lmstudio", "qwen/qwen3.8-27b", monkeypatch)
    Stub()._handle_mymodels_command("/mymodels")
    out = capsys.readouterr().out

    assert "lmstudio" in out and "<-- current" in out
    assert "qwen/qwen3.8-27b" in out


def test_all_flag_lifts_display_cap(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    Stub = _mixin("openrouter", "z-ai/glm-5.3", monkeypatch)
    monkeypatch.setattr(
        "agent.models_dev.list_provider_models",
        lambda pid, *, allow_network=True: [f"vendor/model-{i:03}" for i in range(40)]
        if pid == "openrouter" else [],
        raising=False,
    )
    capped = Stub()._handle_mymodels_command("/mymodels")
    out_capped = capsys.readouterr().out
    assert "more (" in out_capped

    Stub()._handle_mymodels_command("/mymodels all")
    out_all = capsys.readouterr().out
    assert "vendor/model-039" in out_all
