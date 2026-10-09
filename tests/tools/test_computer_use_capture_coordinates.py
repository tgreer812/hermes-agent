"""Coordinate guidance must not turn desktop AX bounds into window click pixels."""

import base64
import json
from io import BytesIO

import pytest
from PIL import Image

from tools.computer_use import tool as cu_tool
from tools.computer_use.backend import CaptureResult, UIElement


@pytest.mark.parametrize("lane", ["native", "aux", "aux_unavailable", "unchanged"])
def test_capture_dispatch_preserves_window_pixels_and_backend_geometry(
    lane, monkeypatch, grant_computer_use_approvals,
):
    # The enclosing window makes the old heuristic report an invented scale even
    # though this is only an offset window. Button centre (328, 203) on the desktop
    # corresponds to screenshot point (220, 102), not a scaled desktop point.
    image = BytesIO()
    Image.new("RGB", (440, 220)).save(image, format="PNG")
    cap = CaptureResult(
        mode="som", width=440, height=220,
        png_b64=base64.b64encode(image.getvalue()).decode(),
        elements=[
            UIElement(index=1, role="Window", bounds=(108, 101, 440, 220)),
            UIElement(index=2, role="Button", label="OK", bounds=(298, 190, 60, 26)),
        ],
        note="Legacy backend guidance must be preserved.",
    )
    backend = cu_tool._NoopBackend()
    monkeypatch.setattr(backend, "capture", lambda **kwargs: cap)
    monkeypatch.setattr(cu_tool, "_new_backend", lambda mode: backend)
    monkeypatch.setattr(cu_tool, "_should_route_through_aux_vision", lambda: lane.startswith("aux"))

    if lane.startswith("aux"):
        import model_tools

        def analyze(coro):
            coro.close()  # The remote vision call is the only mocked boundary.
            return json.dumps({"analysis": "A button."}) if lane == "aux" else ""

        monkeypatch.setattr(model_tools, "_run_async", analyze)

    # Exercise real registration and dispatch, not only the formatting helper.
    import tools.computer_use_tool
    from tools.registry import registry

    cu_tool.reset_backend_for_tests()
    try:
        args = {"action": "capture", "mode": "som"}
        out = registry.dispatch("computer_use", args, session_id="coordinate-contract")
        if lane == "unchanged":
            out = registry.dispatch("computer_use", args, session_id="coordinate-contract")
        payload = out["meta"] if isinstance(out, dict) else json.loads(out)
        summary = out["text_summary"] if isinstance(out, dict) else payload["summary"]
        assert "bounds_scale" not in payload
        assert "captured-window screenshot pixels" in summary
        assert "verified transform" in summary
        assert "estimated scale" not in summary
        assert cap.note in summary
        assert (payload["width"], payload["height"]) == (440, 220)
        if isinstance(out, dict):
            assert out["content"][0]["text"] == summary
            assert "(298, 190, 60, 26)" in summary
        else:
            assert payload["elements"][1]["bounds"] == [298, 190, 60, 26]
        if lane == "aux":
            assert payload["vision_analysis"] == "A button."
        elif lane == "aux_unavailable":
            assert payload["vision_unavailable"] is True
        elif lane == "unchanged":
            assert payload["screen_unchanged"] is True

        registry.dispatch("computer_use", {"action": "click", "coordinate": [220, 102]}, session_id="coordinate-contract")
        click = next(kw for action, kw in backend.calls if action == "click")
        assert (click["x"], click["y"]) == (220, 102)
        registry.dispatch("computer_use", {"action": "click", "element": 2}, session_id="coordinate-contract")
        assert backend.calls[-1][1]["element"] == 2
    finally:
        cu_tool.reset_backend_for_tests()
