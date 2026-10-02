"""Model-free checks of the adapter's tensor plumbing (no downloads needed)."""

import torch

from speechllm.adapter.data import render_user_turn
from speechllm.adapter.model import AUDIO_MARKER, FrameStackProjector


def test_projector_downsamples_and_pads():
    proj = FrameStackProjector(in_dim=8, out_dim=16, hidden=32, k=4)
    out = proj(torch.randn(2, 10, 8))  # 10 frames -> ceil(10/4) = 3 tokens
    assert out.shape == (2, 3, 16)


def test_projector_is_trainable():
    proj = FrameStackProjector(in_dim=8, out_dim=16, hidden=32, k=2)
    x, y = torch.randn(4, 6, 8), torch.randn(4, 3, 16)
    opt = torch.optim.Adam(proj.parameters(), lr=1e-2)
    first = None
    for _ in range(50):
        loss = torch.nn.functional.mse_loss(proj(x), y)
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first


def test_task_tags():
    t = render_user_turn("transcribe", "mn")
    assert t.startswith(AUDIO_MARKER) and "[task: transcribe] [lang: mn]" in t
    # the language is the *answer* for LID, so it must not leak into the prompt
    assert "[lang:" not in render_user_turn("lid", "mn")
    assert "[lang:" not in render_user_turn("lid+asr", "mn")
