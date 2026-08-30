"""Feature extraction for PupuJEPA, which ships training code but no inference path.

Everything here is read off the released checkpoint's own args.json and the
authors' training script, so the mel matches what the model was trained on:
reflect padding by (n_fft - hop)/2, a non-centred STFT, the librosa mel basis,
a log with a floor at 1e-5, and the fixed normalisation constants that appear
literally in train_pupujepa.py. The time and frequency axes are then swapped,
because the config sets flip_ft.

Features come from the *teacher*, the EMA target encoder, run on the complete
patch grid with no masking. That is what the target branch sees during training
for the patches it is asked about, and it is the branch a JEPA is usually
probed from. The student would also work but is the branch trained on a masked
subset.

The encoder takes a fixed 1024 mel frames, which is 10.24 s at 100 frames per
second, so a longer clip is cut into consecutive windows and their pooled token
means are averaged.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import torch
from librosa.filters import mel as librosa_mel_fn

PUPU_SRC = Path.home() / "src" / "PupuJEPA"
PUPU_CKPT = Path("/scratch2/solbon1212/ckpt/pupujepa")
SR = 24000
FRAMES = 1024                       # the image height the patch grid expects

_MEL_MEAN, _MEL_STD = -4.089994845986366, 2.0242277159094813


def _ns(d):
    """args.json as attribute-access config, which is what the model expects."""
    if isinstance(d, dict):
        return types.SimpleNamespace(**{k: _ns(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_ns(v) for v in d]
    return d


def load(device="cuda"):
    if str(PUPU_SRC) not in sys.path:
        sys.path.insert(0, str(PUPU_SRC))
    from model import PupuJEPA
    from safetensors.torch import load_file

    import json5                       # args.json has trailing commas
    cfg = _ns(json5.loads((PUPU_CKPT / "args.json").read_text()))
    if not hasattr(cfg, "train"):
        cfg.train = types.SimpleNamespace()
    if not hasattr(cfg.train, "grad_checkpointing_step"):
        cfg.train.grad_checkpointing_step = None

    model = PupuJEPA(cfg)
    sd = load_file(str(PUPU_CKPT / "model.safetensors"))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    keep = [k for k in missing if k.startswith(("teacher.", "patch_embed."))]
    if keep:
        raise RuntimeError(f"checkpoint is missing encoder weights: {keep[:5]}")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg


_basis: dict = {}


def mel(y: torch.Tensor, cfg) -> torch.Tensor:
    """[B, N] waveform at 24 kHz to [B, 1, T, n_mels], as trained."""
    pp = cfg.preprocess
    key = (pp.fmax, str(y.device))
    if key not in _basis:
        m = librosa_mel_fn(sr=pp.sample_rate, n_fft=pp.n_fft, n_mels=pp.n_mels,
                           fmin=pp.fmin, fmax=pp.fmax)
        _basis[key] = (torch.from_numpy(m).float().to(y.device),
                       torch.hann_window(pp.win_size).to(y.device))
    mb, win = _basis[key]
    pad = int((pp.n_fft - pp.hop_size) / 2)
    y = torch.nn.functional.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
    spec = torch.stft(y, pp.n_fft, hop_length=pp.hop_size, win_length=pp.win_size,
                      window=win, center=False, pad_mode="reflect",
                      normalized=False, onesided=True, return_complex=True)
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
    spec = torch.matmul(mb, spec)
    spec = torch.log(torch.clamp(spec, min=1e-5))
    if pp.normalize:
        spec = (spec - _MEL_MEAN) / (_MEL_STD + 1e-8)
    if pp.flip_ft:
        spec = spec.transpose(-2, -1)
    return spec.unsqueeze(1)


@torch.no_grad()
def tokens(model, cfg, waves: np.ndarray, device="cuda"):
    """[B, N] waveforms at 24 kHz to a list of [n_tokens, D] per clip.

    Tokens rather than one pooled vector, so the caller can take the same mean
    and standard deviation over the sequence that every other arm gets.
    """
    x = torch.from_numpy(np.ascontiguousarray(waves)).float().to(device)
    m = mel(x, cfg)                                   # [B, 1, T, n_mels]
    T = m.shape[2]
    if T < FRAMES:                                    # tile a short clip
        m = m.repeat(1, 1, int(np.ceil(FRAMES / T)), 1)
        T = m.shape[2]
    outs = []
    for s in range(0, T - FRAMES + 1, FRAMES):
        win = m[:, :, s:s + FRAMES]
        grid = model.get_dynamic_grid_size(win)
        tok = model.patch_embed(win)
        rope = model.rope_encoder.get_embed(grid)
        h = model.teacher(tok, rope=rope.unsqueeze(0).unsqueeze(1)
                          if rope.dim() == 2 else rope.unsqueeze(1))
        outs.append(h.float().cpu().numpy())          # [B, n_tok, D]
    cat = np.concatenate(outs, axis=1)
    return [cat[i] for i in range(cat.shape[0])]


def embed(model, cfg, waves, device="cuda"):
    return np.stack([t.mean(0) for t in tokens(model, cfg, waves, device)])
