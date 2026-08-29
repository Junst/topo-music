"""Codec loading + codebook extraction for L1'.

L1' asks whether the RVQ codebooks that music/audio LMs already use are
topographically organised. To answer it we need, per codec:
  - the codebook matrices  E_l  in  R^{K x d}  for each RVQ level l
  - a way to encode audio to per-level code indices

Three codecs, chosen to span the space actually used by music generation:
  encodec_32khz : MusicGen's codec. Music-trained. The most on-target.
  dac_44khz     : Descript Audio Codec. Widely used, 44.1 kHz, music+speech+env.
  encodec_24khz : general-audio EnCodec. Off-target control -- if structure shows
                  up here too, it is not a music fact.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

os.environ.setdefault("HF_HOME", "/scratch2/solbon1212/hf_cache")

CODECS = {
    "encodec_32k": dict(repo="facebook/encodec_32khz", sr=32000, kind="encodec"),
    "dac_44k":     dict(repo="descript/dac_44khz",     sr=44100, kind="dac"),
    "encodec_24k": dict(repo="facebook/encodec_24khz", sr=24000, kind="encodec"),
}


@dataclass
class Codec:
    name: str
    sr: int
    model: torch.nn.Module
    codebooks: np.ndarray          # [L, K, d]
    encode: Callable[[torch.Tensor], torch.Tensor]   # wav [B,1,T] -> codes [B, L, T']


def _encodec_codebooks(model) -> np.ndarray:
    """EnCodec keeps one EuclideanCodebook per RVQ layer, weight [K, d]."""
    layers = model.quantizer.layers
    return np.stack([l.codebook.embed.detach().cpu().numpy() for l in layers], 0)


def _dac_codebooks(model) -> np.ndarray:
    """DAC projects to a low-dim (8) space before lookup; codebook is an
    nn.Embedding of shape [K, d_low]. That low d is itself a finding for L1'."""
    return np.stack(
        [q.codebook.weight.detach().cpu().numpy() for q in model.quantizer.quantizers], 0
    )


def load(name: str, device: str = "cpu") -> Codec:
    cfg = CODECS[name]
    if cfg["kind"] == "encodec":
        from transformers import EncodecModel
        m = EncodecModel.from_pretrained(cfg["repo"]).to(device).eval()
        cb = _encodec_codebooks(m)

        def enc(wav: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                # highest bandwidth -> all RVQ levels
                out = m.encode(wav.to(device), bandwidth=max(m.config.target_bandwidths))
            return out.audio_codes[0].cpu()             # [B, L, T']
    else:
        from transformers import DacModel
        m = DacModel.from_pretrained(cfg["repo"]).to(device).eval()
        cb = _dac_codebooks(m)

        def enc(wav: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                out = m.encode(wav.to(device))
            return out.audio_codes.cpu()                # [B, L, T']

    for p in m.parameters():
        p.requires_grad_(False)
    return Codec(name=name, sr=cfg["sr"], model=m, codebooks=cb, encode=enc)
