"""MARBLE-style learned layer weighting, as a control on the layer sweep.

Table 1 reports each encoder's default output and the best of a four-layer
sweep. A reviewer can reasonably ask whether the best layer is a selection
artifact, since it was chosen after seeing the numbers. MARBLE's own protocol
does not select a layer at all: it learns a softmax weight over every hidden
layer jointly with the linear head, so the representation is a single fixed
combination that no one picked by hand.

This runs that protocol on the same clips, and then applies the identical three
measurements to the resulting weighted sum. Two things can happen and both are
informative. If the weighted sum lands near the best layer, the sweep was not
buying anything a standard protocol would not also find. If it lands near the
default output, the best-layer rows are reporting something the standard
protocol misses, which is worth saying out loud.

Stage `extract` caches every layer once, because the fit is cheap and the
forward pass is not. Stage `fit` learns the weights and writes a JSON in the
same schema as script 09, so scripts 19 and 26 pick the arm up unchanged.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from importlib.machinery import SourceFileLoader

m9 = SourceFileLoader("m9", "scripts/09_clip_key_geometry.py").load_module()

# MERT exposes 24 encoder layers plus the feature projection, MuQ 12 plus its
# input, MATPAC 12 with no separate input tap.
ENCODERS = {"mert": "mert_L12", "muq": "muq_L6", "matpac": "matpac_L6"}


def layers_path(enc, dataset):
    tag = "" if dataset == "gs" else f"_{dataset}"
    return Path(f"runs/layers_{enc}{tag}.npz")


def extract(a):
    """One forward pass per clip, keeping mean and std of every layer."""
    import librosa
    import soundfile as sf

    rows = m9.fmak_rows() if a.dataset == "fmak" else m9.gs_rows(
        ["GS.train.jsonl", "GS.val.jsonl", "GS.test.jsonl"])
    print(f"[{a.enc}] {a.dataset}: {len(rows)} clips", flush=True)

    taps = {}
    if a.enc == "mert":
        from transformers import AutoModel
        model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M",
                                          trust_remote_code=True).to(a.device).eval()

        def tap(i):
            def fn(_m, _in, o):
                taps[i] = o[0] if isinstance(o, tuple) else o
            return fn
        model.feature_projection.register_forward_hook(tap(0))
        for i, lyr in enumerate(model.encoder.layers):
            lyr.register_forward_hook(tap(i + 1))
        native_sr = m9.MERT_SR
    elif a.enc == "muq":
        from muq import MuQ
        # the same fix scripts 09, 10 and 14 use, so a successful run here
        # exercises their MuQ path too
        model = m9.patch_muq_hidden_states(
            MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter").to(a.device).eval())
        native_sr = m9.MUQ_SR
    elif a.enc == "matpac":
        from matpac.model import get_matpac
        model = get_matpac(checkpoint_path=m9.MATPAC_CKPT,
                           pull_time_dimension=False).to(a.device).eval()
        native_sr = m9.MATPAC_SR
    for p_ in model.parameters():
        p_.requires_grad_(False)

    def batch_layers(paths):
        ws = []
        for p in paths:
            x, sr = sf.read(p, dtype="float32", frames=int(a.seconds * 44100),
                            start=int(2.0 * 44100))
            if x.ndim > 1:
                x = x.mean(1)
            ws.append(librosa.resample(x, orig_sr=sr, target_sr=native_sr)
                      if sr != native_sr else x)
        n = min(len(w) for w in ws)
        x = torch.from_numpy(np.stack([w[:n] for w in ws])).to(a.device)
        with torch.no_grad():
            if a.enc == "mert":
                taps.clear()
                model(x)
                H = [taps[i].float() for i in sorted(taps)]
            elif a.enc == "muq":
                H = [h.float() for h in
                     model(x, output_hidden_states=True).hidden_states]
            else:
                _, lay = model(x)                       # [B, 12, T, D]
                H = [lay[:, i].float() for i in range(lay.shape[1])]
        # [B, L, 2D], matching script 09's [mean | std] embedding per layer
        return torch.stack([torch.cat([h.mean(1), h.std(1)], -1)
                            for h in H], 1).cpu().numpy().astype(np.float16)

    Z, Y, t0 = [], [], time.time()
    for b0 in range(0, len(rows), a.batch):
        br = rows[b0:b0 + a.batch]
        Z.append(batch_layers([r[0] for r in br]))
        Y.extend([r[1] * 2 + r[2] for r in br])
        if b0 % (a.batch * 60) == 0:
            print(f"{b0+len(br)}/{len(rows)}  {time.time()-t0:.0f}s", flush=True)
    Z = np.concatenate(Z)
    Y = np.array(Y)
    out = layers_path(a.enc, a.dataset)
    np.savez_compressed(out, Z=Z, Y=Y, keys=np.array(sorted(set(Y.tolist()))))
    print(f"wrote {out}  Z={Z.shape} ({time.time()-t0:.0f}s)", flush=True)


class WeightedSum(torch.nn.Module):
    """Softmax over layers, then one linear head. MARBLE's probing model."""

    def __init__(self, n_layers, dim, n_classes):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.zeros(n_layers))
        self.head = torch.nn.Linear(dim, n_classes)

    def weights(self):
        return torch.softmax(self.alpha, 0)

    def forward(self, Z):                                 # Z is [B, L, D]
        return self.head((Z * self.weights()[None, :, None]).sum(1))


def train(Zt, Yt, epochs, lr, wd, seed=0, fixed=None):
    """`fixed` pins the layer weights to one layer, leaving only the head free.

    The macro-F1 elsewhere in the paper comes from sklearn's logistic
    regression, so a weighted sum trained with AdamW is not directly comparable
    to it: part of any difference would be the optimizer. Pinning this same
    trainer to a single layer gives the like-for-like baseline.
    """
    torch.manual_seed(seed)
    net = WeightedSum(Zt.shape[1], Zt.shape[2], int(Yt.max()) + 1)
    if fixed is not None:
        with torch.no_grad():
            net.alpha.fill_(-20.0)
            net.alpha[fixed] = 20.0
        net.alpha.requires_grad_(False)
    opt = torch.optim.AdamW([q for q in net.parameters() if q.requires_grad],
                            lr=lr, weight_decay=wd)
    Zt, Yt = torch.from_numpy(Zt), torch.from_numpy(Yt)
    n = len(Yt)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for b0 in range(0, n, 256):
            i = perm[b0:b0 + 256]
            opt.zero_grad()
            torch.nn.functional.cross_entropy(net(Zt[i]), Yt[i]).backward()
            opt.step()
    return net


def fit(a):
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import spearmanr
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    d = np.load(layers_path(a.enc, a.dataset))
    Z, Y = d["Z"].astype(np.float32), d["Y"].astype(np.int64)
    D = Z.shape[2] // 2
    Z = Z[:, :, :D]                       # mean pooling, as in Tables 1 and 3
    present = np.array(sorted(set(Y.tolist())))
    pos = {k: i for i, k in enumerate(present)}
    Yc = np.array([pos[y] for y in Y])
    print(f"[{a.enc}] {Z.shape[0]} clips, {Z.shape[1]} layers, dim {D}", flush=True)

    # track ids, so folds never split a track (FMAK is one excerpt per track)
    groups = None
    if a.dataset == "gs":
        rows = m9.gs_rows(["GS.train.jsonl", "GS.val.jsonl", "GS.test.jsonl"])
        groups = np.array([r[0].stem.rsplit("-", 1)[0] for r in rows])
        assert len(groups) == len(Y)

    def standardize(tr, te):
        mu = Z[tr].reshape(-1, Z.shape[1] * D).mean(0).reshape(Z.shape[1], D)
        sd = Z[tr].reshape(-1, Z.shape[1] * D).std(0).reshape(Z.shape[1], D) + 1e-6
        return (Z[tr] - mu) / sd, (Z[te] - mu) / sd

    splitter = (StratifiedGroupKFold(5, shuffle=True, random_state=0)
                if groups is not None else
                StratifiedKFold(5, shuffle=True, random_state=0))
    f1s, alphas = [], []
    for tr, te in splitter.split(Z, Yc, groups):
        Ztr, Zte = standardize(tr, te)
        net = train(Ztr, Yc[tr], a.epochs, a.lr, a.wd)
        with torch.no_grad():
            pred = net(torch.from_numpy(Zte)).argmax(1).numpy()
        f1s.append(f1_score(Yc[te], pred, average="macro"))
        alphas.append(net.weights().detach().numpy())
        print(f"  fold macro-F1 {f1s[-1]:.3f}", flush=True)
    macro_f1 = float(np.mean(f1s))
    print(f"[{a.enc}] grouped macro-F1 {macro_f1:.3f} "
          f"(+- {np.std(f1s):.3f})", flush=True)

    # the same trainer pinned to each swept layer, so the weighted sum is read
    # against single layers fitted the identical way
    fixed_f1 = {}
    for l in (a.fixed_layers or []):
        vals = []
        for tr, te in splitter.split(Z, Yc, groups):
            Ztr, Zte = standardize(tr, te)
            net = train(Ztr, Yc[tr], a.epochs, a.lr, a.wd, fixed=l)
            with torch.no_grad():
                pred = net(torch.from_numpy(Zte)).argmax(1).numpy()
            vals.append(f1_score(Yc[te], pred, average="macro"))
        fixed_f1[str(l)] = float(np.mean(vals))
        print(f"  layer {l} alone, same trainer: macro-F1 "
              f"{fixed_f1[str(l)]:.3f}", flush=True)

    # One set of weights for the geometry, fitted on the whole set the way
    # MARBLE fits a probe, then the same three measurements as everywhere else.
    idx = np.arange(len(Y))
    Zall, _ = standardize(idx, idx[:1])
    net = train(Zall, Yc, a.epochs, a.lr, a.wd)
    w = net.weights().detach().numpy()
    ZP = (Zall * w[None, :, None]).sum(1).astype(np.float64)
    print("  layer weights " + " ".join(f"{v:.3f}" for v in w), flush=True)
    print(f"  argmax layer {int(w.argmax())}, top weight {w.max():.3f}", flush=True)

    iu = np.triu_indices(len(present), 1)
    P = m9.kk_profiles()[present]
    kk = 1.0 - np.corrcoef(P)[iu]
    ton = present // 2
    dchr = m9.circ12(np.abs(ton[:, None] - ton[None, :]))[iu]
    f = (ton * 7) % 12
    dfif = m9.circ12(np.abs(f[:, None] - f[None, :]))[iu]
    dmode = np.abs((present % 2)[:, None] - (present % 2)[None, :])[iu].astype(float)

    C = np.stack([ZP[Yc == i].mean(0) for i in range(len(present))])
    dz = squareform(pdist(C))[iu]
    W = net.head.weight.detach().numpy().astype(np.float64)
    Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
    dw = (1.0 - Wn @ Wn.T)[iu]

    rng = np.random.default_rng(0)

    def geometry(dtarget, label):
        res = {}
        for nm, dd in [("kk_key_distance", kk), ("fifths", dfif),
                       ("chromatic", dchr), ("mode_mismatch", dmode)]:
            obs = float(spearmanr(dd, dtarget).statistic)
            null = np.empty(a.n_perm)
            for i in range(a.n_perm):
                perm = rng.permutation(len(present))
                P_ = squareform(squareform(dtarget, checks=False)[perm][:, perm],
                                checks=False)
                null[i] = spearmanr(dd, P_).statistic
            res[nm] = dict(rho=obs, null_mean=float(null.mean()),
                           null_std=float(null.std()),
                           z=float((obs - null.mean()) / (null.std() + 1e-12)),
                           p=float((np.abs(null) >= abs(obs)).mean()))
            print(f"  [{label}] {nm:<18} rho={obs:+.3f} z={res[nm]['z']:+5.1f} "
                  f"p={res[nm]['p']:.3f}", flush=True)
        return res

    arm = f"{a.enc}_wsum"
    tag = "" if a.dataset == "gs" else f"_{a.dataset}"
    rec = {"probe": {"grouped_macro_f1": macro_f1,
                     "fold_macro_f1": [float(v) for v in f1s],
                     "fixed_layer_macro_f1": fixed_f1},
           "layer_weights": [float(v) for v in w],
           "centroid_geometry": geometry(dz, "centroid"),
           "probe_weight_geometry": geometry(dw, "probe-W ")}
    Path(f"runs/clipkey_{arm}{tag}.json").write_text(json.dumps(
        {"arm": arm, "dataset": a.dataset, "n_clips": int(len(Y)),
         "dim": int(D), "n_keys_present": int(len(present)),
         "poolings": {"mean": rec}}, indent=2))
    np.savez_compressed(f"runs/clipkey_{arm}{tag}.npz",
                        Z=ZP.astype(np.float32), Y=Y, keys=present)
    print(f"wrote runs/clipkey_{arm}{tag}.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", required=True, choices=list(ENCODERS))
    ap.add_argument("--stage", required=True, choices=["extract", "fit"])
    ap.add_argument("--dataset", default="gs", choices=["gs", "fmak"])
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--fixed-layers", type=int, nargs="*", default=None,
                    help="layer indices to also fit alone, as a like-for-like "
                         "baseline for the weighted sum")
    a = ap.parse_args()
    (extract if a.stage == "extract" else fit)(a)
