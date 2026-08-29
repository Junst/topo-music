"""One table over every arm, NSynth (pitch) and GiantSteps (key) side by side.

The two probes ask different questions and their columns must not be conflated:

  NSynth arms  labels are MIDI pitch. rho_abs is absolute pitch height, rho_chr
               is octave equivalence (does C3 sit near C4), rho_5th is a weak
               fifths test between single notes.
  GS arms      labels are tonic pitch class of a 24-way key. rho_abs is then
               |delta pitch class|, which is NOT pitch height and is not
               reported. rho_chr and rho_5th are the real quantities: whether
               codes preferring keys a semitone / a fifth apart sit near each
               other. This is the circle-of-fifths test NSynth cannot support.
"""
from __future__ import annotations

import json, sys
from pathlib import Path


def pick(path: Path):
    lv = json.load(path.open())["levels"]
    cand = [r for r in lv if "octave" in r]
    return max(cand, key=lambda r: r["nmi_pitch"]) if cand else None


def row(name, r, key_probe: bool):
    o, m = r["octave"], r["moran_pitch_codespace"]
    e = r.get("e2_spectral_control", {})
    abs_ = "     —" if key_probe else f"{o['rho_abs_pitch']:>6.3f}"
    return (f"{name:<16}{r['nmi_pitch']:>7.3f}{r['nmi_family']:>7.3f}"
            f"{m['I']:>8.3f}{m['z']:>7.1f}{abs_}"
            f"{o['rho_chroma']:>8.3f}{o['rho_chroma_partial']:>9.3f}"
            f"{o['rho_fifths']:>8.3f}{o['rho_fifths_partial']:>9.3f}"
            f"{e.get('rho_pitch_given_spec', float('nan')):>10.3f}")


def main():
    hdr = (f"{'arm':<16}{'nMI_p':>7}{'nMI_f':>7}{'MoranI':>8}{'z':>7}{'r_abs':>6}"
           f"{'r_chr':>8}{'r_chr|p':>9}{'r_5th':>8}{'r_5th|p':>9}{'r_p|spec':>10}")
    for title, prefix, names, key_probe in [
        ("NSynth — single notes (pitch height, octave equivalence)", "topo_",
         ["cqt"] + [f"mert_L{l}" for l in (0, 4, 8, 12, 16, 20, 24)]
         + ["encodec_32k", "encodec_24k", "dac_44k"], False),
        ("GiantSteps — 24 keys (circle of fifths, tonic chroma)", "topo_gs_",
         ["cqt", "mert_L4", "mert_L16", "mert_L24", "encodec_32k"], True),
    ]:
        print(f"\n### {title}\n{hdr}\n" + "-" * len(hdr))
        for n in names:
            p = Path(f"runs/{prefix}{n}.json")
            if not p.exists():
                print(f"{n:<16} (missing)")
                continue
            r = pick(p)
            print(row(n, r, key_probe) if r else f"{n:<16} (no octave block)")


if __name__ == "__main__":
    main()
