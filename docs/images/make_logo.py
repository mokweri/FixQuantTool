#!/usr/bin/env python3
"""Generate the FixQuant logo mark in light and dark variants.

    python3 docs/images/make_logo.py

Writes fixquant-logo-light.svg and fixquant-logo-dark.svg next to this script.
The mark is an "FQ" monogram inside a chip outline: the chip stands for the
FPGA target, and the Q's tail is a staircase, a continuous stroke becoming
discrete fixed-point steps. Letters are drawn as geometry rather than text, so
the mark renders identically everywhere and needs no font.

Everything is built on a 256-unit canvas and must stay legible at 32 px, so
strokes are heavy and the tail keeps a clear gap from both the ring and the
chip wall. Both variants share this geometry; edit it here, not in the SVGs.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent

THEMES = {
    # Blue letters, orange tail: the pair passes the colour-blind separation,
    # lightness and contrast checks against each theme's page background.
    'light': dict(chip='#1F2328', letter='#2A62A8', tail='#C2701A'),
    'dark': dict(chip='#E6EDF3', letter='#5A94E0', tail='#CF7B2B'),
}

S = 256                      # canvas
BODY = (36, 220)             # chip body extent (square), stroke-centred
BODY_STROKE = 12
PIN_LEN, PIN_STROKE = 18, 11
PINS = (92, 128, 164)        # pin positions along each side

LETTER = 17                  # letter stroke weight
F_X, F_TOP, F_BOT = 52, 86, 158  # F stem box
F_TOP_BAR, F_MID_BAR = 50, 41    # bar lengths; they stop clear of the ring
Q_CX, Q_CY, Q_R = 152, 120, 32   # ring centre-line radius
TAIL_STEP, TAIL_W = 12, 11       # staircase tail: three steps


def mark(t):
    lo, hi = BODY
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {S} {S}" '
             f'width="{S}" height="{S}" role="img" aria-label="FixQuant logo">']

    # Chip body and pins.
    parts.append(f'<rect x="{lo}" y="{lo}" width="{hi-lo}" height="{hi-lo}" rx="22" '
                 f'fill="none" stroke="{t["chip"]}" stroke-width="{BODY_STROKE}"/>')
    edge = BODY_STROKE / 2
    pin = f'stroke="{t["chip"]}" stroke-width="{PIN_STROKE}" stroke-linecap="round"'
    for p in PINS:
        parts.append(f'<line x1="{p}" y1="{lo-edge}" x2="{p}" y2="{lo-edge-PIN_LEN}" {pin}/>')
        parts.append(f'<line x1="{p}" y1="{hi+edge}" x2="{p}" y2="{hi+edge+PIN_LEN}" {pin}/>')
        parts.append(f'<line x1="{lo-edge}" y1="{p}" x2="{lo-edge-PIN_LEN}" y2="{p}" {pin}/>')
        parts.append(f'<line x1="{hi+edge}" y1="{p}" x2="{hi+edge+PIN_LEN}" y2="{p}" {pin}/>')

    # F: stem, top bar, and a shorter middle bar, as plain rectangles.
    w = LETTER
    mid = F_TOP + round((F_BOT - F_TOP) * 0.44)
    for x, y, bw, bh in ((F_X, F_TOP, w, F_BOT - F_TOP),
                         (F_X, F_TOP, F_TOP_BAR, w),
                         (F_X, mid, F_MID_BAR, w - 1)):
        parts.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" '
                     f'fill="{t["letter"]}"/>')

    # Q ring.
    parts.append(f'<circle cx="{Q_CX}" cy="{Q_CY}" r="{Q_R}" fill="none" '
                 f'stroke="{t["letter"]}" stroke-width="{w}"/>')

    # Q tail: a staircase that starts inside the ring and crosses its lower
    # right, as a Q's tail does. It is drawn over the ring in its own colour,
    # so the quantization steps stand apart from the letters without a
    # background-coloured knockout, which would break on a dark page. Crossing
    # the ring is also what stops the mark reading as a magnifying glass.
    x0, y0 = Q_CX + 6, Q_CY + 10
    tail = f'M{x0},{y0}' + f' h{TAIL_STEP} v{TAIL_STEP}' * 3 + ' h8'
    parts.append(f'<path d="{tail}" fill="none" stroke="{t["tail"]}" '
                 f'stroke-width="{TAIL_W}" stroke-linecap="butt" stroke-linejoin="miter"/>')

    parts.append('</svg>')
    return '\n'.join(parts)


for name, theme in THEMES.items():
    path = HERE / f'fixquant-logo-{name}.svg'
    path.write_text(mark(theme) + '\n')
    print(f'wrote {path.relative_to(HERE.parent.parent)}')
