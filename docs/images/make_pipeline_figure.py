#!/usr/bin/env python3
"""Generate the README pipeline figure in light and dark variants.

    python3 docs/images/make_pipeline_figure.py

Writes fixquant-pipeline-light.svg and fixquant-pipeline-dark.svg next to this
script. The README selects between them with <picture> and the
prefers-color-scheme media query, so both variants must stay identical in
layout; edit the shared geometry here rather than either SVG.

The canvas is 1600 units wide. GitHub shows README images at roughly 880 px,
a scale of about 0.55, so the 32-unit titles display near 18 px and the
24-unit captions near 13 px.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent

THEMES = {
    'light': dict(card='#FFFFFF', card_edge='#D0D7DE', ink='#1F2328',
                  muted='#59636E', accent='#2A62A8', accent_fill='#EEF4FB',
                  badge_ink='#FFFFFF', arrow='#59636E'),
    'dark': dict(card='#161B22', card_edge='#30363D', ink='#E6EDF3',
                 muted='#9198A1', accent='#5A94E0', accent_fill='#12243A',
                 badge_ink='#0D1117', arrow='#9198A1'),
}

STAGES = [
    ('FP32', 'model', 'Pretrained', 'PyTorch CNN', 'network'),
    ('Quantization-', 'aware training', 'TQT, 8-bit', 'fixed point', 'steps'),
    ('Integer', 'digital twin', 'Bit-exact with', 'the FPGA', 'twin'),
    ('Versioned', 'release', 'Checksummed,', 'reproducible', 'release'),
    ('Deployment', 'package', 'graph.json,', 'params, refs', 'package'),
]
HIGHLIGHT = 2          # the integer digital twin carries the key message

W, H = 1600, 540
CARD_W, GAP, X0 = 256, 56, 48
TOP, BOTTOM = 40, 420
ICON_Y, TITLE_Y, CAP_Y = 140, 250, 330
FONT = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"


def icon(kind, cx, cy, c):
    """Line icons drawn in an 84-unit box centred on (cx, cy)."""
    s = f'fill="none" stroke="{c}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"'
    if kind == 'network':
        cols = [(-30, [-24, 0, 24]), (0, [-30, -10, 10, 30]), (30, [-12, 12])]
        out = []
        for (x1, ys1), (x2, ys2) in zip(cols, cols[1:]):
            for y1 in ys1:
                for y2 in ys2:
                    out.append(f'<line x1="{cx+x1}" y1="{cy+y1}" x2="{cx+x2}" '
                               f'y2="{cy+y2}" stroke="{c}" stroke-width="1.6" opacity="0.55"/>')
        for x, ys in cols:
            for y in ys:
                out.append(f'<circle cx="{cx+x}" cy="{cy+y}" r="6.5" fill="{c}"/>')
        return ''.join(out)
    if kind == 'steps':
        # A quantizer: the smooth identity (dashed) and its staircase.
        return (f'<line x1="{cx-38}" y1="{cy+34}" x2="{cx+38}" y2="{cy-34}" '
                f'stroke="{c}" stroke-width="2" stroke-dasharray="4 5" opacity="0.6"/>'
                f'<path d="M{cx-38},{cy+34} h19 v-17 h19 v-17 h19 v-17 h19 v-17" {s}/>')
    if kind == 'twin':
        # Two identical chips joined by an equals sign.
        out = []
        for dx in (-30, 30):
            x = cx + dx
            out.append(f'<rect x="{x-17}" y="{cy-17}" width="34" height="34" rx="4" {s}/>')
            for k in (-8, 0, 8):
                out.append(f'<line x1="{x+k}" y1="{cy-17}" x2="{x+k}" y2="{cy-25}" {s}/>'
                           f'<line x1="{x+k}" y1="{cy+17}" x2="{x+k}" y2="{cy+25}" {s}/>')
        out.append(f'<line x1="{cx-6}" y1="{cy-5}" x2="{cx+6}" y2="{cy-5}" {s}/>'
                   f'<line x1="{cx-6}" y1="{cy+5}" x2="{cx+6}" y2="{cy+5}" {s}/>')
        return ''.join(out)
    if kind == 'release':
        # A version tag.
        return (f'<path d="M{cx-34},{cy-24} h42 l26,24 l-26,24 h-42 z" {s}/>'
                f'<circle cx="{cx-20}" cy="{cy}" r="5" fill="{c}"/>'
                f'<text x="{cx+2}" y="{cy+7}" font-size="19" font-weight="700" '
                f'fill="{c}" text-anchor="middle">v1</text>')
    if kind == 'package':
        return (f'<path d="M{cx},{cy-34} l34,17 v36 l-34,17 l-34,-17 v-36 z" {s}/>'
                f'<path d="M{cx-34},{cy-17} l34,17 l34,-17 M{cx},{cy} v36" {s}/>')
    raise ValueError(kind)


def figure(t):
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
           f'width="{W}" height="{H}" font-family="{FONT}" role="img" '
           f'aria-label="FixQuant pipeline: FP32 model, quantization-aware '
           f'training, integer digital twin bit-exact with the FPGA, versioned '
           f'release, deployment package">']
    out.append(f'<defs><marker id="head" viewBox="0 0 10 10" refX="8" refY="5" '
               f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
               f'<path d="M0,0 L10,5 L0,10 z" fill="{t["arrow"]}"/></marker>'
               f'<marker id="gate" viewBox="0 0 10 10" refX="8" refY="5" '
               f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
               f'<path d="M0,0 L10,5 L0,10 z" fill="{t["accent"]}"/></marker></defs>')

    centres = []
    for i, (t1, t2, c1, c2, kind) in enumerate(STAGES):
        x = X0 + i * (CARD_W + GAP)
        cx = x + CARD_W / 2
        centres.append(cx)
        hot = i == HIGHLIGHT
        out.append(f'<rect x="{x}" y="{TOP}" width="{CARD_W}" height="{BOTTOM-TOP}" '
                   f'rx="18" fill="{t["accent_fill"] if hot else t["card"]}" '
                   f'stroke="{t["accent"] if hot else t["card_edge"]}" '
                   f'stroke-width="{3 if hot else 2}"/>')
        out.append(f'<circle cx="{x+34}" cy="{TOP+34}" r="17" fill="{t["accent"]}"/>'
                   f'<text x="{x+34}" y="{TOP+42}" font-size="21" font-weight="700" '
                   f'fill="{t["badge_ink"]}" text-anchor="middle">{i+1}</text>')
        out.append(icon(kind, cx, ICON_Y, t['accent']))
        for k, line in enumerate((t1, t2)):
            out.append(f'<text x="{cx}" y="{TITLE_Y + 40*k}" font-size="32" '
                       f'font-weight="700" fill="{t["ink"]}" text-anchor="middle">{line}</text>')
        for k, line in enumerate((c1, c2)):
            out.append(f'<text x="{cx}" y="{CAP_Y + 32*k}" font-size="24" '
                       f'fill="{t["accent"] if hot else t["muted"]}" '
                       f'font-weight="{600 if hot else 400}" '
                       f'text-anchor="middle">{line}</text>')

    # Forward arrows between cards, ending just short of the next card's edge.
    ay = (TOP + BOTTOM) / 2
    for i in range(len(STAGES) - 1):
        x1 = X0 + i * (CARD_W + GAP) + CARD_W + 8
        x2 = x1 + GAP - 16
        out.append(f'<line x1="{x1}" y1="{ay}" x2="{x2}" y2="{ay}" '
                   f'stroke="{t["arrow"]}" stroke-width="3.5" marker-end="url(#head)"/>')

    # Accuracy gate: from the digital twin back to QAT, below the cards.
    gy = BOTTOM + 52
    tx, qx = centres[HIGHLIGHT], centres[1]
    out.append(f'<path d="M{tx},{BOTTOM+6} V{gy} H{qx} V{BOTTOM+12}" fill="none" '
               f'stroke="{t["accent"]}" stroke-width="3" stroke-dasharray="9 7" '
               f'marker-end="url(#gate)"/>')
    out.append(f'<text x="{(tx+qx)/2}" y="{gy+36}" font-size="24" '
               f'fill="{t["accent"]}" text-anchor="middle">accuracy gate</text>')
    out.append('</svg>')
    return '\n'.join(out)


for name, theme in THEMES.items():
    path = HERE / f'fixquant-pipeline-{name}.svg'
    path.write_text(figure(theme) + '\n')
    print(f'wrote {path.relative_to(HERE.parent.parent)}')
