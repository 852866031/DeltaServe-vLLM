#!/usr/bin/env python3
"""Overview figure: DeltaServe in one picture (the README shows it before the
detailed architecture diagram).

One spec, two outputs:
  overview.drawio      editable source (open in draw.io / diagrams.net)
  overview.svg/png     rendered with matplotlib, embedded in the README

    python docs/figures/make_overview.py
"""

from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent

BW, BH = 240, 84
COL0, CSTEP = 30, 350          # 110 px gutters between columns
ROW0, RSTEP = 80, 170          # 86 px gutter between rows
W = COL0 + 3 * CSTEP + BW + 30
LEGEND_Y = ROW0 + RSTEP + BH + 40
H = LEGEND_Y + 50
TITLE = "DeltaServe in one picture: inference and LoRA finetuning share the GPU"

KIND = {
    "stock":  ("#f5f5f5", "#666666"),
    "plugin": ("#dae8fc", "#6c8ebf"),
    "data":   ("#d5e8d4", "#82b366"),
}
INK, SUB, EDGE = "#1f2328", "#444c56", "#4d5866"
PX = 0.72  # px → pt at dpi 100

NODES = {}
EDGES = []


def node(nid, c, r, kind, title, sub):
    NODES[nid] = dict(x=COL0 + c * CSTEP, y=ROW0 + r * RSTEP, w=BW, h=BH,
                      kind=kind, title=title, sub=sub)


def edge(s, t, exit, entry, label="", dashed=False):
    EDGES.append(dict(s=s, t=t, exit=exit, entry=entry, label=label, dashed=dashed))


node("req", 0, 0, "stock", "Inference requests", "OpenAI-compatible API")
node("data", 0, 1, "data", "Finetuning dataset", "SFT samples")
node("sched", 1, 0, "plugin", "SLO-aware scheduler",
     "admits FT samples only while\nTTFT / TBT stay within the SLO")
node("fwd", 2, 0, "stock", "vLLM forward, one batch",
     "inference tokens + FT samples\nmulti-LoRA · CUDA graphs")
node("resp", 3, 0, "stock", "Responses", "unchanged by co-serving")
node("act", 3, 1, "data", "Saved FT activations", "shared GPU buffers")
node("bwd", 2, 1, "plugin", "Backward process",
     "trains the LoRA adapter;\npauses whenever inference\nneeds the GPU")

R, L_, T, B = (1, .5), (0, .5), (.5, 0), (.5, 1)
edge("req", "sched", R, L_, "requests")
edge("data", "sched", R, (0, .8), "FT samples")
edge("sched", "fwd", R, L_, "mixed batch")
edge("fwd", "resp", R, L_, "tokens")
edge("fwd", "act", (.8, 1), T, "FT rows", dashed=True)
edge("act", "bwd", L_, R, "read", dashed=True)
edge("bwd", "fwd", (.3, 0), (.3, 1), "updated LoRA")


def pt(n, f):
    b = NODES[n]
    return b["x"] + f[0] * b["w"], b["y"] + f[1] * b["h"]


def route(e):
    (x0, y0), (x1, y1) = pt(e["s"], e["exit"]), pt(e["t"], e["entry"])
    horiz = e["exit"][0] in (0, 1)
    if (horiz and abs(y0 - y1) < .5) or (not horiz and abs(x0 - x1) < .5):
        return [(x0, y0), (x1, y1)]
    if horiz:
        xm = (x0 + x1) / 2
        return [(x0, y0), (xm, y0), (xm, y1), (x1, y1)]
    ym = (y0 + y1) / 2
    return [(x0, y0), (x0, ym), (x1, ym), (x1, y1)]


def render():
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.text(W / 2, 36, TITLE, ha="center", va="center", fontsize=20 * PX,
            fontweight="bold", color=INK)
    for b in NODES.values():
        fc, ec = KIND[b["kind"]]
        ax.add_patch(FancyBboxPatch((b["x"], b["y"]), b["w"], b["h"],
                                    boxstyle="round,pad=0,rounding_size=10",
                                    fc=fc, ec=ec, lw=1.6, zorder=3))
        lines = b["sub"].split("\n")
        ty = b["y"] + b["h"] / 2 - 9 * len(lines)
        ax.text(b["x"] + b["w"] / 2, ty, b["title"], ha="center", va="center",
                fontsize=16 * PX, fontweight="bold", color=INK, zorder=4)
        ax.text(b["x"] + b["w"] / 2, ty + 13 + 9 * len(lines), b["sub"],
                ha="center", va="center", fontsize=12.5 * PX, color=SUB,
                linespacing=1.3, zorder=4)
    for e in EDGES:
        p = route(e)
        ls = (0, (5, 3)) if e["dashed"] else "-"
        for a, b in zip(p[:-2], p[1:-1]):
            ax.plot([a[0], b[0]], [a[1], b[1]], color=EDGE, lw=1.5, ls=ls, zorder=2.5)
        ax.add_patch(FancyArrowPatch(p[-2], p[-1], arrowstyle="-|>", mutation_scale=14,
                                     color=EDGE, lw=1.5, ls=ls, shrinkA=0, shrinkB=0,
                                     zorder=2.5))
        segs = list(zip(p[:-1], p[1:]))
        a, b = max(segs, key=lambda s: abs(s[0][0] - s[1][0]) + abs(s[0][1] - s[1][1]))
        # a straight horizontal edge carries its label above the line, so the
        # arrow stays visible; elsewhere the label sits on the line
        above = len(p) == 2 and abs(a[1] - b[1]) < .5
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 - (13 if above else 0),
                e["label"], ha="center", va="center", fontsize=13 * PX,
                color=EDGE, zorder=5,
                bbox=None if above else dict(boxstyle="round,pad=0.2", fc="white",
                                             ec="none"))
    x = COL0
    for k, txt in (("stock", "stock vLLM"), ("plugin", "DeltaServe"),
                   ("data", "data")):
        fc, ec = KIND[k]
        ax.add_patch(FancyBboxPatch((x, LEGEND_Y), 34, 20,
                                    boxstyle="round,pad=0,rounding_size=4",
                                    fc=fc, ec=ec, lw=1.2))
        ax.text(x + 44, LEGEND_Y + 10, txt, va="center", fontsize=13 * PX, color=INK)
        x += 70 + 8 * len(txt)
    ax.plot([x, x + 40], [LEGEND_Y + 10] * 2, color=EDGE, lw=1.5, ls=(0, (5, 3)))
    ax.text(x + 50, LEGEND_Y + 10, "finetuning data path", va="center",
            fontsize=13 * PX, color=INK)
    fig.savefig(OUT / "overview.svg")
    fig.savefig(OUT / "overview.png", dpi=150)
    plt.close(fig)


def drawio():
    cells = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>']
    cells.append(f'<mxCell id="title" value="{escape(TITLE)}" style="text;html=1;'
                 f'align=center;verticalAlign=middle;fontSize=20;fontStyle=1;" vertex="1" '
                 f'parent="1"><mxGeometry x="0" y="16" width="{W}" height="40" '
                 f'as="geometry"/></mxCell>')
    for nid, b in NODES.items():
        fc, ec = KIND[b["kind"]]
        sub = "<br>".join(escape(s) for s in b["sub"].split("\n"))
        value = (f"<b>{escape(b['title'])}</b><br>"
                 f"<font style=\"font-size: 12px;\" color=\"{SUB}\">{sub}</font>")
        cells.append(f'<mxCell id="{nid}" value="{escape(value)}" style="rounded=1;'
                     f'arcSize=12;whiteSpace=wrap;html=1;fontSize=16;fillColor={fc};'
                     f'strokeColor={ec};fontColor={INK};" vertex="1" parent="1">'
                     f'<mxGeometry x="{b["x"]}" y="{b["y"]}" width="{b["w"]}" '
                     f'height="{b["h"]}" as="geometry"/></mxCell>')
    for i, e in enumerate(EDGES):
        style = (f"edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;fontSize=13;"
                 f"strokeColor={EDGE};fontColor={EDGE};strokeWidth=1.5;"
                 f"exitX={e['exit'][0]};exitY={e['exit'][1]};exitDx=0;exitDy=0;"
                 f"entryX={e['entry'][0]};entryY={e['entry'][1]};entryDx=0;entryDy=0;"
                 f"labelBackgroundColor=default;defaultLabelBackgroundColor=page;")
        if e["dashed"]:
            style += "dashed=1;dashPattern=5 3;"
        cells.append(f'<mxCell id="e{i}" value="{escape(e["label"])}" style="{style}" '
                     f'edge="1" parent="1" source="{e["s"]}" target="{e["t"]}">'
                     f'<mxGeometry relative="1" as="geometry"/></mxCell>')
    xml = (f'<mxfile host="deltaserve"><diagram name="overview" id="overview">'
           f'<mxGraphModel adaptiveColors="auto" grid="1" gridSize="10" page="0" '
           f'pageWidth="{W}" pageHeight="{H}"><root>' + "".join(cells)
           + "</root></mxGraphModel></diagram></mxfile>")
    (OUT / "overview.drawio").write_text(xml)


if __name__ == "__main__":
    render()
    drawio()
    print(f"wrote {OUT / 'overview.png'}, .svg, .drawio")
