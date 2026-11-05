#!/usr/bin/env python3
"""Architecture figure: how the DeltaServe plugin sits on stock vLLM.

One spec, two outputs, so they never drift apart:
  architecture.drawio   editable source (open in draw.io / diagrams.net)
  architecture.svg/png  rendered with matplotlib, embedded in the README

    python docs/figures/make_architecture.py
"""

from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent

# ── grid ────────────────────────────────────────────────────────────────
BW, BH = 230, 70            # box size
COL0, CSTEP = 250, 300      # first column x, column pitch (70 px gutters)
LANE_X = COL0 - 20
LANE_W = 20 + 4 * CSTEP + BW + 20
R1, R2 = 40, 150            # row offsets inside a two-row lane
LEFT_X, LEFT_W = 20, 190
W = LANE_X + LANE_W + 20


def col(i):
    return COL0 + i * CSTEP


# kind → (fill, stroke)
KIND = {
    "stock":  ("#f5f5f5", "#666666"),
    "plugin": ("#dae8fc", "#6c8ebf"),
    "data":   ("#d5e8d4", "#82b366"),
    "signal": ("#ffe6cc", "#d79b00"),
    "launch": ("#fff2cc", "#d6b656"),
}
LANE_FILL, LANE_STROKE = "#fafafa", "#b3b3b3"
INK, SUB, EDGE = "#1f2328", "#444c56", "#4d5866"

# ── lanes (x, y, w, h, title) ───────────────────────────────────────────
LANES = {}
NODES = {}   # id → dict(x, y, w, h, title, sub, kind, lane)
EDGES = []   # dict(s, t, exit, entry, label, dashed, inherit, lpos)


def lane(lid, y, h, title):
    LANES[lid] = dict(x=LANE_X, y=y, w=LANE_W, h=h, title=title)


def node(nid, lid, c, r, kind, title, sub, x=None, w=BW, h=BH):
    if lid:
        L = LANES[lid]
        x = col(c) if x is None else x
        y = L["y"] + (R1 if r == 1 else R2)
    else:
        y = r
    NODES[nid] = dict(x=x, y=y, w=w, h=h, title=title, sub=sub, kind=kind, lane=lid)


def edge(s, t, exit, entry, label="", dashed=False, inherit=False, lpos="c"):
    EDGES.append(dict(s=s, t=t, exit=exit, entry=entry, label=label,
                      dashed=dashed, inherit=inherit, lpos=lpos))


# rows of lanes
lane("api", 60, 130, "API server process   (vLLM frontend)")
lane("eng", 215, 240, "EngineCore process   (scheduler + executor driver)")
lane("wrk", 480, 240, "Worker process   (one per TP rank · owns the GPU)")
lane("shm", 745, 130, "Shared between the worker and its backward child   (CUDA IPC / shared memory)")
lane("bwd", 900, 240, "Backward child   (one per TP rank · own CUDA context · spawned by the worker)")

node("a1", "api", 0, 1, "stock", "OpenAI-compatible server", "vllm serve · FastAPI")
node("a3", "api", 1, 1, "stock", "AsyncLLM", "frontend · ZMQ to EngineCore")
node("a2", "api", 2, 1, "plugin", "deltaserve/api.py",
     "/start_finetuning · /stop_finetuning\n/save_estimator_state")

node("b2", "eng", 0, 1, "plugin", "EngineHooks",
     "arrival counter · profiling pass\nestimator save · input-queue swap")
node("b1", "eng", 1, 1, "stock", "EngineCoreProc", "busy loop · input-socket thread")
node("b5", "eng", 2, 1, "plugin", "FinetuningStore",
     "+ coordinator: claim → commit /\nrelease · buffer-full · ack flags")
node("b6", "eng", 0, 2, "stock", "vLLM Scheduler", "token budget · paged KV\ncontinuous batching")
node("b3", "eng", 1, 2, "plugin", "FinetuneScheduler",
     "scheduler_cls · FT injection\nthrottles · interrupt tiers A / B")
node("b4", "eng", 2, 2, "plugin", "SLO estimator",
     "5-regime step-time model\nlaunch profiling · online refit")

node("c1", "wrk", 0, 1, "plugin", "DeltaServeWorker",
     "worker_cls · spawns the child\nexports weights + buffers")
node("c2", "wrk", 1, 1, "plugin", "DeltaServeModelRunner",
     "FT mask · timing ring · relay\ntier-C abort · FT retire")
node("c3", "wrk", 2, 1, "plugin", "FtCudagraphDispatcher",
     "has_ft piecewise graph set\nabort poll between segments")
node("c7", "wrk", 3, 1, "plugin", "FtAbortPoller",
     "per-layer arrival check\nTP: gloo MAX vote")
node("c4", "wrk", 1, 2, "stock", "Model forward",
     "Llama-3 / Qwen3 · punica\nmulti-LoRA · CUDA graphs")
node("c5", "wrk", 2, 2, "plugin", "Accumulator hooks",
     "per-layer FT-row saves\ntorch.ops.vllm.dserve_save_rows")

node("s0", "shm", 0, 1, "signal", "Pause grant", "mp.Event · held until\nthe prefill is done")
node("s1", "shm", 1, 1, "data", "Base + FT-LoRA weights", "zero-copy CUDA IPC")
node("s2", "shm", 2, 1, "data", "Activation buffers",
     "layer_in · gate_up · qkv · ctx\nresid_mid · final hidden")
node("s3", "shm", 3, 1, "data", "Served LoRA slot", "trainer publishes\npunica reads")
node("s4", "shm", 4, 1, "signal", "FtArrivalSignal",
     "shm counter, bumped by the\nEngineCore input thread")

node("d1", "bwd", 0, 1, "plugin", "BackwardService loop",
     "_maybe_pause at every boundary\nrun-ahead ring · TP pause vote")
node("d2", "bwd", 1, 1, "plugin", "LoraSftTrainerService",
     "family file: llama3.py / qwen3.py\nfp32 masters")
node("d3", "bwd", 2, 1, "plugin", "Layer backward × L",
     "3 CUDA graphs per layer\neager fallback")
node("d4", "bwd", 3, 1, "plugin", "Fused AdamW + publish",
     "one kernel for all LoRA tensors\nα/r baked into B")
node("d5", "bwd", 1, 2, "plugin", "LM-head backward",
     "batched rows · chunked fp32\neager · pause boundaries")
node("d6", "bwd", 2, 2, "plugin", "TP collectives",
     "NCCL residual reduces\nbucketed factor grads")

# left column: client + launch path
node("u", None, 0, 100, "stock", "Clients", "HTTP · eval replays", x=LEFT_X, w=LEFT_W)
LAUNCH = dict(x=LEFT_X - 10, y=215, w=LEFT_W + 20, h=505, title="Launch path")
node("l1", None, 0, 255, "launch", "configs/*.yaml",
     "finetune · throttles\nslo · debug", x=LEFT_X, w=LEFT_W)
node("l2", None, 0, 375, "launch", "config_loader",
     "YAML → CLI args", x=LEFT_X, w=LEFT_W)
node("l3", None, 0, 495, "launch", "--additional-config",
     '{"deltaserve": {...}}', x=LEFT_X, w=LEFT_W)
node("l4", None, 0, 615, "launch", "plugin.register()",
     "entry point, runs in\nevery vLLM process", x=LEFT_X, w=LEFT_W, h=BH + 10)

# edges: exit / entry are (fx, fy) on the box (0..1)
R, L_, T, B = (1, .5), (0, .5), (.5, 0), (.5, 1)
edge("u", "a1", R, L_)
edge("a1", "a3", R, L_, "submit")
edge("a2", "a3", L_, R, "RPC")
edge("a3", "b1", B, T, "ZMQ")
edge("b2", "b1", R, L_, "wraps")
edge("b1", "b3", B, T, "schedule()")
edge("b3", "b6", L_, R, "extends", inherit=True)
edge("b3", "b4", R, L_, "predict")
edge("b3", "b5", (1, .25), (0, .75), "claim")
edge("b3", "c2", (.3, 1), (.3, 0), "SchedulerOutput + FT fields", lpos="l")
edge("c2", "b3", (.7, 0), (.7, 1), "ModelRunnerOutput: timing · ack · abort", lpos="r")
edge("c1", "c2", R, L_, "adopts")
edge("c2", "c3", R, L_, "dispatch")
edge("c2", "c4", B, T, "forward")
edge("c4", "c5", R, L_, "hooks")
edge("c5", "c7", (1, .25), (0, .75), "poll", dashed=True)
edge("c1", "s0", B, T, "grant / revoke", dashed=True)
edge("c4", "s1", B, T, "reads", dashed=True)
edge("c5", "s2", B, T, "writes FT rows", dashed=True)
edge("c7", "s4", B, T, "reads", dashed=True)
edge("d1", "s0", T, B, "waits", dashed=True)
edge("d2", "s1", T, B, "reads", dashed=True)
edge("d3", "s2", T, B, "reads", dashed=True)
edge("d4", "s3", T, B, "publish", dashed=True)
edge("d1", "d2", R, L_, "trigger")
edge("d2", "d5", B, T, "1 · head")
edge("d2", "d3", R, L_, "2 · layers")
edge("d3", "d4", R, L_, "3 · step")
edge("d3", "d6", B, T, "all-reduce")
edge("l1", "l2", B, T)
edge("l2", "l3", B, T)
edge("l3", "l4", B, T)

LEGEND_Y = 1165
H = LEGEND_Y + 60
TITLE = "DeltaServe on stock vLLM: what the plugin adds to each process"


# ── geometry ────────────────────────────────────────────────────────────
def pt(n, f):
    b = NODES[n]
    return b["x"] + f[0] * b["w"], b["y"] + f[1] * b["h"]


def route(e):
    """Orthogonal polyline from the exit point to the entry point."""
    (x0, y0), (x1, y1) = pt(e["s"], e["exit"]), pt(e["t"], e["entry"])
    horiz = e["exit"][0] in (0, 1)
    if (horiz and abs(y0 - y1) < .5) or (not horiz and abs(x0 - x1) < .5):
        return [(x0, y0), (x1, y1)]
    if horiz:
        xm = (x0 + x1) / 2
        return [(x0, y0), (xm, y0), (xm, y1), (x1, y1)]
    ym = (y0 + y1) / 2
    return [(x0, y0), (x0, ym), (x1, ym), (x1, y1)]


# ── matplotlib render ───────────────────────────────────────────────────
PX = 0.72  # px → pt at dpi 100


def render():
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(W / 2, 30, TITLE, ha="center", va="center", fontsize=20 * PX,
            fontweight="bold", color=INK)

    def container(c, bold_title):
        ax.add_patch(FancyBboxPatch((c["x"], c["y"]), c["w"], c["h"],
                                    boxstyle="round,pad=0,rounding_size=6",
                                    fc=LANE_FILL, ec=LANE_STROKE, lw=1.2, zorder=1))
        ax.text(c["x"] + 14, c["y"] + 20, bold_title, ha="left", va="center",
                fontsize=14 * PX, fontweight="bold", color=SUB, zorder=2)

    for c in LANES.values():
        container(c, c["title"])
    container(LAUNCH, LAUNCH["title"])

    for b in NODES.values():
        fc, ec = KIND[b["kind"]]
        ax.add_patch(FancyBboxPatch((b["x"], b["y"]), b["w"], b["h"],
                                    boxstyle="round,pad=0,rounding_size=8",
                                    fc=fc, ec=ec, lw=1.4, zorder=3))
        lines = b["sub"].split("\n")
        cy = b["y"] + b["h"] / 2
        ty = cy - 8 * len(lines)
        ax.text(b["x"] + b["w"] / 2, ty, b["title"], ha="center", va="center",
                fontsize=14 * PX, fontweight="bold", color=INK, zorder=4)
        ax.text(b["x"] + b["w"] / 2, ty + 11 + 8 * len(lines), b["sub"],
                ha="center", va="center", fontsize=11 * PX, color=SUB,
                linespacing=1.25, zorder=4)

    for e in EDGES:
        p = route(e)
        ls = (0, (5, 3)) if e["dashed"] else "-"
        for a, b in zip(p[:-2], p[1:-1]):
            ax.plot([a[0], b[0]], [a[1], b[1]], color=EDGE, lw=1.3, ls=ls,
                    solid_capstyle="butt", zorder=2.5)
        style = "-|>" if not e["inherit"] else "-|>"
        ax.add_patch(FancyArrowPatch(p[-2], p[-1], arrowstyle=style,
                                     mutation_scale=12, color=EDGE, lw=1.3, ls=ls,
                                     fc="white" if e["inherit"] else EDGE,
                                     shrinkA=0, shrinkB=0, zorder=2.5))
        if e["label"]:
            # label on the longest segment
            segs = list(zip(p[:-1], p[1:]))
            a, b = max(segs, key=lambda s: abs(s[0][0] - s[1][0]) + abs(s[0][1] - s[1][1]))
            mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
            if e["lpos"] == "l":
                ax.text(mx - 8, my, e["label"], ha="right", va="center",
                        fontsize=12 * PX, color=EDGE, zorder=5,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
            elif e["lpos"] == "r":
                ax.text(mx + 8, my, e["label"], ha="left", va="center",
                        fontsize=12 * PX, color=EDGE, zorder=5,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
            else:
                ax.text(mx, my, e["label"], ha="center", va="center",
                        fontsize=12 * PX, color=EDGE, zorder=5,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))

    # legend
    items = [("stock", "stock vLLM (unmodified)"), ("plugin", "DeltaServe plugin"),
             ("data", "shared GPU memory"), ("signal", "shared signal"),
             ("launch", "launch path")]
    x = LANE_X
    for k, txt in items:
        fc, ec = KIND[k]
        ax.add_patch(FancyBboxPatch((x, LEGEND_Y), 34, 20,
                                    boxstyle="round,pad=0,rounding_size=4",
                                    fc=fc, ec=ec, lw=1.2))
        ax.text(x + 44, LEGEND_Y + 10, txt, va="center", fontsize=13 * PX, color=INK)
        x += 60 + 7.8 * len(txt)
    for dashed, txt in ((False, "call / data flow"), (True, "shared-memory access")):
        ax.plot([x, x + 40], [LEGEND_Y + 10] * 2, color=EDGE, lw=1.3,
                ls=(0, (5, 3)) if dashed else "-")
        ax.text(x + 50, LEGEND_Y + 10, txt, va="center", fontsize=13 * PX, color=INK)
        x += 64 + 7.0 * len(txt)
    ax.add_patch(FancyArrowPatch((x, LEGEND_Y + 10), (x + 40, LEGEND_Y + 10),
                                 arrowstyle="-|>", mutation_scale=12, color=EDGE,
                                 fc="white", lw=1.3))
    ax.text(x + 50, LEGEND_Y + 10, "subclass of", va="center", fontsize=13 * PX, color=INK)

    fig.savefig(OUT / "architecture.svg")
    fig.savefig(OUT / "architecture.png", dpi=150)
    plt.close(fig)


# ── draw.io XML ─────────────────────────────────────────────────────────
def drawio():
    cells = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>']

    def v(cid, value, style, x, y, w, h, parent="1"):
        cells.append(f'<mxCell id="{cid}" value="{escape(value)}" style="{style}" '
                     f'vertex="1" parent="{parent}"><mxGeometry x="{x:g}" y="{y:g}" '
                     f'width="{w:g}" height="{h:g}" as="geometry"/></mxCell>')

    v("title", TITLE, "text;html=1;align=center;verticalAlign=middle;fontSize=20;"
      "fontStyle=1;strokeColor=none;fillColor=none;", 0, 10, W, 40)
    lane_style = (f"swimlane;startSize=34;html=1;rounded=1;arcSize=2;fontSize=14;"
                  f"align=left;spacingLeft=10;fillColor={LANE_FILL};swimlaneFillColor={LANE_FILL};"
                  f"strokeColor={LANE_STROKE};fontColor={SUB};swimlaneLine=0;")
    for lid, c in LANES.items():
        v(lid, c["title"], lane_style, c["x"], c["y"], c["w"], c["h"])
    v("launch", LAUNCH["title"], lane_style, LAUNCH["x"], LAUNCH["y"], LAUNCH["w"], LAUNCH["h"])

    for nid, b in NODES.items():
        fc, ec = KIND[b["kind"]]
        sub = "<br>".join(escape(s) for s in b["sub"].split("\n"))
        value = (f"<b>{escape(b['title'])}</b><br>"
                 f"<font style=\"font-size: 11px;\" color=\"{SUB}\">{sub}</font>")
        style = (f"rounded=1;arcSize=12;whiteSpace=wrap;html=1;fontSize=14;"
                 f"fillColor={fc};strokeColor={ec};fontColor={INK};")
        parent, x, y = "1", b["x"], b["y"]
        if b["lane"]:
            parent = b["lane"]
            x, y = x - LANES[b["lane"]]["x"], y - LANES[b["lane"]]["y"]
        elif nid.startswith("l"):
            parent = "launch"
            x, y = x - LAUNCH["x"], y - LAUNCH["y"]
        cells.append(f'<mxCell id="{nid}" value="{escape(value)}" style="{style}" '
                     f'vertex="1" parent="{parent}"><mxGeometry x="{x:g}" y="{y:g}" '
                     f'width="{b["w"]:g}" height="{b["h"]:g}" as="geometry"/></mxCell>')

    for i, e in enumerate(EDGES):
        style = (f"edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;fontSize=12;"
                 f"strokeColor={EDGE};fontColor={EDGE};strokeWidth=1.3;"
                 f"exitX={e['exit'][0]};exitY={e['exit'][1]};exitDx=0;exitDy=0;"
                 f"entryX={e['entry'][0]};entryY={e['entry'][1]};entryDx=0;entryDy=0;"
                 f"labelBackgroundColor=default;defaultLabelBackgroundColor=page;")
        if e["dashed"]:
            style += "dashed=1;dashPattern=5 3;"
        if e["inherit"]:
            style += "endArrow=block;endFill=0;endSize=10;"
        off = ""
        if e["lpos"] in ("l", "r"):
            dx = -1 if e["lpos"] == "l" else 1
            w = 6.5 * len(e["label"]) / 2 + 10
            style += "labelBackgroundColor=none;"
            off = f'<mxPoint x="{dx * w:g}" y="0" as="offset"/>'
        cells.append(f'<mxCell id="e{i}" value="{escape(e["label"])}" style="{style}" '
                     f'edge="1" parent="1" source="{e["s"]}" target="{e["t"]}">'
                     f'<mxGeometry relative="1" as="geometry">{off}</mxGeometry></mxCell>')

    # legend
    x = LANE_X
    for k, txt in (("stock", "stock vLLM (unmodified)"), ("plugin", "DeltaServe plugin"),
                   ("data", "shared GPU memory"), ("signal", "shared signal"),
                   ("launch", "launch path")):
        fc, ec = KIND[k]
        v(f"lg_{k}", "", f"rounded=1;html=1;fillColor={fc};strokeColor={ec};",
          x, LEGEND_Y, 34, 20)
        v(f"lgt_{k}", txt, "text;html=1;fontSize=13;align=left;verticalAlign=middle;",
          x + 40, LEGEND_Y - 5, 8.2 * len(txt) + 20, 30)
        x += 60 + 7.8 * len(txt)
    v("lgt_lines", "solid: call / data · dashed: shared memory · "
      "hollow arrow: subclass of",
      "text;html=1;fontSize=13;align=left;verticalAlign=middle;", x, LEGEND_Y - 5, 540, 30)

    xml = (f'<mxfile host="deltaserve"><diagram name="architecture" id="arch">'
           f'<mxGraphModel adaptiveColors="auto" dx="{W}" dy="{H}" grid="1" gridSize="10" '
           f'page="0" pageWidth="{W}" pageHeight="{H}"><root>'
           + "".join(cells) + "</root></mxGraphModel></diagram></mxfile>")
    (OUT / "architecture.drawio").write_text(xml)
    return xml


if __name__ == "__main__":
    render()
    drawio()
    print(f"wrote {OUT / 'architecture.png'}, .svg, .drawio")
