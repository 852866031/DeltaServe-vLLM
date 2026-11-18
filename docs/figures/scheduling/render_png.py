#!/usr/bin/env python
"""Render the scheduling .drawio diagrams in this directory to PNG.

A small renderer for the subset of draw.io these files use (rectangles,
rounded rectangles, rhombi, text, straight and right-angle connectors, HTML
labels with <b>, <br> and font sizes). It needs only Pillow and fontTools, so
the figures can be regenerated without draw.io:

    python docs/figures/scheduling/render_png.py

draw.io itself stays the editor of record; open the .drawio files there to
change a figure, then rerun this script.
"""

import html
import math
import re
import sys
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path

from fontTools.ttLib import TTFont
from PIL import Image, ImageChops, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
SS = 4          # supersampling while drawing
OUT_SCALE = 2   # pixels per draw.io unit in the PNG
MARGIN = 24          # canvas slack while drawing; the PNG is cropped to its content
PAD = 10             # white border left around the content, in draw.io units
FONT_DIRS = ("/usr/share/fonts/truetype/liberation", "/usr/share/fonts/truetype/dejavu")
FACES = {False: ("LiberationSans-Regular.ttf", "DejaVuSans.ttf"),
         True: ("LiberationSans-Bold.ttf", "DejaVuSans-Bold.ttf")}


def _find(name):
    for d in FONT_DIRS:
        p = Path(d) / name
        if p.exists():
            return str(p)
    raise FileNotFoundError(name)


class Fonts:
    """Liberation Sans, falling back to DejaVu Sans per character (circled
    digits, arrows and box-drawing characters are only in the latter)."""

    def __init__(self):
        self.paths = {b: [_find(n) for n in names] for b, names in FACES.items()}
        self.cmaps = {b: [set(TTFont(p).getBestCmap()) for p in ps]
                      for b, ps in self.paths.items()}
        self.cache = {}

    def font(self, bold, idx, px):
        key = (bold, idx, px)
        if key not in self.cache:
            self.cache[key] = ImageFont.truetype(self.paths[bold][idx], px)
        return self.cache[key]

    def pieces(self, text, bold, px):
        """[(font, substring)] with the fallback applied."""
        out, cur, cur_idx = [], "", None
        for ch in text:
            idx = 0 if ord(ch) in self.cmaps[bold][0] else 1
            if idx != cur_idx and cur:
                out.append((self.font(bold, cur_idx, px), cur))
                cur = ""
            cur_idx = idx
            cur += ch
        if cur:
            out.append((self.font(bold, cur_idx, px), cur))
        return out

    def width(self, text, bold, px):
        return sum(f.getlength(s) for f, s in self.pieces(text, bold, px))


FONTS = Fonts()


def parse_style(s):
    d = {}
    for part in (s or "").split(";"):
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            d[k] = v
        else:
            d[part] = "1"
    return d


class _Label(HTMLParser):
    """HTML label -> lines of runs (text, bold, size or None)."""

    def __init__(self):
        super().__init__()
        self.lines, self.bold, self.size = [[]], 0, [None]

    def handle_starttag(self, tag, attrs):
        if tag == "b":
            self.bold += 1
        elif tag == "br":
            self.lines.append([])
        elif tag in ("font", "small"):
            size = self.size[-1]
            m = re.search(r"font-size:\s*(\d+)px", dict(attrs).get("style", "") or "")
            if m:
                size = int(m.group(1))
            self.size.append(size)

    def handle_endtag(self, tag):
        if tag == "b":
            self.bold -= 1
        elif tag in ("font", "small") and len(self.size) > 1:
            self.size.pop()

    def handle_data(self, data):
        data = data.replace("　", "   ")
        if data:
            self.lines[-1].append((data, self.bold > 0, self.size[-1]))


def label_lines(value):
    p = _Label()
    p.feed(value or "")
    return [ln for ln in p.lines]


def wrap(lines, base, bold_all, max_w, s):
    """Word-wrap runs to max_w (None = no wrapping). Returns
    [(width, height, [(text, bold, px)])]."""
    out = []
    for runs in lines:
        words = []
        for text, bold, size in runs:
            px = round((size or base) * s)
            for w in re.findall(r"\S+|\s+", text):
                words.append((w, bold or bold_all, px))
        cur, cur_w, rows = [], 0.0, []
        for w, bold, px in words:
            ww = FONTS.width(w, bold, px)
            if max_w and cur and cur_w + ww > max_w and w.strip():
                rows.append(cur)
                cur, cur_w = [], 0.0
            if not cur and not w.strip():
                continue
            cur.append((w, bold, px))
            cur_w += ww
        rows.append(cur)
        for row in rows:
            while row and not row[-1][0].strip():
                row = row[:-1]
            width = sum(FONTS.width(w, b, px) for w, b, px in row)
            height = max([px for _, _, px in row] or [round(base * s)]) * 1.25
            out.append((width, height, row))
    return out


def draw_label(dr, value, style, box, s, color_default="#1B1F24"):
    x, y, w, h = box
    if not (value or "").strip():
        return
    base = float(style.get("fontSize", 12))
    bold_all = int(style.get("fontStyle", 0)) & 1 == 1
    color = style.get("fontColor", color_default)
    pad = 4 * s
    is_text = "text" in style
    max_w = (w - 2 * pad) if style.get("whiteSpace") == "wrap" else None
    rows = wrap(label_lines(value), base, bold_all, max_w, s)
    total = sum(r[1] for r in rows)
    valign = style.get("verticalAlign", "top" if is_text and "verticalAlign" not in style else "middle")
    if valign == "top":
        cy = y + float(style.get("spacingTop", 0)) * s + (2 * s if not is_text else 0)
    elif valign == "bottom":
        cy = y + h - total - 2 * s
    else:
        cy = y + (h - total) / 2
    align = style.get("align", "center")
    left = float(style.get("spacingLeft", 0)) * s
    for width, height, row in rows:
        if align == "left":
            cx = x + (0 if is_text else pad) + left
        elif align == "right":
            cx = x + w - width - (0 if is_text else pad)
        else:
            cx = x + (w - width) / 2
        for text, bold, px in row:
            for font, piece in FONTS.pieces(text, bold, px):
                dr.text((cx, cy + (height - px * 1.15) / 2), piece, font=font, fill=color)
                cx += font.getlength(piece)
        cy += height


def dashed(dr, pts, color, width, on, off):
    """A dashed polyline."""
    carry, draw_on = 0.0, True
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg == 0:
            continue
        ux, uy = (x1 - x0) / seg, (y1 - y0) / seg
        pos = 0.0
        while pos < seg:
            span = (on if draw_on else off) - carry
            step = min(span, seg - pos)
            if draw_on:
                dr.line([(x0 + ux * pos, y0 + uy * pos),
                         (x0 + ux * (pos + step), y0 + uy * (pos + step))],
                        fill=color, width=width)
            pos += step
            carry += step
            if carry >= (on if draw_on else off) - 1e-6:
                carry, draw_on = 0.0, not draw_on


def rounded_path(x, y, w, h, r, n=10):
    pts = []
    for cx, cy, a0 in ((x + w - r, y + r, -90), (x + w - r, y + h - r, 0),
                       (x + r, y + h - r, 90), (x + r, y + r, 180)):
        for i in range(n + 1):
            a = math.radians(a0 + 90 * i / n)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    pts.append(pts[0])
    return pts


class Vertex:
    def __init__(self, cell, geo):
        self.id = cell.get("id")
        self.value = cell.get("value", "")
        self.style = parse_style(cell.get("style"))
        self.x, self.y = float(geo.get("x", 0)), float(geo.get("y", 0))
        self.w, self.h = float(geo.get("width", 0)), float(geo.get("height", 0))
        self.rhombus = "rhombus" in self.style

    def center(self):
        return self.x + self.w / 2, self.y + self.h / 2

    def rel(self, fx, fy):
        return self.x + self.w * fx, self.y + self.h * fy

    def clip(self, toward):
        """Where the segment centre -> toward leaves this shape."""
        cx, cy = self.center()
        dx, dy = toward[0] - cx, toward[1] - cy
        if dx == 0 and dy == 0:
            return cx, cy
        hw, hh = self.w / 2, self.h / 2
        if self.rhombus:
            t = 1.0 / (abs(dx) / hw + abs(dy) / hh)
        else:
            t = min(hw / abs(dx) if dx else math.inf, hh / abs(dy) if dy else math.inf)
        return cx + dx * t, cy + dy * t


def side_of(fx, fy):
    if fx <= 0.001:
        return "W"
    if fx >= 0.999:
        return "E"
    if fy <= 0.001:
        return "N"
    if fy >= 0.999:
        return "S"
    return None


def route(style, src, dst, waypoints):
    ex, ey = style.get("exitX"), style.get("exitY")
    nx, ny = style.get("entryX"), style.get("entryY")
    ortho = style.get("edgeStyle") == "orthogonalEdgeStyle"
    p_exit = src.rel(float(ex), float(ey)) if ex is not None else None
    p_entry = dst.rel(float(nx), float(ny)) if nx is not None else None
    if not ortho:
        mids = list(waypoints)
        a = p_exit or src.clip(mids[0] if mids else (p_entry or dst.center()))
        b = p_entry or dst.clip(mids[-1] if mids else a)
        return [a] + mids + [b]
    a = p_exit or src.clip(waypoints[0] if waypoints else dst.center())
    b = p_entry or dst.clip(waypoints[-1] if waypoints else src.center())
    s_side = side_of(float(ex), float(ey)) if ex is not None else None
    e_side = side_of(float(nx), float(ny)) if nx is not None else None
    pts = [a]
    chain = list(waypoints) + [b]
    horizontal_first = s_side in ("E", "W")
    for i, q in enumerate(chain):
        p = pts[-1]
        if abs(p[0] - q[0]) > 0.5 and abs(p[1] - q[1]) > 0.5:
            last = i == len(chain) - 1
            if last and not waypoints and s_side and e_side and \
                    (s_side in "EW") == (e_side in "EW"):
                if s_side in "EW":          # Z shape through the middle x
                    mx = (p[0] + q[0]) / 2
                    pts += [(mx, p[1]), (mx, q[1])]
                else:
                    my = (p[1] + q[1]) / 2
                    pts += [(p[0], my), (q[0], my)]
            elif last and e_side:
                pts.append((p[0], q[1]) if e_side in "EW" else (q[0], p[1]))
            else:
                pts.append((q[0], p[1]) if horizontal_first else (p[0], q[1]))
        pts.append(q)
        horizontal_first = abs(pts[-1][1] - pts[-2][1]) < 0.5 if len(pts) > 1 else horizontal_first
        horizontal_first = not horizontal_first
    return pts


def arrow_head(dr, tip, frm, color, s, size=11):
    ang = math.atan2(tip[1] - frm[1], tip[0] - frm[0])
    L, half = size * s, size * s * 0.42
    bx, by = tip[0] - L * math.cos(ang), tip[1] - L * math.sin(ang)
    dr.polygon([tip,
                (bx + half * math.sin(ang), by - half * math.cos(ang)),
                (bx - half * math.sin(ang), by + half * math.cos(ang))], fill=color)
    return bx, by


def point_at(pts, frac):
    lens = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:])]
    target = sum(lens) * min(max(frac, 0.0), 1.0)
    for (a, b), ln in zip(zip(pts, pts[1:]), lens):
        if target <= ln and ln > 0:
            t = target / ln
            return a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
        target -= ln
    return pts[-1]


def render(model, out_path):
    cells = model.find("root").findall("mxCell")
    verts, order = {}, []
    for c in cells:
        geo = c.find("mxGeometry")
        if c.get("vertex") == "1" and geo is not None:
            v = Vertex(c, geo)
            verts[v.id] = v
            order.append(("v", v))
        elif c.get("edge") == "1":
            order.append(("e", c))
    xs, ys = [], []
    for v in verts.values():
        xs += [v.x, v.x + v.w]
        ys += [v.y, v.y + v.h]
    for kind, c in order:
        if kind == "e":
            for p in c.iter("mxPoint"):
                xs.append(float(p.get("x")))
                ys.append(float(p.get("y")))
    x0, y0 = min(xs) - MARGIN, min(ys) - MARGIN
    W, H = max(xs) + MARGIN - x0, max(ys) + MARGIN - y0
    s = SS * OUT_SCALE
    img = Image.new("RGB", (round(W * s), round(H * s)), "white")
    dr = ImageDraw.Draw(img)

    def T(p):
        return (p[0] - x0) * s, (p[1] - y0) * s

    labels = []
    for kind, item in order:
        if kind == "v":
            v, st = item, item.style
            bx, by = T((v.x, v.y))
            bw, bh = v.w * s, v.h * s
            fill = st.get("fillColor", "none" if "text" in st else "#FFFFFF")
            stroke = st.get("strokeColor", "none" if "text" in st else "#000000")
            sw = max(1, round(float(st.get("strokeWidth", 1)) * s))
            is_dashed = st.get("dashed") == "1"
            fill = None if fill == "none" else fill
            stroke = None if stroke == "none" else stroke
            if v.rhombus:
                poly = [(bx + bw / 2, by), (bx + bw, by + bh / 2),
                        (bx + bw / 2, by + bh), (bx, by + bh / 2)]
                dr.polygon(poly, fill=fill)
                if stroke:
                    dr.line(poly + [poly[0]], fill=stroke, width=sw, joint="curve")
            elif "text" not in st:
                if st.get("rounded") == "1":
                    r = min(bw, bh) * float(st.get("arcSize", 15)) / 100
                    r = min(r, min(bw, bh) / 2)
                else:
                    r = 0
                if fill:
                    dr.rounded_rectangle([bx, by, bx + bw, by + bh], radius=r, fill=fill)
                if stroke:
                    path = rounded_path(bx, by, bw, bh, r) if r else \
                        [(bx, by), (bx + bw, by), (bx + bw, by + bh), (bx, by + bh), (bx, by)]
                    if is_dashed:
                        dashed(dr, path, stroke, sw, 6 * s, 5 * s)
                    else:
                        dr.line(path, fill=stroke, width=sw, joint="curve")
            draw_label(dr, v.value, st, (bx, by, bw, bh), s)
        else:
            c, st = item, parse_style(item.get("style"))
            src, dst = verts.get(c.get("source")), verts.get(c.get("target"))
            if src is None or dst is None:
                continue
            geo = c.find("mxGeometry")
            way = []
            arr = geo.find("Array") if geo is not None else None
            if arr is not None:
                way = [(float(p.get("x")), float(p.get("y"))) for p in arr.findall("mxPoint")]
            pts = [T(p) for p in route(st, src, dst, way)]
            color = st.get("strokeColor", "#000000")
            sw = max(1, round(float(st.get("strokeWidth", 1)) * s))
            line = list(pts)
            if st.get("endArrow", "classic") != "none":
                line[-1] = arrow_head(dr, pts[-1], pts[-2], color, s)
            if st.get("startArrow", "none") != "none":
                line[0] = arrow_head(dr, pts[0], pts[1], color, s)
            if st.get("dashed") == "1":
                dashed(dr, line, color, sw, 7 * s, 5 * s)
            else:
                dr.line(line, fill=color, width=sw, joint="curve")
            if (c.get("value") or "").strip():
                frac = 0.5 + float(geo.get("x", 0) or 0) / 2 if geo is not None else 0.5
                # A label on a single horizontal segment sits above the line
                # (as draw.io places it); elsewhere it breaks the line.
                above = len(pts) == 2 and abs(pts[0][1] - pts[1][1]) < 0.5 * s
                room = math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]) - 6 * s
                labels.append((c.get("value"), st, point_at(pts, frac), above, room))
    for value, st, (cx, cy), above, room in labels:
        base = float(st.get("fontSize", 12))
        rows = wrap(label_lines(value), base, False, None, s)
        if above and max(r[0] for r in rows) > room:
            # Too wide for the gap between the two shapes: shrink, to 80 % at most.
            base *= max(0.8, room / max(r[0] for r in rows))
            st = dict(st, fontSize=base)
            rows = wrap(label_lines(value), base, False, None, s)
        w = max(r[0] for r in rows) + 8 * s
        h = sum(r[1] for r in rows) + 2 * s
        if above:
            cy -= h / 2 + 3 * s
        else:
            dr.rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], fill="white")
        st2 = dict(st, text="1", align="center", verticalAlign="middle")
        st2.pop("whiteSpace", None)
        draw_label(dr, value, st2, (cx - w / 2, cy - h / 2, w, h), s, "#4A5568")
    img = img.resize((round(W * OUT_SCALE), round(H * OUT_SCALE)), Image.LANCZOS)
    box = ImageChops.difference(img, Image.new("RGB", img.size, "white")).getbbox()
    if box:
        pad = PAD * OUT_SCALE
        img = img.crop((max(box[0] - pad, 0), max(box[1] - pad, 0),
                        min(box[2] + pad, img.width), min(box[3] + pad, img.height)))
    img.save(out_path)
    return img.size


def main(argv):
    files = [Path(a) for a in argv] or sorted(HERE.glob("[0-9][0-9]-*.drawio"))
    for f in files:
        root = ET.fromstring(f.read_text())
        model = root.find(".//mxGraphModel")
        size = render(model, f.with_suffix(".png"))
        print(f"{f.name} -> {f.with_suffix('.png').name} {size[0]}x{size[1]}")


if __name__ == "__main__":
    main(sys.argv[1:])
