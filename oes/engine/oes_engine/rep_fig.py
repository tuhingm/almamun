"""AER-style figures from a declarative spec, drawn to SVG and vector PDF from one scene (no matplotlib needed).

Style: white background, Times-like serif, black axes on the left and bottom only (no box, no grid), outward ticks, thin lines,
colour-blind-safe palette (Okabe-Ito) that stays distinguishable in grayscale because every series also has its own dash
pattern. No title inside the picture: the caption ("Figure N. ...", note, source) belongs to the text of the paper.

Spec (JSON-able):  {"ncols": 2, "panels": [panel, ...]}   panel = {"title", "xlabel", "ylabel", "xlim", "ylim", "xticks": [[pos, label]],
"yticks": [[pos, label]], "zero": bool, "legend": bool, "layers": [layer, ...]}
Layers: line, scatter, bar, band, hline, vline, errh, stem, heatmap, text (see Axes.draw_layer).
"""
import base64
import math

PALETTE = ["#000000", "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
DASHES = [None, (5, 3), (1.6, 2.6), (6, 2, 1.6, 2), (3, 2), None]
GRAY = "#6b6b6b"
LIGHT = "#bdbdbd"
MINUS = "−"
SERIF = "'Times New Roman', Times, 'Liberation Serif', 'Nimbus Roman', serif"

# Times-Roman / Bold / Italic advance widths (1/1000 em) for ASCII 32..126, from the standard PDF base-14 metrics.
_W = {
    "R": [250, 333, 408, 500, 500, 833, 778, 180, 333, 333, 500, 564, 250, 333, 250, 278, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 278, 278, 564, 564, 564, 444, 921, 722, 667, 667, 722, 611, 556, 722, 722, 333, 389, 722, 611, 889, 722, 722, 556, 722, 667, 556, 611, 722, 722, 944, 722, 722, 611, 333, 278, 333, 469, 500, 333, 444, 500, 444, 500, 444, 333, 500, 500, 278, 278, 500, 278, 778, 500, 500, 500, 500, 333, 389, 278, 500, 500, 722, 500, 500, 444, 480, 200, 480, 541],
    "B": [250, 333, 555, 500, 500, 1000, 833, 278, 333, 333, 500, 570, 250, 333, 250, 278, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 333, 333, 570, 570, 570, 500, 930, 722, 667, 722, 722, 667, 611, 778, 778, 389, 500, 778, 667, 944, 722, 778, 611, 778, 722, 556, 667, 722, 722, 1000, 722, 722, 667, 333, 278, 333, 581, 500, 333, 500, 556, 444, 556, 444, 333, 500, 556, 278, 333, 556, 278, 833, 556, 500, 556, 556, 444, 389, 333, 556, 500, 722, 500, 500, 444, 394, 220, 394, 520],
    "I": [250, 333, 420, 500, 500, 833, 778, 214, 333, 333, 500, 675, 250, 333, 250, 278, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 333, 333, 675, 675, 675, 500, 920, 611, 611, 667, 722, 611, 611, 722, 722, 333, 444, 667, 556, 833, 667, 722, 611, 722, 611, 500, 556, 722, 611, 833, 611, 556, 556, 389, 278, 389, 422, 500, 333, 500, 500, 444, 500, 444, 278, 500, 500, 278, 278, 444, 278, 722, 500, 500, 500, 500, 389, 389, 278, 500, 444, 667, 444, 444, 389, 400, 275, 400, 541],
}


def text_width(s, size, style="R"):
    t = _W.get(style, _W["R"])
    return sum((t[ord(c) - 32] if 32 <= ord(c) <= 126 else 500) for c in s) * size / 1000.0


def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _hex(c):
    c = c.lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    return tuple(int(c[i:i + 2], 16) / 255 for i in (0, 2, 4))


def _n(x):
    return f"{x:.2f}".rstrip("0").rstrip(".")


# ------------------------------------------------------------------------------------------------- scene
class Scene:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.items = []

    def line(self, pts, color="#000", lw=0.8, dash=None, alpha=1.0, close=False):
        self.items.append(("line", [(float(x), float(y)) for x, y in pts], color, lw, dash, alpha, close))

    def poly(self, pts, fill, alpha=1.0, stroke=None, lw=0.5):
        self.items.append(("poly", [(float(x), float(y)) for x, y in pts], fill, alpha, stroke, lw))

    def rect(self, x, y, w, h, fill=None, stroke=None, lw=0.5, alpha=1.0):
        self.poly([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], fill, alpha, stroke, lw) if fill else \
            self.line([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], stroke or "#000", lw, close=True)

    def circle(self, x, y, r, fill=None, stroke=None, lw=0.5, alpha=1.0):
        self.items.append(("circle", float(x), float(y), r, fill, stroke, lw, alpha))

    def text(self, x, y, s, size=8, anchor="start", rot=0, style="R", fill="#000"):
        if s != "":
            self.items.append(("text", float(x), float(y), str(s), size, anchor, rot, style, fill))

    # -- SVG
    def svg(self):
        W, H = self.w, self.h
        o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_n(W)} {_n(H)}" width="{_n(W)}pt" height="{_n(H)}pt" '
             f'font-family="{SERIF}" role="img"><rect width="{_n(W)}" height="{_n(H)}" fill="#ffffff"/>']
        for it in self.items:
            k = it[0]
            if k == "line":
                _, pts, col, lw, dash, alpha, close = it
                d = "M" + " L".join(f"{_n(x)},{_n(y)}" for x, y in pts) + (" Z" if close else "")
                da = f' stroke-dasharray="{" ".join(_n(v) for v in dash)}"' if dash else ""
                op = f' stroke-opacity="{alpha}"' if alpha < 1 else ""
                o.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{lw}"{da}{op} stroke-linejoin="round"/>')
            elif k == "poly":
                _, pts, fill, alpha, stroke, lw = it
                d = "M" + " L".join(f"{_n(x)},{_n(y)}" for x, y in pts) + " Z"
                st = f' stroke="{stroke}" stroke-width="{lw}"' if stroke else ""
                op = f' fill-opacity="{alpha}"' if alpha < 1 else ""
                o.append(f'<path d="{d}" fill="{fill}"{op}{st}/>')
            elif k == "circle":
                _, x, y, r, fill, stroke, lw, alpha = it
                f_ = f'fill="{fill}"' if fill else 'fill="none"'
                st = f' stroke="{stroke}" stroke-width="{lw}"' if stroke else ""
                op = f' fill-opacity="{alpha}"' if alpha < 1 and fill else ""
                o.append(f'<circle cx="{_n(x)}" cy="{_n(y)}" r="{_n(r)}" {f_}{op}{st}/>')
            else:
                _, x, y, s, size, anchor, rot, style, fill = it
                tr = f' transform="rotate({_n(rot)} {_n(x)} {_n(y)})"' if rot else ""
                fs = ' font-weight="bold"' if style == "B" else ' font-style="italic"' if style == "I" else ""
                o.append(f'<text x="{_n(x)}" y="{_n(y)}" font-size="{_n(size)}" text-anchor="{anchor}" fill="{fill}"{fs}{tr}>{_esc(s)}</text>')
        o.append("</svg>")
        return "".join(o)

    # -- PDF (vector, base-14 Times fonts, no embedding needed)
    def pdf(self):
        W, H = self.w, self.h
        alphas = [1.0]   # /G0 is always the opaque state

        def gs(a):
            if a not in alphas:
                alphas.append(a)
            return f"/G{alphas.index(a)} gs "

        def col(c, stroke):
            r, g, b = _hex(c)
            return f"{r:.3f} {g:.3f} {b:.3f} {'RG' if stroke else 'rg'} "

        def Y(y):
            return H - y

        c = []
        c.append("1 1 1 rg 0 0 %s %s re f " % (_n(W), _n(H)))
        for it in self.items:
            k = it[0]
            if k == "line":
                _, pts, colr, lw, dash, alpha, close = it
                s = col(colr, True) + f"{lw} w 1 j 0 J " + (gs(alpha) if alpha < 1 else "/G0 gs ")
                s += (f"[{' '.join(_n(v) for v in dash)}] 0 d " if dash else "[] 0 d ")
                s += f"{_n(pts[0][0])} {_n(Y(pts[0][1]))} m " + " ".join(f"{_n(x)} {_n(Y(y))} l" for x, y in pts[1:]) + (" h S " if close else " S ")
                c.append(s)
            elif k == "poly":
                _, pts, fill, alpha, stroke, lw = it
                s = col(fill, False) + (gs(alpha) if alpha < 1 else "/G0 gs ")
                s += f"{_n(pts[0][0])} {_n(Y(pts[0][1]))} m " + " ".join(f"{_n(x)} {_n(Y(y))} l" for x, y in pts[1:]) + " h "
                if stroke:
                    s += col(stroke, True) + f"{lw} w [] 0 d B "
                else:
                    s += "f "
                c.append(s)
            elif k == "circle":
                _, x, y, r, fill, stroke, lw, alpha = it
                kk = 0.5523 * r
                y = Y(y)
                path = (f"{_n(x + r)} {_n(y)} m {_n(x + r)} {_n(y + kk)} {_n(x + kk)} {_n(y + r)} {_n(x)} {_n(y + r)} c "
                        f"{_n(x - kk)} {_n(y + r)} {_n(x - r)} {_n(y + kk)} {_n(x - r)} {_n(y)} c "
                        f"{_n(x - r)} {_n(y - kk)} {_n(x - kk)} {_n(y - r)} {_n(x)} {_n(y - r)} c "
                        f"{_n(x + kk)} {_n(y - r)} {_n(x + r)} {_n(y - kk)} {_n(x + r)} {_n(y)} c h ")
                s = (gs(alpha) if alpha < 1 else "/G0 gs ")
                if fill:
                    s += col(fill, False)
                if stroke:
                    s += col(stroke, True) + f"{lw} w [] 0 d "
                s += path + ("B " if fill and stroke else "f " if fill else "S ")
                c.append(s)
            else:
                _, x, y, s_, size, anchor, rot, style, fill = it
                s_ = s_.replace(MINUS, "–")
                try:
                    raw = s_.encode("cp1252")
                except UnicodeEncodeError:
                    raw = s_.encode("cp1252", "replace")
                w = text_width(s_, size, style)
                shift = -w if anchor == "end" else -w / 2 if anchor == "middle" else 0
                phi = -math.radians(rot)
                cs, sn = math.cos(phi), math.sin(phi)
                tx, ty = x + shift * cs, Y(y) + shift * sn
                esc = "".join("\\" + ch if ch in "()\\" else ch for ch in raw.decode("latin-1"))
                fnt = {"R": "F1", "B": "F2", "I": "F3"}[style]
                c.append("/G0 gs " + col(fill, False) + f"BT /{fnt} {_n(size)} Tf {cs:.4f} {sn:.4f} {-sn:.4f} {cs:.4f} {_n(tx)} {_n(ty)} Tm ({esc}) Tj ET ")
        stream = "".join(c).encode("latin-1")
        order = alphas
        gsd = " ".join(f"/G{i} << /ca {a} /CA {a} >>" for i, a in enumerate(order))
        objs = [
            "<< /Type /Catalog /Pages 2 0 R >>",
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {_n(W)} {_n(H)}] /Contents 4 0 R /Resources << /Font << /F1 5 0 R /F2 6 0 R /F3 7 0 R >> /ExtGState << {gsd} >> >> >>",
            None,
            "<< /Type /Font /Subtype /Type1 /BaseFont /Times-Roman /Encoding /WinAnsiEncoding >>",
            "<< /Type /Font /Subtype /Type1 /BaseFont /Times-Bold /Encoding /WinAnsiEncoding >>",
            "<< /Type /Font /Subtype /Type1 /BaseFont /Times-Italic /Encoding /WinAnsiEncoding >>",
        ]
        out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offs = []
        for i, body in enumerate(objs, 1):
            offs.append(len(out))
            if body is None:
                out += f"{i} 0 obj\n<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream\nendobj\n"
            else:
                out += f"{i} 0 obj\n{body}\nendobj\n".encode()
        x0 = len(out)
        out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
        for o in offs:
            out += f"{o:010d} 00000 n \n".encode()
        out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{x0}\n%%EOF\n".encode()
        return bytes(out)

    def pdf_b64(self):
        return base64.b64encode(self.pdf()).decode()


# ------------------------------------------------------------------------------------------------- axes
def nice_ticks(lo, hi, n=5):
    if not math.isfinite(lo) or not math.isfinite(hi):
        return [0.0, 1.0]
    if hi == lo:
        hi = lo + 1.0
    raw = (hi - lo) / max(n - 1, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if raw <= m * mag * 1.0000001)
    a = math.ceil(lo / step - 1e-09)
    b = math.floor(hi / step + 1e-09)
    return [round(i * step, 12) for i in range(a, b + 1)]


def tick_labels(ticks):
    if not ticks:
        return []
    for d in range(0, 7):
        if all(abs(t * 10 ** d - round(t * 10 ** d)) < 1e-06 * max(1, abs(t * 10 ** d)) for t in ticks):
            break
    out = []
    for t in ticks:
        s = f"{t:.{d}f}"
        if float(s) == 0:
            s = s.lstrip("-")
        out.append(s.replace("-", MINUS))
    return out


class Axes:
    def __init__(self, sc, x0, y0, w, h, panel):
        self.sc, self.x0, self.y0, self.w, self.h, self.p = sc, x0, y0, w, h, panel
        self.layers = panel.get("layers", [])
        self.size = 7.5

    # data -> scene
    def X(self, v):
        a, b = self.xlim
        return self.x0 + (v - a) / (b - a) * self.w

    def Y(self, v):
        a, b = self.ylim
        return self.y0 + self.h - (v - a) / (b - a) * self.h

    def extent(self):
        xs, ys = [], []
        for L in self.layers:
            t = L["type"]
            if t in ("line", "scatter", "stem", "bar"):
                xs += [v for v in L["x"] if v is not None]
                ys += [v for v in L["y"] if v is not None]
                if t in ("bar", "stem"):
                    ys.append(0.0)
                if t == "bar":
                    w = L.get("width", 0.8)
                    xs += [min(L["x"]) - w / 2, max(L["x"]) + w / 2]
            elif t == "band":
                xs += L["x"]
                ys += [v for v in L["lo"] + L["hi"] if v is not None]
            elif t == "errh":
                ys += list(L["y"])
                xs += [v for v in L["lo"] + L["hi"] + L["x"] if v is not None]
            elif t == "hline":
                ys.append(L["y"])
            elif t == "vline":
                xs.append(L["x"])
            elif t == "errv":
                xs += list(L["x"])
                ys += [v for v in L["lo"] + L["hi"] + L["y"] if v is not None]
            elif t == "box":
                xs += [min(L["x"]) - 0.6, max(L["x"]) + 0.6]
                for b in L["boxes"]:
                    ys += [b["lo"], b["hi"]] + list(b.get("out", []))
        return xs, ys

    def setup(self):
        p = self.p
        if self.layers and self.layers[0]["type"] == "heatmap":
            m = self.layers[0]
            self.xlim, self.ylim = (-0.5, len(m["xlabels"]) - 0.5), (len(m["ylabels"]) - 0.5, -0.5)
            return
        xs, ys = self.extent()
        xl, yl = p.get("xlim"), p.get("ylim")
        if xl is None:
            lo, hi = (min(xs), max(xs)) if xs else (0, 1)
            pad = (hi - lo) * 0.04 or 0.5
            xl = [lo - pad, hi + pad]
        if yl is None:
            lo, hi = (min(ys), max(ys)) if ys else (0, 1)
            if p.get("zero") and False:
                lo, hi = min(lo, 0), max(hi, 0)
            pad = (hi - lo) * 0.06 or 0.5
            yl = [lo - pad, hi + pad]
        self.xlim, self.ylim = tuple(map(float, xl)), tuple(map(float, yl))

    def frame(self):
        sc, p, S = self.sc, self.p, self.size
        if self.layers and self.layers[0]["type"] == "heatmap":
            return
        sc.line([(self.x0, self.y0), (self.x0, self.y0 + self.h), (self.x0 + self.w, self.y0 + self.h)], "#000", 0.7)
        xt = p.get("xticks") or [[t, lab] for t, lab in zip(nice_ticks(*self.xlim, 6), tick_labels(nice_ticks(*self.xlim, 6)))]
        yt = p.get("yticks") or [[t, lab] for t, lab in zip(nice_ticks(*self.ylim, 5), tick_labels(nice_ticks(*self.ylim, 5)))]
        for pos, lab in xt:
            if self.xlim[0] - 1e-9 <= pos <= self.xlim[1] + 1e-9:
                x = self.X(pos)
                sc.line([(x, self.y0 + self.h), (x, self.y0 + self.h + 3)], "#000", 0.6)
                sc.text(x, self.y0 + self.h + 3 + S + 1, lab, S, "middle", rot=p.get("xrot", 0) and -p["xrot"] and 0)
        for pos, lab in yt:
            if self.ylim[0] - 1e-9 <= pos <= self.ylim[1] + 1e-9:
                y = self.Y(pos)
                sc.line([(self.x0 - 3, y), (self.x0, y)], "#000", 0.6)
                sc.text(self.x0 - 5, y + S * 0.34, lab, S, "end")
        if p.get("xlabel"):
            sc.text(self.x0 + self.w / 2, self.y0 + self.h + 3 + 2 * S + 6, p["xlabel"], 8, "middle")
        if p.get("ylabel"):
            sc.text(self.x0 - 31 if not p.get("ylab_off") else self.x0 - p["ylab_off"], self.y0 + self.h / 2, p["ylabel"], 8, "middle", rot=-90)
        if p.get("zero") and self.ylim[0] < 0 < self.ylim[1]:
            sc.line([(self.x0, self.Y(0)), (self.x0 + self.w, self.Y(0))], LIGHT, 0.5)

    def style(self, L, i):
        k = L.get("k", i)
        return PALETTE[k % len(PALETTE)], DASHES[k % len(DASHES)]

    def draw_layers(self):
        sc = self.sc
        for i, L in enumerate(self.layers):
            t = L["type"]
            col, dash = self.style(L, i)
            if t == "line":
                run = []
                for x, y in zip(L["x"], L["y"]):
                    if y is None or (isinstance(y, float) and not math.isfinite(y)):
                        if len(run) > 1:
                            sc.line(run, col, L.get("lw", 0.9), dash)
                        run = []
                    else:
                        run.append((self.X(x), self.Y(y)))
                if len(run) > 1:
                    sc.line(run, col, L.get("lw", 0.9), dash)
            elif t == "band":
                up = [(self.X(x), self.Y(v)) for x, v in zip(L["x"], L["hi"]) if v is not None]
                dn = [(self.X(x), self.Y(v)) for x, v in zip(L["x"], L["lo"]) if v is not None][::-1]
                sc.poly(up + dn, L.get("fill", "#9a9a9a"), L.get("alpha", 0.28))
            elif t == "scatter":
                r = L.get("r", 1.7)
                for x, y in zip(L["x"], L["y"]):
                    if y is None or x is None:
                        continue
                    if L.get("hollow"):
                        sc.circle(self.X(x), self.Y(y), r, None, col, 0.5)
                    else:
                        sc.circle(self.X(x), self.Y(y), r, L.get("fill", "#333333"), None, 0.5, L.get("alpha", 0.55))
            elif t == "bar":
                bw = L.get("width", 0.8)
                for x, y in zip(L["x"], L["y"]):
                    if y is None:
                        continue
                    xa, xb = self.X(x - bw / 2), self.X(x + bw / 2)
                    ya, yb = self.Y(0), self.Y(y)
                    sc.poly([(xa, ya), (xb, ya), (xb, yb), (xa, yb)], L.get("fill", "#8a8a8a"), 1.0, "#000", 0.4)
            elif t == "stem":
                for x, y in zip(L["x"], L["y"]):
                    sc.line([(self.X(x), self.Y(0)), (self.X(x), self.Y(y))], "#000", 0.5)
                    sc.circle(self.X(x), self.Y(y), 1.3, "#000")
            elif t == "hline":
                sc.line([(self.x0, self.Y(L["y"])), (self.x0 + self.w, self.Y(L["y"]))], L.get("color", "#000" if not L.get("dash") else "#555"), L.get("lw", 0.5), L.get("dash", (3, 2)))
            elif t == "vline":
                sc.line([(self.X(L["x"]), self.y0), (self.X(L["x"]), self.y0 + self.h)], L.get("color", "#555"), L.get("lw", 0.5), L.get("dash", (3, 2)))
            elif t == "errh":
                for yy, est, lo, hi in zip(L["y"], L["x"], L["lo"], L["hi"]):
                    y = self.Y(yy)
                    sc.line([(self.X(lo), y), (self.X(hi), y)], "#000", 0.8)
                    for e in (lo, hi):
                        sc.line([(self.X(e), y - 2.2), (self.X(e), y + 2.2)], "#000", 0.8)
                    sc.circle(self.X(est), y, 2.4, "#000")
            elif t == "errv":
                for xx, est, lo, hi in zip(L["x"], L["y"], L["lo"], L["hi"]):
                    x = self.X(xx)
                    sc.line([(x, self.Y(lo)), (x, self.Y(hi))], "#000", 0.8)
                    for e in (lo, hi):
                        sc.line([(x - 2.6, self.Y(e)), (x + 2.6, self.Y(e))], "#000", 0.8)
                    sc.circle(x, self.Y(est), 2.4, "#000")
            elif t == "box":
                hw = L.get("half", 0.28)
                for xx, b in zip(L["x"], L["boxes"]):
                    xa, xb, xm = self.X(xx - hw), self.X(xx + hw), self.X(xx)
                    sc.poly([(xa, self.Y(b["q1"])), (xb, self.Y(b["q1"])), (xb, self.Y(b["q3"])), (xa, self.Y(b["q3"]))], L.get("fill", "#dcdcdc"), 1.0, "#000", 0.7)
                    sc.line([(xa, self.Y(b["med"])), (xb, self.Y(b["med"]))], "#000", 1.4)
                    sc.line([(xm, self.Y(b["q3"])), (xm, self.Y(b["hi"]))], "#000", 0.7)
                    sc.line([(xm, self.Y(b["q1"])), (xm, self.Y(b["lo"]))], "#000", 0.7)
                    for e in (b["lo"], b["hi"]):
                        sc.line([(self.X(xx - hw / 2), self.Y(e)), (self.X(xx + hw / 2), self.Y(e))], "#000", 0.7)
                    for o in b.get("out", []):
                        sc.circle(xm, self.Y(o), 1.5, None, "#000", 0.6)
            elif t == "text":
                sc.text(self.X(L["x"]), self.Y(L["y"]), L["s"], L.get("size", 7), L.get("anchor", "start"), style=L.get("style", "R"))

    def heatmap(self):
        L = self.layers[0]
        sc = self.sc
        M, xl, yl = L["matrix"], L["xlabels"], L["ylabels"]
        vmin, vmax = L.get("vmin", -1.0), L.get("vmax", 1.0)
        cw, ch = self.w / len(xl), self.h / len(yl)
        neg, pos, mid = _hex("#0072B2"), _hex("#D55E00"), (1, 1, 1)
        for i, row in enumerate(M):
            for j, v in enumerate(row):
                if v is None:
                    continue
                z = max(-1.0, min(1.0, (v - (vmin + vmax) / 2) / ((vmax - vmin) / 2)))
                tgt = pos if z > 0 else neg
                rgb = tuple(mid[q] + (tgt[q] - mid[q]) * abs(z) * 0.85 for q in range(3))
                hexc = "#%02x%02x%02x" % tuple(int(round(c * 255)) for c in rgb)
                x, y = self.x0 + j * cw, self.y0 + i * ch
                sc.poly([(x, y), (x + cw, y), (x + cw, y + ch), (x, y + ch)], hexc, 1.0, "#ffffff", 0.6)
                if L.get("annot", True):
                    lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                    sc.text(x + cw / 2, y + ch / 2 + 2.5, f"{v:.2f}".replace("-", MINUS), min(7.5, cw / 4.2), "middle", fill="#ffffff" if lum < 0.45 else "#000")
        for j, lab in enumerate(xl):
            sc.text(self.x0 + (j + 0.5) * cw, self.y0 + self.h + 10, lab, 7.5, "end", rot=-40)
        for i, lab in enumerate(yl):
            sc.text(self.x0 - 4, self.y0 + (i + 0.5) * ch + 2.5, lab, 7.5, "end")

    def legend(self):
        items = [(L.get("label"), *self.style(L, i), L["type"]) for i, L in enumerate(self.layers) if L.get("label")]
        if not items:
            return
        x, y = self.x0, self.y0 - 7
        for lab, col, dash, t in items:
            if t == "scatter":
                self.sc.circle(x + 6, y - 2.6, 1.8, "#333", None, 0.5, 0.7)
            elif t == "band":
                self.sc.poly([(x, y - 6), (x + 12, y - 6), (x + 12, y), (x, y)], "#9a9a9a", 0.4)
            else:
                self.sc.line([(x, y - 2.6), (x + 13, y - 2.6)], col, 1.0, dash)
            self.sc.text(x + 16, y, lab, 7.5)
            x += 16 + text_width(lab, 7.5) + 12


def render(spec, width_in=6.5, height_in=3.6):
    panels = spec["panels"]
    ncols = spec.get("ncols", 1)
    nrows = math.ceil(len(panels) / ncols)
    W, H = width_in * 72, height_in * 72
    sc = Scene(W, H)
    cw, ch = W / ncols, H / nrows
    for idx, p in enumerate(panels):
        r, c = divmod(idx, ncols)
        has_leg = any(L.get("label") for L in p.get("layers", []))
        heat = p.get("layers") and p["layers"][0]["type"] == "heatmap"
        ml = p.get("ml", 40 if not heat else 66)
        mr = 8
        mt = 20 + (10 if has_leg else 0)
        mb = 33 if not heat else 56
        ax = Axes(sc, c * cw + ml, r * ch + mt, cw - ml - mr, ch - mt - mb, p)
        ax.setup()
        if p.get("title"):
            sc.text(c * cw + ml, r * ch + 11, p["title"], 8.5, "start", style="B")
        if heat:
            ax.heatmap()
        else:
            ax.frame()
            ax.draw_layers()
            if has_leg:
                ax.legend()
    return sc
