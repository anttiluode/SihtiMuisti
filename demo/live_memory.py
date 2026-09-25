"""Live visual memory: watch a camera's past fade by octave and fold into events.

    python demo/live_memory.py                     # webcam 0
    python demo/live_memory.py --video clip.mp4    # a file
    python demo/live_memory.py --synthetic 1       # the test world, no camera needed
    python demo/live_memory.py --synthetic 1 --headless 600 --save shot.png

Keys (window mode):
    SPACE  "have I seen this before?" - the current view is a moment query; the best
           stored item is outlined and counted as recalled
    p      switch policy (rd / fade_recall / thumbnails_L3), memory restarts
    s      save a screenshot
    q/ESC  quit

What you see: left, the camera as the memory receives it (64x64 grey). Right, top,
a timeline: every block is one stored item, its width the stretch of time it covers,
its shade how much detail it still holds (dark = sharp). Right, bottom: the stored
items themselves, newest last, as the memory can actually reconstruct them.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sihtimuisti.memory import make_policy  # noqa: E402
from sihtimuisti.scorer import Background, Scorer  # noqa: E402

SHADES = [(107, 54, 13), (149, 79, 24), (191, 106, 37), (229, 135, 57), (236, 167, 109), (239, 182, 134)]
POLICIES = ["rd", "fade_recall", "thumbnails_L3"]
W, H = 1280, 640


def to_64(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    h, w = g.shape
    s = min(h, w)
    g = g[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    return cv2.resize(g, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float64) / 255.0


def gray_tile(img, size):
    x = np.clip(img * 255, 0, 255).astype(np.uint8)
    x = cv2.resize(x, (size, size), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)


class Source:
    def __init__(self, args):
        self.world = None
        if args.synthetic is not None:
            from sihtimuisti.world import make_world
            self.world = make_world(args.synthetic, 100000 if not args.headless else max(args.headless, 600))
            self.rng = np.random.default_rng(args.synthetic)
            self.t = 0
        else:
            self.cap = cv2.VideoCapture(args.video if args.video else args.camera)
            if not self.cap.isOpened():
                sys.exit("could not open camera/video")

    def read(self):
        if self.world is not None:
            if self.t >= self.world.T:
                return None
            f = self.world.frame(self.t, self.rng)
            self.t += 1
            return np.clip(f, 0, 1)
        ok, frame = self.cap.read()
        return to_64(frame) if ok else None


class App:
    def __init__(self, budget_frames, policy):
        self.budget = budget_frames * 4096
        self.set_policy(policy)

    def set_policy(self, name):
        self.policy = name
        self.mem = make_policy(name, self.budget)
        self.bg, self.scorer = Background(), Scorer()
        self.t, self.hit, self.msg = 0, None, f"policy {name}"

    def step(self, f):
        self.bg.update(f)
        self.mem.ingest(f, self.t)
        self.t += 1

    def query(self, f):
        s = self.scorer.moment_scores(f, self.mem.items, self.bg.snapshot())
        k = int(np.argmax(s))
        it = self.mem.items[k]
        self.mem.recall(it)
        self.hit = it.id
        ago = self.t - 1 - it.tmax
        self.msg = (f"looks like stored item t {it.tmin}-{it.tmax} ({ago} steps ago, "
                    f"{64 >> it.level}x{64 >> it.level}, recalled {it.recalls}x)")

    def render(self, live):
        canvas = np.full((H, W, 3), 245, np.uint8)
        canvas[40:40 + 512, 20:20 + 512] = gray_tile(live, 512)
        cv2.putText(canvas, "now (what the memory receives)", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1)
        items = self.mem.items
        x0, x1, y0 = 560, W - 20, 50
        cv2.putText(canvas, f"{self.policy}: {len(items)} items, {self.mem.used()}/{self.budget} scalars",
                    (x0, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1)
        T = max(self.t, 1)
        for it in items:
            a = x0 + int((x1 - x0) * it.tmin / T)
            b = max(a + 1, x0 + int((x1 - x0) * (it.tmax + 1) / T))
            cv2.rectangle(canvas, (a, y0), (b - 1, y0 + 40), SHADES[it.level], -1)
            if it.id == self.hit:
                cv2.rectangle(canvas, (a - 2, y0 - 3), (b + 1, y0 + 43), (30, 30, 220), 2)
        for i, lab in enumerate(["64", "32", "16", "8", "4", "2"]):
            cv2.rectangle(canvas, (x0 + i * 70, y0 + 52), (x0 + i * 70 + 14, y0 + 66), SHADES[i], -1)
            cv2.putText(canvas, f"{lab}px", (x0 + i * 70 + 18, y0 + 64), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)
        show = items[-40:]
        size, gap, cols = 76, 6, 8
        for i, it in enumerate(show):
            r, c = divmod(i, cols)
            y, x = 140 + r * (size + gap + 12), x0 + c * (size + gap)
            if y + size > H - 62:
                break
            canvas[y:y + size, x:x + size] = gray_tile(it.view(), size)
            if it.id == self.hit:
                cv2.rectangle(canvas, (x - 2, y - 2), (x + size + 1, y + size + 1), (30, 30, 220), 2)
            cv2.putText(canvas, f"{it.tmin}-{it.tmax}", (x, y + size + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (80, 80, 80), 1)
        cv2.putText(canvas, self.msg, (20, H - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 180), 1)
        cv2.putText(canvas, "SPACE seen before?   p policy   s save   q quit", (20, H - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1)
        return canvas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video")
    ap.add_argument("--synthetic", type=int)
    ap.add_argument("--policy", default="rd", choices=POLICIES)
    ap.add_argument("--budget", type=int, default=8, help="budget in 64x64 frame-equivalents")
    ap.add_argument("--every", type=int, default=3, help="store one of every N camera frames")
    ap.add_argument("--headless", type=int, default=0, help="run N stored frames, save, exit")
    ap.add_argument("--save", default="live_memory.png")
    ap.add_argument("--ask", type=int, help="headless + synthetic: ask about a fresh view of this past frame")
    args = ap.parse_args()

    src, app = Source(args), App(args.budget, args.policy)
    n = 0
    if args.headless:
        f = None
        while app.t < args.headless:
            g = src.read()
            if g is None:
                break
            f = g
            if n % (1 if args.synthetic is not None else args.every) == 0:
                app.step(f)
            n += 1
        if args.ask is not None and src.world is not None:
            f = np.clip(src.world.frame(args.ask, np.random.default_rng(99)), 0, 1)
        app.query(f)
        if args.ask is not None:
            app.msg = f"asked about frame {args.ask}: " + app.msg
        cv2.imwrite(args.save, app.render(f))
        print(app.msg, "->", args.save)
        return

    last = time.time()
    while True:
        f = src.read()
        if f is None:
            break
        if n % args.every == 0:
            app.step(f)
        n += 1
        cv2.imshow("SihtiMuisti", app.render(f))
        if args.synthetic is not None:
            dt = time.time() - last
            time.sleep(max(0.0, 1 / 30 - dt))
            last = time.time()
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord(" ") and app.mem.items:
            app.query(f)
        if k == ord("p"):
            app.set_policy(POLICIES[(POLICIES.index(app.policy) + 1) % len(POLICIES)])
        if k == ord("s"):
            name = f"memory_{int(time.time())}.png"
            cv2.imwrite(name, app.render(f))
            app.msg = f"saved {name}"
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
