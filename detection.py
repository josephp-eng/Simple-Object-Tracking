import argparse
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------- paths / run options
BASE_DIR = Path(__file__).resolve().parent
MEDIA_DIR = BASE_DIR / "Media"
VIDEO_PATH = MEDIA_DIR / "IMG_5048.MOV"
OUT_PATH = MEDIA_DIR / "tracked.mp4"
DEBUG_DIR = MEDIA_DIR           # where 's' saves frame_debug_<frame>.png

SCALE = 1
DISPLAY_SCALE = 0.4
WRITE_VIDEO = True              # True: write tracked.mp4 and stop at the end of the video

# ---------------------------------------------------------------- detection (tuned for IMG_5048.MOV)
@dataclass
class DetectParams:
    h_lo: int = 115
    h_hi: int = 145
    h2_lo: int = 145
    h2_hi: int = 179
    s_lo: int = 127
    s_hi: int = 255
    v_lo: int = 45
    v_hi: int = 180
    kernel: int = 5
    dilate_iter: int = 3
    min_area: int = 991
    max_area: int = 18076
    min_circ: float = 0.71
    min_aspect: float = 0.56      # height / width
    max_aspect: float = 1.33
    max_extent: float = 0.82      # area / bbox area
    min_solidity: float = 0.87    # area / convex hull area

BLUR = (7, 7)
MIN_BLOB_AREA = 1       # px, contours smaller than this are ignored entirely
MIN_BLOB_SIDE = 4       # px, contours with a thinner bounding box are ignored entirely

# Trackbars for --tune: (label, DetectParams field, slider max, divisor for float fields)
TRACKBARS = [
    ("H low", "h_lo", 179, 1),
    ("H high", "h_hi", 179, 1),
    ("S low", "s_lo", 255, 1),
    ("S high", "s_hi", 255, 1),
    ("V low", "v_lo", 255, 1),
    ("V high", "v_hi", 255, 1),
    ("H2 low", "h2_lo", 179, 1),
    ("H2 high", "h2_hi", 179, 1),
    ("Kernel", "kernel", 21, 1),
    ("Dilate iter", "dilate_iter", 10, 1),
    ("Min area", "min_area", 20000, 1),
    ("Max area", "max_area", 100000, 1),
    ("Min circ x100", "min_circ", 100, 100),
    ("Min aspect x100", "min_aspect", 300, 100),
    ("Max aspect x100", "max_aspect", 300, 100),
    ("Max extent x100", "max_extent", 100, 100),
    ("Min solidity x100", "min_solidity", 100, 100),
]

# ---------------------------------------------------------------- tracking
GATE_K = 2.8        # gate radius = GATE_K * sqrt(apple area): scales as apples shrink
MIN_GATE = 40       # px
MAX_COAST = 45      # frames a track may go without detection before it is dropped
VEL_SMOOTH = 0.5    # 0 = use only newest velocity, 1 = never update
VEL_DECAY = 0.97    # friction while coasting
AREA_SMOOTH = 0.3   # weight of the newest blob area in the running area estimate
MERGE_PAD = 0.15    # bbox padding for the merge test
TRAIL_LEN = 600

# ---------------------------------------------------------------- drawing (BGR colors)
APPLE_COLORS = [(0, 255, 0), (255, 160, 0)]
COAST_COLOR = (0, 200, 255)
REJECT_COLOR = (0, 0, 255)
CENTER_COLOR = (0, 0, 255)
FRAME_TEXT_COLOR = (0, 255, 255)
LINE_THICKNESS = 3
CENTER_RADIUS = 6
CROSSHAIR_HALF = 14     # px from center to crosshair tip
MIN_COAST_BOX_R = 6     # px, smallest half-size of the estimated box while coasting


def make_mask(frame, p):
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, BLUR, 0), cv2.COLOR_BGR2HSV)
    raw = None
    for h_lo, h_hi in [(p.h_lo, p.h_hi), (p.h2_lo, p.h2_hi)]:
        m = cv2.inRange(hsv, (h_lo, p.s_lo, p.v_lo), (h_hi, p.s_hi, p.v_hi))
        raw = m if raw is None else (raw | m)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.kernel, p.kernel))
    mask = cv2.morphologyEx(raw, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return cv2.dilate(mask, k, iterations=p.dilate_iter)


def find_blobs(mask, p):
    """Every contour, with 'ok' True if it passes all apple filters, else 'why' says which failed."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    blobs = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_BLOB_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if min(w, h) < MIN_BLOB_SIDE:
            continue
        perim = cv2.arcLength(c, True)
        circ = 4 * np.pi * area / perim ** 2 if perim > 0 else 0
        hull = cv2.contourArea(cv2.convexHull(c))
        solidity = area / hull if hull > 0 else 0
        aspect, extent = h / w, area / (w * h)
        M = cv2.moments(c)
        b = {"c": (M["m10"] / M["m00"], M["m01"] / M["m00"]), "area": area,
             "contour": c, "box": (x, y, w, h), "ok": False, "why": ""}
        if area < p.min_area:                              b["why"] = "small"
        elif area > p.max_area:                            b["why"] = "big"
        elif circ < p.min_circ:                            b["why"] = f"circ{circ:.2f}"
        elif not (p.min_aspect <= aspect <= p.max_aspect): b["why"] = f"asp{aspect:.1f}"
        elif extent > p.max_extent:                        b["why"] = f"ext{extent:.2f}"
        elif solidity < p.min_solidity:                    b["why"] = f"sol{solidity:.2f}"
        else:                                              b["ok"] = True
        blobs.append(b)
    return blobs


class Track:
    def __init__(self, name, color):
        self.name, self.color = name, color
        self.reset()

    def reset(self):
        self.active = False
        self.pos = self.vel = self.area = None
        self.coast = 0
        self.state = "-"
        self.contour = None
        self.box = None
        self.trail = deque(maxlen=TRAIL_LEN)

    def init(self, blob):
        self.active = True
        self.pos = np.array(blob["c"], float)
        self.vel = np.zeros(2)
        self.area = blob["area"]
        self.coast = 0
        self.state = "detected"
        self.contour = blob["contour"]
        self.box = blob["box"]
        self.trail.clear()
        self.trail.append(tuple(self.pos.astype(int)))

    def predict(self):
        return self.pos + self.vel

    def gate(self):
        return max(MIN_GATE, GATE_K * np.sqrt(self.area))

    def update(self, blob):
        new = np.array(blob["c"], float)
        self.vel = VEL_SMOOTH * self.vel + (1 - VEL_SMOOTH) * (new - self.pos)
        self.pos = new
        self.area = (1 - AREA_SMOOTH) * self.area + AREA_SMOOTH * blob["area"]
        self.coast = 0
        self.state = "detected"
        self.contour = blob["contour"]
        self.box = blob["box"]
        self.trail.append(tuple(self.pos.astype(int)))

    def coast_step(self, why):
        self.pos = self.pos + self.vel
        self.vel = self.vel * VEL_DECAY
        self.coast += 1
        self.state = why
        self.contour = None
        self.box = None
        self.trail.append(tuple(self.pos.astype(int)))
        if self.coast > MAX_COAST:
            self.state = "lost"


class Tracker:
    """Keeps the two apple tracks and matches each frame's blobs to them."""

    def __init__(self):
        self.tracks = [Track(f"Apple {i + 1}", color) for i, color in enumerate(APPLE_COLORS)]

    def reset(self):
        for t in self.tracks:
            t.reset()

    def states(self):
        return tuple(t.state for t in self.tracks)

    def status(self):
        return "  ".join(
            f"{t.name}={t.state}" + (f"@({int(t.pos[0])},{int(t.pos[1])})" if t.active else "")
            for t in self.tracks)

    def associate(self, blobs):
        """Update the tracks from this frame's blobs. Returns True if the tracks were lost and reset."""
        tracks = self.tracks
        good = [b for b in blobs if b["ok"]]

        # Not started yet: wait until exactly two apple-shaped blobs are seen together
        if not all(t.active for t in tracks):
            if len(good) == 2:
                for t, b in zip(tracks, sorted(good, key=lambda b: b["c"][0])):
                    t.init(b)
            return False

        preds = [t.predict() for t in tracks]

        # Merge test: apples are close AND one blob's bounding box covers both predictions
        if np.linalg.norm(preds[0] - preds[1]) < max(t.gate() for t in tracks):
            for b in blobs:
                x, y, w, h = b["box"]
                px, py = MERGE_PAD * w, MERGE_PAD * h
                if all(x - px <= p[0] <= x + w + px and y - py <= p[1] <= y + h + py for p in preds):
                    for t in tracks:
                        t.coast_step("merged")
                    return self._drop_if_lost()

        # Nearest-neighbour to PREDICTED position, inside the gate, closest pairs first
        pairs = []
        for i, t in enumerate(tracks):
            for j, b in enumerate(good):
                d = np.linalg.norm(np.array(b["c"]) - preds[i])
                if d <= t.gate():
                    pairs.append((d, i, j))
        used_t, used_b = set(), set()
        for d, i, j in sorted(pairs):
            if i in used_t or j in used_b:
                continue
            tracks[i].update(good[j])
            used_t.add(i)
            used_b.add(j)
        for i, t in enumerate(tracks):
            if i not in used_t:
                t.coast_step("missed")

        return self._drop_if_lost()

    def _drop_if_lost(self):
        if any(t.state == "lost" for t in self.tracks):     # give up and wait for a fresh start
            self.reset()
            return True
        return False


def draw(frame, blobs, tracks, frame_idx, show_debug):
    vis = frame.copy()
    if show_debug:
        for b in blobs:
            if not b["ok"]:
                cx, cy = map(int, b["c"])
                cv2.drawContours(vis, [b["contour"]], -1, REJECT_COLOR, 2)
                cv2.putText(vis, b["why"], (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, REJECT_COLOR, 2)
        cv2.putText(vis, f"frame {frame_idx}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, FRAME_TEXT_COLOR, 4)
    for t in tracks:
        if not t.active:
            continue
        pts = np.array(t.trail, np.int32).reshape(-1, 1, 2)
        if len(pts) > 1:
            cv2.polylines(vis, [pts], False, t.color, LINE_THICKNESS)

        cx, cy = map(int, t.pos)
        color = t.color if t.state == "detected" else COAST_COLOR

        if t.box is not None:                       # detected: box from the blob
            x, y, w, h = t.box
        else:                                       # coasting: box estimated from size
            r = max(MIN_COAST_BOX_R, int(np.sqrt(t.area / np.pi)))
            x, y, w, h = cx - r, cy - r, 2 * r, 2 * r

        cv2.rectangle(vis, (x, y), (x + w, y + h), color, LINE_THICKNESS)
        # estimated center (moments centroid) with a crosshair, inside the box
        cv2.circle(vis, (cx, cy), CENTER_RADIUS, CENTER_COLOR, -1)
        cv2.line(vis, (cx - CROSSHAIR_HALF, cy), (cx + CROSSHAIR_HALF, cy), CENTER_COLOR, 2)
        cv2.line(vis, (cx, cy - CROSSHAIR_HALF), (cx, cy + CROSSHAIR_HALF), CENTER_COLOR, 2)

        label = t.name if t.state == "detected" else f"{t.name} ({t.state})"
        cv2.putText(vis, label, (x, max(30, y - 12)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
        cv2.putText(vis, f"({cx},{cy})", (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    return vis


# ---------------------------------------------------------------- tune mode
def create_trackbars(p):
    cv2.namedWindow("controls", cv2.WINDOW_NORMAL)
    for label, field, max_val, div in TRACKBARS:
        cv2.createTrackbar(label, "controls", int(round(getattr(p, field) * div)), max_val, lambda _: None)


def read_trackbars(p):
    vals = {}
    for label, field, _, div in TRACKBARS:
        pos = cv2.getTrackbarPos(label, "controls")
        vals[field] = pos if div == 1 else pos / div
    vals["kernel"] = max(1, vals["kernel"])
    return replace(p, **vals)


# ---------------------------------------------------------------- main loop
def main():
    ap = argparse.ArgumentParser(description="Blob-based apple detection and tracking.")
    ap.add_argument("--tune", action="store_true",
                    help="show trackbars to tune the detection thresholds live (loops the video, writes no file)")
    args = ap.parse_args()

    write_video = WRITE_VIDEO and not args.tune
    params = DetectParams()
    if args.tune:
        create_trackbars(params)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {VIDEO_PATH}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    tracker = Tracker()
    writer = None
    paused, show_debug, view_mask = False, args.tune, False
    frame, mask, blobs, frame_idx = None, None, [], 0
    last_states = None

    while True:
        if args.tune:
            params = read_trackbars(params)

        if not paused or frame is None:
            ok, raw = cap.read()
            if not ok:
                if write_video:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                tracker.reset()
                last_states = None
                continue
            frame = cv2.resize(raw, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
            frame_idx += 1
            mask = make_mask(frame, params)
            blobs = find_blobs(mask, params)
            if tracker.associate(blobs):
                print(f"frame {frame_idx}: tracks lost, waiting for both apples")
            states = tracker.states()
            if states != last_states:
                print(f"frame {frame_idx}: {tracker.status()}")
                last_states = states
        elif args.tune:
            # paused while tuning: re-run detection on the held frame so slider changes show up
            mask = make_mask(frame, params)
            blobs = find_blobs(mask, params)

        vis = draw(frame, blobs, tracker.tracks, frame_idx, show_debug)

        if write_video and not paused:
            if writer is None:
                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(str(OUT_PATH), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(vis)

        show = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if view_mask else vis
        cv2.imshow("tracking", cv2.resize(show, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE,
                                          interpolation=cv2.INTER_AREA))
        key = cv2.waitKey(1 if write_video else 30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('d'):
            show_debug = not show_debug
        elif key == ord('m'):
            view_mask = not view_mask
        elif key == ord('s'):
            path = DEBUG_DIR / f"frame_debug_{frame_idx:04d}.png"
            cv2.imwrite(str(path), vis)
            print(f"saved {path}")

    if writer:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()

    if args.tune:
        print("\nFinal tuned values (copy into the DetectParams defaults):")
        print(params)


if __name__ == "__main__":
    main()
