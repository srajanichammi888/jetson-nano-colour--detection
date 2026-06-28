╔══════════════════════════════════════════════════════════════════╗
║   Colour Detection — Jetson Nano + Raspberry Pi Camera  v4      ║
║   Single unified file: detection + HSV tuner in one place       ║
╠══════════════════════════════════════════════════════════════════╣
║  Run modes (pass as command-line argument):                      ║
║    python3 colour_detection.py detect   ← real-time detection   ║
║    python3 colour_detection.py tune     ← HSV calibration tool  ║
║  Default (no argument) = detect                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  Detect mode keys:                                               ║
║    Q / Esc  Quit                                                 ║
║    S        Save screenshot                                      ║
║    P        Pause / resume                                       ║
║    C        Toggle contour overlay                               ║
║    T        Switch to Tuner mode (live, no restart needed)       ║
║  Tune mode keys:                                                 ║
║    S        Print current HSV values to console                  ║
║    A        Apply tuned range to a named colour (interactive)    ║
║    D        Switch back to Detect mode                           ║
║    Q / Esc  Quit                                                 ║
╠══════════════════════════════════════════════════════════════════╣
║  Dependencies:  pip3 install opencv-python numpy                 ║
║  Hardware:      Jetson Nano (JetPack 4.x) + RPi Camera v2 (CSI) ║
╚══════════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
import time
import csv
import os
import sys


# ═══════════════════════════════════════════════════════════════════
#  SECTION 1 — SHARED CONFIGURATION
#  Everything lives here. Tune detection behaviour from one place.
# ═══════════════════════════════════════════════════════════════════
CFG = {
    # ── Camera / GStreamer pipeline ──────────────────────────────
    "capture_width":    1280,
    "capture_height":   720,
    "display_width":    800,
    "display_height":   450,
    "framerate":        30,
    "flip_method":      0,      # 0=none  1=CCW  2=180°  3=CW

    # ── Preprocessing ────────────────────────────────────────────
    "blur_ksize":       11,     # Gaussian kernel size (must be odd)

    # ── Morphological cleanup ────────────────────────────────────
    "morph_ksize":      9,      # Structuring element size (ellipse)
    "morph_open_iter":  2,      # OPEN iterations  (removes noise)
    "morph_dilate_iter":1,      # DILATE iterations (fills gaps)

    # ── Detection ────────────────────────────────────────────────
    # Minimum contour area as a fraction of total frame pixels.
    # 0.0015 = 0.15% ≈ 540 px² at 800×450. Scales with resolution.
    "min_area_frac":    0.0015,

    # ── Centroid tracker ─────────────────────────────────────────
    # EMA weight for smoothing bounding-box jitter (0–1).
    # Higher = reacts faster but jitters more.
    "track_alpha":      0.35,

    # ── Distance estimation ──────────────────────────────────────
    # Uses pinhole model: distance = (real_cm × focal_px) / px_width
    # HOW TO CALIBRATE focal_px:
    #   1. Hold an object of known width (e.g. 10 cm) at a known
    #      distance (e.g. 30 cm) in front of the camera.
    #   2. Note its pixel width in the detect window.
    #   3. focal_px = (pixel_width × 30) / 10
    "known_object_cm":  10.0,   # assumed real width of objects (cm)
    "focal_px":         600.0,  # calibrate this value!
    "show_distance":    True,   # show distance label on bounding box

    # ── CSV logging ──────────────────────────────────────────────
    "enable_csv_log":   False,  # True → append detections to CSV
    "csv_path":         "detections.csv",

    # ── Display ──────────────────────────────────────────────────
    "show_contour":     False,  # draw contour outline (press C)
    "hud_alpha":        0.6,    # HUD panel transparency (0–1)
}

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ═══════════════════════════════════════════════════════════════════
#  SECTION 2 — COLOUR DEFINITIONS
#
#  HSV ranges (OpenCV scale: H 0–180, S 0–255, V 0–255)
#
#  ORDERING MATTERS: colours listed first take priority when pixels
#  overlap. Brown must be before Orange because they share the same
#  hue band (8–20°); only Value (brightness) separates them.
#
#  HOW TO ADD A NEW COLOUR:
#    1. Run:  python3 colour_detection.py tune
#    2. Adjust trackbars until only your target colour is white
#    3. Press S to print the HSV values
#    4. Add an entry below using those values
# ═══════════════════════════════════════════════════════════════════
COLOR_RANGES = {
    "Red": {
        # Red wraps around hue=0/180 in HSV, so two masks are needed
        "lower1": np.array([0,   120,  70]),
        "upper1": np.array([10,  255, 255]),
        "lower2": np.array([170, 120,  70]),
        "upper2": np.array([180, 255, 255]),
        "bgr":    (0, 0, 220),
        "dual":   True,
    },
    "Brown": {
        # Same hue as orange but dark (V ≤ 150). Must come before Orange.
        "lower1": np.array([8,   80,  20]),
        "upper1": np.array([20, 255, 150]),
        "bgr":    (19, 69, 139),
        "dual":   False,
    },
    "Orange": {
        # Bright version of the brown hue band (V > 150).
        "lower1": np.array([10, 120, 151]),
        "upper1": np.array([25, 255, 255]),
        "bgr":    (0, 130, 255),
        "dual":   False,
    },
    "Yellow": {
        "lower1": np.array([25, 100, 100]),
        "upper1": np.array([35, 255, 255]),
        "bgr":    (0, 220, 255),
        "dual":   False,
    },
    "Green": {
        "lower1": np.array([36,  80,  60]),
        "upper1": np.array([85, 255, 255]),
        "bgr":    (0, 200, 0),
        "dual":   False,
    },
    "Cyan": {
        "lower1": np.array([85,  100, 100]),
        "upper1": np.array([100, 255, 255]),
        "bgr":    (220, 200, 0),
        "dual":   False,
    },
    "Blue": {
        "lower1": np.array([100, 100,  70]),
        "upper1": np.array([130, 255, 255]),
        "bgr":    (220, 0, 0),
        "dual":   False,
    },
    "Purple": {
        "lower1": np.array([130,  60,  60]),
        "upper1": np.array([155, 255, 255]),
        "bgr":    (200, 0, 180),
        "dual":   False,
    },
    "Pink": {
        "lower1": np.array([155,  50, 100]),
        "upper1": np.array([170, 255, 255]),
        "bgr":    (180, 100, 255),
        "dual":   False,
    },
    "White": {
        "lower1": np.array([0,   0,  200]),
        "upper1": np.array([180, 30, 255]),
        "bgr":    (230, 230, 230),
        "dual":   False,
    },
    "Black": {
        "lower1": np.array([0,   0,   0]),
        "upper1": np.array([180, 255, 50]),
        "bgr":    (80, 80, 80),
        "dual":   False,
    },
}


# ═══════════════════════════════════════════════════════════════════
#  SECTION 3 — SHARED GSTREAMER PIPELINE
#  One definition used by both detect and tune modes.
# ═══════════════════════════════════════════════════════════════════
def build_pipeline(display_w=None, display_h=None):
    """Return GStreamer pipeline string for the RPi Camera CSI interface."""
    cw = CFG["capture_width"]
    ch = CFG["capture_height"]
    dw = display_w or CFG["display_width"]
    dh = display_h or CFG["display_height"]
    fr = CFG["framerate"]
    fm = CFG["flip_method"]
    return (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width=(int){cw}, height=(int){ch}, "
        f"format=(string)NV12, framerate=(fraction){fr}/1 ! "
        f"nvvidconv flip-method={fm} ! "
        f"video/x-raw, width=(int){dw}, height=(int){dh}, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! "
        f"appsink max-buffers=1 drop=true"
    )


def open_camera(display_w=None, display_h=None):
    """Open RPi camera via GStreamer; fall back to USB cam if unavailable."""
    cap = cv2.VideoCapture(build_pipeline(display_w, display_h),
                           cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print("[CAM] RPi Camera opened via GStreamer.")
        return cap
    print("[WARN] GStreamer failed — trying USB camera (index 0)...")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("[CAM] USB camera opened.")
        return cap
    print("[ERROR] No camera found.")
    return None


# ═══════════════════════════════════════════════════════════════════
#  SECTION 4 — GPU / CPU ACCELERATION
# ═══════════════════════════════════════════════════════════════════
def _check_cuda():
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            cv2.cuda.setDevice(0)
            print("[CUDA] GPU acceleration enabled.")
            return True
    except AttributeError:
        pass
    print("[CUDA] Not available — using CPU (ARM NEON SIMD).")
    return False


USE_CUDA = _check_cuda()

# Pre-build morphological kernel once at startup (not per frame)
_mk = CFG["morph_ksize"]
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_mk, _mk))

USE_CUDA_MORPH = False
if USE_CUDA:
    try:
        _GPU_OPEN   = cv2.cuda.createMorphologyFilter(
            cv2.MORPH_OPEN,   cv2.CV_8UC1, MORPH_KERNEL,
            iterations=CFG["morph_open_iter"])
        _GPU_DILATE = cv2.cuda.createMorphologyFilter(
            cv2.MORPH_DILATE, cv2.CV_8UC1, MORPH_KERNEL,
            iterations=CFG["morph_dilate_iter"])
        USE_CUDA_MORPH = True
    except Exception:
        pass


def preprocess(frame):
    """Gaussian blur + BGR→HSV. Runs on GPU if available, else CPU."""
    k = CFG["blur_ksize"]
    if USE_CUDA:
        gm = cv2.cuda_GpuMat()
        gm.upload(frame)
        gm = cv2.cuda.createGaussianFilter(
            cv2.CV_8UC3, cv2.CV_8UC3, (k, k), 0).apply(gm)
        gm = cv2.cuda.cvtColor(gm, cv2.COLOR_BGR2HSV)
        return gm.download()
    blurred = cv2.GaussianBlur(frame, (k, k), 0)
    return cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)


def morphology(mask):
    """MORPH_OPEN then MORPH_DILATE. GPU if possible, else CPU."""
    if USE_CUDA_MORPH:
        gm = cv2.cuda_GpuMat()
        gm.upload(mask)
        gm = _GPU_OPEN.apply(gm)
        gm = _GPU_DILATE.apply(gm)
        return gm.download()
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,   MORPH_KERNEL,
                            iterations=CFG["morph_open_iter"])
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, MORPH_KERNEL,
                            iterations=CFG["morph_dilate_iter"])
    return mask


# ═══════════════════════════════════════════════════════════════════
#  SECTION 5 — CENTROID TRACKER
# ═══════════════════════════════════════════════════════════════════
class CentroidTracker:
    """
    Per-colour EMA centroid smoother.
    Eliminates the jittery bounding-box positions caused by mask noise.
    """
    def __init__(self, alpha=0.35):
        self.alpha = alpha
        self._pos  = {}          # colour_name → (float cx, float cy)

    def update(self, name, cx, cy):
        if name not in self._pos:
            self._pos[name] = (float(cx), float(cy))
        else:
            px, py = self._pos[name]
            self._pos[name] = (
                self.alpha * cx + (1 - self.alpha) * px,
                self.alpha * cy + (1 - self.alpha) * py,
            )
        return int(self._pos[name][0]), int(self._pos[name][1])

    def purge(self, active):
        """Remove entries for colours not seen in the current frame."""
        for k in list(self._pos):
            if k not in active:
                del self._pos[k]


_tracker = CentroidTracker(CFG["track_alpha"])


# ═══════════════════════════════════════════════════════════════════
#  SECTION 6 — DISTANCE ESTIMATOR
# ═══════════════════════════════════════════════════════════════════
def estimate_distance(pixel_width):
    """Rough depth estimate (cm) via pinhole camera model."""
    if pixel_width > 0 and CFG["known_object_cm"] > 0:
        return (CFG["known_object_cm"] * CFG["focal_px"]) / pixel_width
    return None


# ═══════════════════════════════════════════════════════════════════
#  SECTION 7 — CSV LOGGER
# ═══════════════════════════════════════════════════════════════════
class Logger:
    def __init__(self, path, enabled):
        self.enabled = enabled
        self._f = self._w = None
        if enabled:
            is_new = not os.path.exists(path)
            self._f = open(path, "a", newline="")
            self._w = csv.writer(self._f)
            if is_new:
                self._w.writerow(
                    ["timestamp", "colour", "cx", "cy",
                     "area_px2", "coverage_pct", "distance_cm"])

    def log(self, detections):
        if not self.enabled or not self._w:
            return
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        for d in detections:
            dist = f'{d["distance_cm"]:.1f}' if d["distance_cm"] else "N/A"
            self._w.writerow([ts, d["name"],
                              d["smooth_center"][0], d["smooth_center"][1],
                              d["area"], d["coverage"], dist])
        self._f.flush()

    def close(self):
        if self._f:
            self._f.close()


# ═══════════════════════════════════════════════════════════════════
#  SECTION 8 — CORE DETECTION ENGINE
# ═══════════════════════════════════════════════════════════════════
def detect_colors(frame):
    """
    Run full colour detection pipeline on one frame.

    Returns
    -------
    annotated : np.ndarray   Frame with bounding boxes + labels drawn.
    detected  : list[dict]   One dict per detected object (see keys below).

    Detection dict keys
    -------------------
    name, bgr, bbox (x,y,w,h), raw_center, smooth_center,
    area, coverage, contour, distance_cm
    """
    fh, fw    = frame.shape[:2]
    frame_area = fh * fw
    min_area   = frame_area * CFG["min_area_frac"]

    hsv     = preprocess(frame)

    # claimed mask: pixels already assigned to a colour are zeroed out
    # for all subsequent colours, preventing double-labelling.
    claimed = np.zeros((fh, fw), dtype=np.uint8)
    detected = []

    for name, cfg in COLOR_RANGES.items():
        # Build colour mask
        mask = cv2.inRange(hsv, cfg["lower1"], cfg["upper1"])
        if cfg.get("dual"):
            mask = cv2.bitwise_or(
                mask, cv2.inRange(hsv, cfg["lower2"], cfg["upper2"]))

        # Remove already-claimed pixels
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(claimed))

        # Morphological cleanup
        mask = morphology(mask)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            scx, scy   = _tracker.update(name, x + w // 2, y + h // 2)
            coverage   = round((area / frame_area) * 100, 2)
            dist       = estimate_distance(w) if CFG["show_distance"] else None

            detected.append({
                "name":          name,
                "bgr":           cfg["bgr"],
                "bbox":          (x, y, w, h),
                "raw_center":    (x + w // 2, y + h // 2),
                "smooth_center": (scx, scy),
                "area":          int(area),
                "coverage":      coverage,
                "contour":       cnt,
                "distance_cm":   dist,
            })
            cv2.drawContours(claimed, [cnt], -1, 255, cv2.FILLED)

    _tracker.purge({d["name"] for d in detected})

    # ── Annotate ─────────────────────────────────────────────────
    out = frame.copy()
    for d in detected:
        x, y, w, h = d["bbox"]
        bgr = d["bgr"]

        if CFG["show_contour"]:
            cv2.drawContours(out, [d["contour"]], -1, bgr, 1)

        cv2.rectangle(out, (x, y), (x + w, y + h), bgr, 2)

        if d["distance_cm"] and CFG["show_distance"]:
            label = f'{d["name"]}  {d["coverage"]}%  ~{d["distance_cm"]:.0f}cm'
        else:
            label = f'{d["name"]}  {d["coverage"]}%'

        (tw, th), _ = cv2.getTextSize(label, FONT, 0.52, 1)
        ly = max(y, th + 12)
        cv2.rectangle(out, (x, ly - th - 8), (x + tw + 8, ly), bgr, cv2.FILLED)
        cv2.putText(out, label, (x + 4, ly - 4),
                    FONT, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.drawMarker(out, d["smooth_center"], bgr,
                       cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)

    return out, detected


# ═══════════════════════════════════════════════════════════════════
#  SECTION 9 — HUD OVERLAY  (detect mode)
# ═══════════════════════════════════════════════════════════════════
def draw_hud(frame, fps, detected, paused=False):
    h, w   = frame.shape[:2]
    overlay = frame.copy()

    unique = list(dict.fromkeys(d["name"] for d in detected))
    panel_h = 56 + len(unique) * 22 + 10
    cv2.rectangle(overlay, (0, 0), (230, panel_h), (15, 15, 15), cv2.FILLED)

    if paused:
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 200), 4)
        cv2.putText(overlay, "PAUSED  —  P to resume",
                    (w // 2 - 140, 32), FONT, 0.7, (60, 60, 255), 2, cv2.LINE_AA)

    cv2.addWeighted(overlay, CFG["hud_alpha"],
                    frame,   1 - CFG["hud_alpha"], 0, frame)

    # FPS — colour-coded: green ≥25, orange ≥15, red <15
    fc = (0, 220, 90) if fps >= 25 else (0, 165, 255) if fps >= 15 else (0, 0, 220)
    cv2.putText(frame, f"FPS: {fps:.1f}", (8, 22),
                FONT, 0.6, fc, 1, cv2.LINE_AA)

    mode = "GPU" if USE_CUDA else "CPU"
    cv2.putText(frame, mode, (8, 42), FONT, 0.42, (160, 160, 160), 1, cv2.LINE_AA)

    # Detected colour swatches
    panel_y = 58
    for name in unique:
        bgr = COLOR_RANGES[name]["bgr"]
        cv2.rectangle(frame, (8, panel_y), (20, panel_y + 13), bgr, cv2.FILLED)
        cv2.putText(frame, name, (28, panel_y + 11),
                    FONT, 0.42, (215, 215, 215), 1, cv2.LINE_AA)
        panel_y += 22

    cv2.putText(frame, "Q=Quit  S=Screenshot  P=Pause  C=Contour  T=Tune",
                (8, h - 8), FONT, 0.36, (130, 130, 130), 1, cv2.LINE_AA)
    return frame


# ═══════════════════════════════════════════════════════════════════
#  SECTION 10 — CONSOLE LOG  (detect mode)
# ═══════════════════════════════════════════════════════════════════
_frame_n = 0

def console_log(fps, detected):
    global _frame_n
    _frame_n += 1
    if _frame_n % 15:
        return
    if not detected:
        print(f"\r[{fps:5.1f} fps]  —  nothing detected          ", end="")
        return
    parts = []
    for d in detected:
        dist = f' ~{d["distance_cm"]:.0f}cm' if d["distance_cm"] else ""
        parts.append(f'{d["name"]}({d["coverage"]}%{dist})')
    print(f"\r[{fps:5.1f} fps]  {", ".join(parts)}   ", end="")


# ═══════════════════════════════════════════════════════════════════
#  SECTION 11 — MODE A: DETECT
# ═══════════════════════════════════════════════════════════════════
def run_detect():
    """Main real-time colour detection loop."""
    print("\n" + "═" * 60)
    print("  MODE: Detect   (press T inside window to switch to Tune)")
    print("═" * 60)

    cap = open_camera()
    if cap is None:
        return

    logger     = Logger(CFG["csv_path"], CFG["enable_csv_log"])
    prev_t     = time.time()
    fps        = 0.0
    shot_idx   = 0
    paused     = False
    frozen     = None

    cv2.namedWindow("Colour Detection", cv2.WINDOW_AUTOSIZE)

    while True:
        # ── Paused state ─────────────────────────────────────────
        if paused and frozen is not None:
            cv2.imshow("Colour Detection", frozen)
            k = cv2.waitKey(30) & 0xFF
            if k in (ord("p"), ord("P")):
                paused = False
            elif k in (ord("q"), 27):
                break
            elif k in (ord("s"), ord("S")):
                fn = f"screenshot_{shot_idx:03d}.jpg"
                cv2.imwrite(fn, frozen)
                print(f"\n[INFO] Saved: {fn}")
                shot_idx += 1
            continue

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        annotated, detected = detect_colors(frame)

        now   = time.time()
        fps   = 0.1 * (1.0 / max(now - prev_t, 1e-6)) + 0.9 * fps
        prev_t = now

        output = draw_hud(annotated, fps, detected)
        logger.log(detected)
        console_log(fps, detected)

        cv2.imshow("Colour Detection", output)
        k = cv2.waitKey(1) & 0xFF

        if k in (ord("q"), 27):
            break
        elif k in (ord("s"), ord("S")):
            fn = f"screenshot_{shot_idx:03d}.jpg"
            cv2.imwrite(fn, output)
            print(f"\n[INFO] Saved: {fn}")
            shot_idx += 1
        elif k in (ord("p"), ord("P")):
            paused = True
            frozen = draw_hud(output.copy(), fps, detected, paused=True)
            print("\n[INFO] Paused.")
        elif k in (ord("c"), ord("C")):
            CFG["show_contour"] = not CFG["show_contour"]
            print(f"\n[INFO] Contour {'ON' if CFG['show_contour'] else 'OFF'}.")
        elif k in (ord("t"), ord("T")):
            # Switch to tuner without restarting
            print("\n[INFO] Switching to Tuner mode...")
            cap.release()
            cv2.destroyAllWindows()
            logger.close()
            run_tune()
            return

    cap.release()
    logger.close()
    cv2.destroyAllWindows()
    print("\n[INFO] Done.")


# ═══════════════════════════════════════════════════════════════════
#  SECTION 12 — MODE B: HSV TUNER
#
#  Shows three panels side-by-side: Original | Mask | Result
#  Drag the six trackbars to isolate any colour.
#  Press S to print the HSV values you can paste into COLOR_RANGES.
#  Press A to apply them live to a named colour entry.
#  Press D to switch back to Detect mode.
# ═══════════════════════════════════════════════════════════════════
def run_tune():
    """Interactive HSV calibration tool."""
    print("\n" + "═" * 60)
    print("  MODE: HSV Tuner   (press D inside window to switch to Detect)")
    print("  Drag trackbars → isolate target colour in Mask panel")
    print("  S = print values   A = apply to named colour   Q = quit")
    print("═" * 60)

    # Tuner uses a slightly smaller display to fit three panels on screen
    cap = open_camera(display_w=640, display_h=360)
    if cap is None:
        return

    WIN = "HSV Tuner  |  Original · Mask · Result"
    CTL = "Tuner Controls"
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(CTL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CTL, 420, 220)

    def _noop(_): pass

    # Create trackbars with correct per-channel maximums
    bars = [
        ("H Low",  "H_Low",  0,   180),
        ("S Low",  "S_Low",  0,   255),
        ("V Low",  "V_Low",  0,   255),
        ("H High", "H_High", 180, 180),
        ("S High", "S_High", 255, 255),
        ("V High", "V_High", 255, 255),
    ]
    for label, key, default, maxval in bars:
        cv2.createTrackbar(label, CTL, default, maxval, _noop)

    colour_names = list(COLOR_RANGES.keys())

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        hl = cv2.getTrackbarPos("H Low",  CTL)
        sl = cv2.getTrackbarPos("S Low",  CTL)
        vl = cv2.getTrackbarPos("V Low",  CTL)
        hh = cv2.getTrackbarPos("H High", CTL)
        sh = cv2.getTrackbarPos("S High", CTL)
        vh = cv2.getTrackbarPos("V High", CTL)

        lower = np.array([hl, sl, vl])
        upper = np.array([hh, sh, vh])

        mask   = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(frame, frame, mask=mask)

        # Three-panel display: Original | Mask | Result
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        panel_w, panel_h = 426, 240
        combined = np.hstack([
            cv2.resize(frame,    (panel_w, panel_h)),
            cv2.resize(mask_3ch, (panel_w, panel_h)),
            cv2.resize(result,   (panel_w, panel_h)),
        ])

        # Label headers
        for i, label in enumerate(["Original", "Mask", "Result"]):
            cv2.putText(combined, label,
                        (i * panel_w + 10, 22),
                        FONT, 0.65, (255, 255, 0), 1, cv2.LINE_AA)

        # Current HSV range readout at bottom
        info = (f"lower=[{hl},{sl},{vl}]   upper=[{hh},{sh},{vh}]   "
                f"S=print  A=apply  D=detect  Q=quit")
        cv2.putText(combined, info, (10, panel_h - 8),
                    FONT, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow(WIN, combined)
        k = cv2.waitKey(1) & 0xFF

        if k in (ord("q"), 27):
            break

        elif k in (ord("s"), ord("S")):
            # Print values the user can copy-paste into COLOR_RANGES
            print(f'\n    "lower1": np.array([{hl}, {sl}, {vl}]),')
            print(f'    "upper1": np.array([{hh}, {sh}, {vh}]),')
            print()

        elif k in (ord("a"), ord("A")):
            # Apply the current trackbar range to a named colour live
            print("\n  Existing colour names:")
            for i, n in enumerate(colour_names):
                print(f"    {i}: {n}")
            print("  Enter number (or new name to add): ", end="", flush=True)
            try:
                inp = input().strip()
                if inp.isdigit():
                    target = colour_names[int(inp)]
                else:
                    target = inp
                    if target not in COLOR_RANGES:
                        COLOR_RANGES[target] = {
                            "bgr": (128, 128, 128), "dual": False}
                        colour_names.append(target)
                COLOR_RANGES[target]["lower1"] = lower.copy()
                COLOR_RANGES[target]["upper1"] = upper.copy()
                COLOR_RANGES[target]["dual"]   = False
                print(f"  [OK] Applied to '{target}'. "
                      f"Switch to Detect (D) to test it.")
            except (IndexError, ValueError, EOFError):
                print("  [WARN] Invalid input, no change made.")

        elif k in (ord("d"), ord("D")):
            print("\n[INFO] Switching to Detect mode...")
            cap.release()
            cv2.destroyAllWindows()
            run_detect()
            return

    cap.release()
    cv2.destroyAllWindows()
    print("\n[INFO] Tuner closed.")


# ═══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "detect"

    print("╔══════════════════════════════════════════════════════╗")
    print("║  Colour Detection  v4  —  Jetson Nano + RPi Camera  ║")
    print("╚══════════════════════════════════════════════════════╝")

    if mode == "tune":
        run_tune()
    elif mode == "detect":
        run_detect()
    else:
        print(f"[ERROR] Unknown mode '{mode}'. Use:  detect  or  tune")
        sys.exit(1)