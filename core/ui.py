import cv2
import numpy as np

# ─── Palette constants ─────────────────────────────────────────────────────────
PALETTE = [
    ("Red",    (0,   0,   255)),
    ("Orange", (0,   140, 255)),
    ("Yellow", (0,   220, 255)),
    ("Green",  (0,   200, 80 )),
    ("Blue",   (255, 80,  30 )),
    ("Purple", (200, 0,   200)),
    ("White",  (255, 255, 255)),
]
PAL_Y    = 45
PAL_X0   = 35
PAL_STEP = 62
PAL_R    = 22


def hit_palette(tip_x, tip_y):
    """Returns palette index if finger is hovering a colour swatch, else -1."""
    if abs(tip_y - PAL_Y) > PAL_R + 8:
        return -1
    for i in range(len(PALETTE)):
        if abs(tip_x - (PAL_X0 + i * PAL_STEP)) <= PAL_R + 8:
            return i
    return -1


def blend_canvas(frame, canvas):
    """
    Non-black canvas pixels show as drawing.
    Black (zero) canvas pixels are transparent — camera shows through.
    """
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    bg = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
    fg = cv2.bitwise_and(canvas, canvas, mask=mask)
    return cv2.add(bg, fg)


def render_canvas(strokes, current_stroke, h, w):
    """Re-render all strokes onto a fresh black canvas every frame."""
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for stroke in strokes:
        stroke.draw_on(canvas)
    if current_stroke:
        current_stroke.draw_on(canvas)
    return canvas


def draw_ui(frame, color_idx, eraser_mode, fist_count, fist_clear_frames, gesture, selected_stroke):
    h, w = frame.shape[:2]

    # ── Colour palette ────────────────────────────────────────────────────────
    for i, (_, bgr) in enumerate(PALETTE):
        cx = PAL_X0 + i * PAL_STEP
        cv2.circle(frame, (cx, PAL_Y), PAL_R, bgr, -1)
        cv2.circle(frame, (cx, PAL_Y), PAL_R, (40, 40, 40), 1)
        if i == color_idx and not eraser_mode:
            cv2.circle(frame, (cx, PAL_Y), PAL_R + 5, (255, 255, 255), 2)

    # ── Mode label ────────────────────────────────────────────────────────────
    labels = {
        "ERASER":  ("[ ERASER ]",           (60,  60,  255)),
        "NEUTRAL": ("[ HOVER — safe ]",      (200, 200,   0)),
        "PINCH":   ("[ MOVING ]",            (0,   200, 255)),
        "FIST":    ("[ FIST — hold clear ]", (0,   100, 255)),
        "DRAW":    ("[ DRAW ]",              (60,  220,  60)),
        "IDLE":    ("[ IDLE ]",              (120, 120, 120)),
    }
    lbl_key   = "ERASER" if eraser_mode else gesture
    text, col = labels.get(lbl_key, ("", (255, 255, 255)))
    cv2.putText(frame, text, (w - 270, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

    # ── Selected stroke highlight ─────────────────────────────────────────────
    if selected_stroke:
        b = selected_stroke.bounds()
        if b:
            x1, y1, x2, y2 = b
            pad = 12
            cv2.rectangle(frame,
                          (x1 - pad, y1 - pad),
                          (x2 + pad, y2 + pad),
                          (0, 220, 255), 2)
            cv2.putText(frame, "moving", (x1 - pad, y1 - pad - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)

    # ── Fist clear progress bar ───────────────────────────────────────────────
    if fist_count > 0:
        bx, by = w // 2 - 110, h - 40
        bw     = int((fist_count / fist_clear_frames) * 220)
        cv2.rectangle(frame, (bx, by), (bx + 220, by + 20), (50,  50,  50),  -1)
        cv2.rectangle(frame, (bx, by), (bx + bw,  by + 20), (0,   100, 255), -1)
        cv2.rectangle(frame, (bx, by), (bx + 220, by + 20), (180, 180, 180),  1)
        cv2.putText(frame, "HOLD FIST TO CLEAR", (bx + 28, by - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # ── Gesture guide ─────────────────────────────────────────────────────────
    guide = [
        "1 finger  : draw",
        "2 close   : erase",
        "2 spread  : hover",
        "pinch     : move stroke",
        "fist 1sec : clear all",
    ]
    for j, line in enumerate(guide):
        cv2.putText(frame, line, (w - 185, 65 + j * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, (140, 140, 140), 1)

    # ── Controls hint ─────────────────────────────────────────────────────────
    cv2.putText(frame, "Ctrl+Z: Undo  |  Ctrl+S: Save PNG  |  Q/ESC: Quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1)