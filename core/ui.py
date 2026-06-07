import cv2
import numpy as np
import math

# ─── Palette ───────────────────────────────────────────────────────────────────
PALETTE = [
    ("Red",    (0,   0,   255)),
    ("Orange", (0,   140, 255)),
    ("Yellow", (0,   220, 255)),
    ("Green",  (0,   200, 80 )),
    ("Blue",   (255, 80,  30 )),
    ("Purple", (200, 0,   200)),
    ("White",  (255, 255, 255)),
]
PAL_Y    = 38
PAL_X0   = 35
PAL_STEP = 58
PAL_R    = 18

# ─── SIZE button ───────────────────────────────────────────────────────────────
# Sits right after the last palette swatch on the same row
BTN_CX = PAL_X0 + len(PALETTE) * PAL_STEP + 42   # center x  ≈ 485
BTN_Y  = PAL_Y                                     # same row
BTN_HW = 36    # half-width  → button is 72px wide
BTN_HH = 18    # half-height → button is 36px tall

# ─── Sliding bar ───────────────────────────────────────────────────────────────
BAR_X0   = BTN_CX + BTN_HW + 14   # bar starts just right of button  ≈ 535
BAR_MAXW = 420                     # fully-open bar width (reaches ≈ 955)
BAR_Y    = PAL_Y                   # same row as button / palette
BAR_HIT  = 20                      # hit zone half-height


def hit_palette(tip_x, tip_y):
    if abs(tip_y - PAL_Y) > PAL_R + 8:
        return -1
    for i in range(len(PALETTE)):
        if abs(tip_x - (PAL_X0 + i * PAL_STEP)) <= PAL_R + 8:
            return i
    return -1


def hit_size_button(tip_x, tip_y):
    """True if finger is over the SIZE toggle button."""
    return abs(tip_x - BTN_CX) <= BTN_HW + 6 and abs(tip_y - BTN_Y) <= BTN_HH + 6


def hit_brush_bar(tip_x, tip_y, bar_anim):
    """
    Returns size 1-100 if finger is over the currently drawn portion of the bar.
    Only registers when bar is mostly open (anim > 0.6).
    """
    if bar_anim < 0.6:
        return -1
    current_right = BAR_X0 + int(bar_anim * BAR_MAXW)
    if not (BAR_X0 <= tip_x <= current_right):
        return -1
    if abs(tip_y - BAR_Y) > BAR_HIT:
        return -1
    # Map position across the FULL bar range so size is consistent
    return max(1, min(100, int(np.interp(tip_x, [BAR_X0, BAR_X0 + BAR_MAXW], [1, 100]))))


def blend_canvas(frame, canvas):
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    bg = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
    fg = cv2.bitwise_and(canvas, canvas, mask=mask)
    return cv2.add(bg, fg)


def render_canvas(strokes, current_stroke, h, w):
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for stroke in strokes:
        stroke.draw_on(canvas)
    if current_stroke:
        current_stroke.draw_on(canvas)
    return canvas


def _draw_size_button(frame, brush_size, bar_open):
    """Draws the SIZE toggle button."""
    x1 = BTN_CX - BTN_HW
    y1 = BTN_Y  - BTN_HH
    x2 = BTN_CX + BTN_HW
    y2 = BTN_Y  + BTN_HH
    bg_col  = (60, 60, 60)   if not bar_open else (80, 80, 80)
    bdr_col = (120, 120, 120) if not bar_open else (0, 200, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), bg_col,  -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), bdr_col,  1)
    cv2.putText(frame, "SIZE",
                (x1 + 8, BTN_Y - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
    cv2.putText(frame, str(brush_size),
                (x1 + 18, BTN_Y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)


def _draw_sliding_bar(frame, brush_size, bar_anim, current_color):
    """Draws the animated brush size bar — slides out to the right of the button."""
    if bar_anim <= 0.01:
        return

    current_right = BAR_X0 + int(bar_anim * BAR_MAXW)
    full_right    = BAR_X0 + BAR_MAXW

    # Dark background panel (clips to current animation extent)
    cv2.rectangle(frame,
                  (BAR_X0 - 4, BAR_Y - BAR_HIT),
                  (current_right + 4, BAR_Y + BAR_HIT),
                  (30, 30, 30), -1)

    # Grey track
    cv2.line(frame, (BAR_X0, BAR_Y), (current_right, BAR_Y), (70, 70, 70), 2)

    # Coloured fill up to selected position
    fill_x = int(np.interp(brush_size, [1, 100], [BAR_X0, full_right]))
    fill_x = min(fill_x, current_right)   # don't draw past animation front
    if fill_x > BAR_X0:
        cv2.line(frame, (BAR_X0, BAR_Y), (fill_x, BAR_Y), current_color, 3)

    # Position indicator dot (only when bar is open enough to show it)
    if fill_x <= current_right - 4:
        cv2.circle(frame, (fill_x, BAR_Y), 7, current_color, -1)
        cv2.circle(frame, (fill_x, BAR_Y), 7, (255, 255, 255), 1)

    # Scale labels (fade in with animation)
    if bar_anim > 0.4:
        cv2.putText(frame, "1",
                    (BAR_X0, BAR_Y - BAR_HIT + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 100, 100), 1)
    if bar_anim > 0.9:
        cv2.putText(frame, "100",
                    (full_right - 18, BAR_Y - BAR_HIT + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 100, 100), 1)


def draw_ui(frame, color_idx, eraser_mode, brush_size, bar_open, bar_anim,
            fist_count, fist_clear_frames, gesture, selected_stroke,
            is_rotating=False):
    h, w = frame.shape[:2]

    # ── Header panel ──────────────────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 72), (20, 20, 20), -1)

    # ── Colour palette ────────────────────────────────────────────────────────
    for i, (_, bgr) in enumerate(PALETTE):
        cx = PAL_X0 + i * PAL_STEP
        cv2.circle(frame, (cx, PAL_Y), PAL_R, bgr, -1)
        cv2.circle(frame, (cx, PAL_Y), PAL_R, (55, 55, 55), 1)
        if i == color_idx and not eraser_mode:
            cv2.circle(frame, (cx, PAL_Y), PAL_R + 4, (255, 255, 255), 2)

    # ── SIZE button ───────────────────────────────────────────────────────────
    _draw_size_button(frame, brush_size, bar_open)

    # ── Sliding bar ───────────────────────────────────────────────────────────
    current_color = PALETTE[color_idx][1] if not eraser_mode else (150, 150, 150)
    _draw_sliding_bar(frame, brush_size, bar_anim, current_color)

    # ── Mode label ────────────────────────────────────────────────────────────
    labels = {
        "ERASER":  ("[ ERASER ]",       (60,  60,  255)),
        "NEUTRAL": ("[ HOVER ]",         (200, 200,   0)),
        "PINCH":   ("[ MOVING ]",        (0,   200, 255)),
        "FIST":    ("[ HOLD CLEAR ]",    (0,   100, 255)),
        "DRAW":    ("[ DRAW ]",          (60,  220,  60)),
        "IDLE":    ("[ IDLE ]",          (120, 120, 120)),
    }
    lbl_key   = "ERASER" if eraser_mode else gesture
    text, col = labels.get(lbl_key, ("", (255, 255, 255)))
    if is_rotating:
        text = text + " [ ROTATING ]"
        col  = (255, 165, 0)
    cv2.putText(frame, text, (w - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

    # ── Selected stroke box ───────────────────────────────────────────────────
    if selected_stroke:
        b = selected_stroke.bounds()
        if b:
            x1, y1, x2, y2 = b
            pad       = 12
            box_color = (255, 165, 0) if is_rotating else (0, 220, 255)
            angle_deg = int(math.degrees(selected_stroke._angle) % 360)
            box_label = f"rotating {angle_deg}°" if is_rotating else "moving"
            cv2.rectangle(frame, (x1-pad, y1-pad), (x2+pad, y2+pad), box_color, 2)
            cv2.putText(frame, box_label, (x1-pad, y1-pad-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1)

    # ── Fist clear progress bar ───────────────────────────────────────────────
    if fist_count > 0:
        bx, by = w // 2 - 110, h - 40
        bw     = int((fist_count / fist_clear_frames) * 220)
        cv2.rectangle(frame, (bx, by), (bx+220, by+20), (50,  50,  50),  -1)
        cv2.rectangle(frame, (bx, by), (bx+bw,  by+20), (0,   100, 255), -1)
        cv2.rectangle(frame, (bx, by), (bx+220, by+20), (180, 180, 180),  1)
        cv2.putText(frame, "HOLD FIST TO CLEAR", (bx+28, by-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # ── Gesture guide ─────────────────────────────────────────────────────────
    guide = [
        "1 finger  : draw",
        "2 close   : erase",
        "2 spread  : hover",
        "pinch     : move stroke",
        "pinch+2nd : rotate",
        "fist 1sec : clear all",
    ]
    for j, line in enumerate(guide):
        cv2.putText(frame, line, (w - 200, h - 165 + j * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)

    # ── Controls hint ─────────────────────────────────────────────────────────
    cv2.putText(frame, "Ctrl+Z: Undo  |  Ctrl+S: Save  |  Q/ESC: Quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (110, 110, 110), 1)