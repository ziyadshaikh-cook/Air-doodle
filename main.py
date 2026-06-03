import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from datetime import datetime
import copy

# ─── Configuration ─────────────────────────────────────────────────────────────
BRUSH_THICKNESS   = 6
ERASER_THICKNESS  = 50
FIST_CLEAR_FRAMES = 30
PINCH_THRESHOLD   = 40
PINCH_DEAD_ZONE   = 3
MIN_DRAW_DIST     = 4
MAX_DRAW_DIST     = 80
NEUTRAL_SPREAD    = 55
SMOOTH_BUFFER     = 5
MAX_UNDO          = 25
SELECT_RADIUS     = 70   # px padding around stroke bounding box for pinch selection

PALETTE = [
    ("Red",    (0,   0,   255)),
    ("Orange", (0,   140, 255)),
    ("Yellow", (0,   220, 255)),
    ("Green",  (0,   200, 80 )),
    ("Blue",   (255, 80,  30 )),
    ("Purple", (200, 0,   200)),
    ("White",  (255, 255, 255)),
]
PAL_Y = 45; PAL_X0 = 35; PAL_STEP = 62; PAL_R = 22

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils


# ─── Stroke class ──────────────────────────────────────────────────────────────
# Each drawn stroke is an independent object with its own point list.
# Pinch selects and moves one stroke at a time — no more whole-canvas shift.

class Stroke:
    def __init__(self, color, thickness, is_eraser=False):
        self.points    = []
        self.color     = color
        self.thickness = thickness
        self.is_eraser = is_eraser

    def add_point(self, pt):
        self.points.append(pt)

    def bounds(self):
        if not self.points:
            return None
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return min(xs), min(ys), max(xs), max(ys)

    def translate(self, dx, dy):
        self.points = [(p[0] + dx, p[1] + dy) for p in self.points]

    def draw_on(self, canvas):
        if not self.points:
            return
        color = (0, 0, 0) if self.is_eraser else self.color
        if len(self.points) == 1:
            cv2.circle(canvas, self.points[0], self.thickness // 2, color, -1)
            return
        for i in range(1, len(self.points)):
            cv2.line(canvas, self.points[i - 1], self.points[i], color, self.thickness)
        cv2.circle(canvas, self.points[-1], self.thickness // 2, color, -1)


# ─── Helpers ───────────────────────────────────────────────────────────────────

def lm_px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)

def finger_up(lms, tip, pip):
    return lms[tip].y < lms[pip].y

def smooth_point(buf, pt):
    buf.append(pt)
    return int(np.mean([p[0] for p in buf])), int(np.mean([p[1] for p in buf]))

def push_undo(stack, strokes):
    stack.append(copy.deepcopy(strokes))
    if len(stack) > MAX_UNDO:
        stack.pop(0)

def finalize_stroke(strokes, current_stroke):
    """Append current stroke to strokes list if it has points. Returns None."""
    if current_stroke and current_stroke.points:
        strokes.append(current_stroke)
    return None

def detect_gesture(lms, w, h):
    """
    Priority order (first match wins):
      FIST    — all fingertips below PIP joint → hold 1 sec to clear
      PINCH   — thumb close to index → move individual stroke
      NEUTRAL — index + middle up, spread wide → safe hover
      ERASER  — index + middle up, close together → duster erase
      DRAW    — only index up → draw
      IDLE    — everything else
    """
    idx_up  = finger_up(lms, 8,  6)
    mid_up  = finger_up(lms, 12, 10)
    ring_up = finger_up(lms, 16, 14)
    pky_up  = finger_up(lms, 20, 18)

    tx, ty       = lm_px(lms[4], w, h)
    ix, iy       = lm_px(lms[8], w, h)
    pinch_dist   = np.hypot(tx - ix, ty - iy)
    pinch_center = ((tx + ix) // 2, (ty + iy) // 2)

    mx, my = lm_px(lms[12], w, h)
    spread = np.hypot(ix - mx, iy - my)

    is_fist = all(
        lms[t].y > lms[p].y
        for t, p in [(8, 6), (12, 10), (16, 14), (20, 18)]
    )

    if is_fist:
        return "FIST", None
    elif pinch_dist < PINCH_THRESHOLD:
        return "PINCH", pinch_center
    elif idx_up and mid_up and not ring_up and not pky_up and spread > NEUTRAL_SPREAD:
        return "NEUTRAL", None
    elif idx_up and mid_up and not ring_up and not pky_up and spread <= NEUTRAL_SPREAD:
        return "ERASER", None
    elif idx_up and not mid_up:
        return "DRAW", None
    else:
        return "IDLE", None

def find_stroke_at(strokes, px, py):
    """
    Returns the topmost non-eraser stroke whose bounding box
    (expanded by SELECT_RADIUS) contains the point (px, py).
    Iterates in reverse so the most recently drawn stroke wins.
    """
    for stroke in reversed(strokes):
        if stroke.is_eraser or not stroke.points:
            continue
        b = stroke.bounds()
        if not b:
            continue
        x1, y1, x2, y2 = b
        if (x1 - SELECT_RADIUS <= px <= x2 + SELECT_RADIUS and
                y1 - SELECT_RADIUS <= py <= y2 + SELECT_RADIUS):
            return stroke
    return None

def render_canvas(strokes, current_stroke, h, w):
    """Re-render all strokes onto a fresh black canvas every frame."""
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for stroke in strokes:
        stroke.draw_on(canvas)
    if current_stroke:
        current_stroke.draw_on(canvas)
    return canvas

def hit_palette(tip_x, tip_y):
    if abs(tip_y - PAL_Y) > PAL_R + 8:
        return -1
    for i in range(len(PALETTE)):
        if abs(tip_x - (PAL_X0 + i * PAL_STEP)) <= PAL_R + 8:
            return i
    return -1

def blend_canvas(frame, canvas):
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    bg = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
    fg = cv2.bitwise_and(canvas, canvas, mask=mask)
    return cv2.add(bg, fg)

def draw_ui(frame, color_idx, eraser_mode, fist_count, gesture, selected_stroke):
    h, w = frame.shape[:2]

    # Palette
    for i, (_, bgr) in enumerate(PALETTE):
        cx = PAL_X0 + i * PAL_STEP
        cv2.circle(frame, (cx, PAL_Y), PAL_R, bgr, -1)
        cv2.circle(frame, (cx, PAL_Y), PAL_R, (40, 40, 40), 1)
        if i == color_idx and not eraser_mode:
            cv2.circle(frame, (cx, PAL_Y), PAL_R + 5, (255, 255, 255), 2)

    # Mode label
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

    # Highlight selected stroke with a bounding box
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

    # Fist clear progress bar
    if fist_count > 0:
        bx, by = w // 2 - 110, h - 40
        bw     = int((fist_count / FIST_CLEAR_FRAMES) * 220)
        cv2.rectangle(frame, (bx, by), (bx + 220, by + 20), (50,  50,  50),  -1)
        cv2.rectangle(frame, (bx, by), (bx + bw,  by + 20), (0,   100, 255), -1)
        cv2.rectangle(frame, (bx, by), (bx + 220, by + 20), (180, 180, 180),  1)
        cv2.putText(frame, "HOLD FIST TO CLEAR", (bx + 28, by - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Gesture guide
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

    cv2.putText(frame, "Ctrl+Z: Undo  |  Ctrl+S: Save PNG  |  Q/ESC: Quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1)


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ret, frame = cap.read()
    if not ret:
        print("ERROR: Cannot access webcam.")
        cap.release()
        return

    h, w = frame.shape[:2]

    strokes        = []     # all completed Stroke objects
    current_stroke = None   # stroke actively being drawn right now
    undo_stack     = []

    color_idx       = 0
    eraser_mode     = False
    fist_count      = 0
    pinch_prev      = None
    selected_stroke = None  # the specific stroke being moved by pinch
    smooth_buf      = deque(maxlen=SMOOTH_BUFFER)

    with mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame  = cv2.flip(frame, 1)
            result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            lms     = None
            gesture = "IDLE"

            if result.multi_hand_landmarks:
                hand_lms = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                lms = hand_lms.landmark

            if lms:
                gesture, pinch_pt = detect_gesture(lms, w, h)

                raw_ix, raw_iy = lm_px(lms[8], w, h)
                ix, iy         = smooth_point(smooth_buf, (raw_ix, raw_iy))

                # Finalize any in-progress stroke if gesture left draw/erase
                if gesture not in ("DRAW", "ERASER") and current_stroke:
                    current_stroke = finalize_stroke(strokes, current_stroke)

                # Deselect stroke if gesture left pinch
                if gesture != "PINCH":
                    selected_stroke = None
                    pinch_prev      = None

                # ── FIST: hold to clear all strokes ──────────────────────
                if gesture == "FIST":
                    fist_count += 1
                    if fist_count >= FIST_CLEAR_FRAMES:
                        push_undo(undo_stack, strokes)
                        strokes    = []
                        fist_count = 0
                        print("[Cleared]")

                # ── PINCH: find and move individual stroke ────────────────
                elif gesture == "PINCH":
                    fist_count = 0

                    if selected_stroke is None:
                        # First frame of this pinch — find nearest stroke
                        candidate = find_stroke_at(strokes, pinch_pt[0], pinch_pt[1])
                        if candidate:
                            push_undo(undo_stack, strokes)
                            selected_stroke = candidate
                            pinch_prev      = pinch_pt
                        # If no stroke found near pinch, do nothing
                    else:
                        # Continue dragging the selected stroke
                        if pinch_prev is not None:
                            dx = pinch_pt[0] - pinch_prev[0]
                            dy = pinch_pt[1] - pinch_prev[1]
                            if abs(dx) > PINCH_DEAD_ZONE or abs(dy) > PINCH_DEAD_ZONE:
                                selected_stroke.translate(dx, dy)
                        pinch_prev = pinch_pt

                    cv2.circle(frame, pinch_pt, 12, (255, 200, 0), 2)

                # ── NEUTRAL: do nothing safely ────────────────────────────
                elif gesture == "NEUTRAL":
                    fist_count  = 0
                    eraser_mode = False

                # ── ERASER: drag two fingers to erase like a duster ───────
                elif gesture == "ERASER":
                    fist_count  = 0
                    eraser_mode = True

                    mx, my = lm_px(lms[12], w, h)
                    ex     = (ix + mx) // 2
                    ey     = (iy + my) // 2

                    # Show eraser circle on live frame
                    cv2.circle(frame, (ex, ey), ERASER_THICKNESS // 2, (80, 80, 255), 2)

                    if current_stroke is None:
                        push_undo(undo_stack, strokes)
                        current_stroke = Stroke((0, 0, 0), ERASER_THICKNESS, is_eraser=True)
                        current_stroke.add_point((ex, ey))
                    else:
                        last = current_stroke.points[-1]
                        dist = np.hypot(ex - last[0], ey - last[1])
                        if dist > MIN_DRAW_DIST:
                            current_stroke.add_point((ex, ey))

                # ── DRAW ──────────────────────────────────────────────────
                elif gesture == "DRAW":
                    fist_count  = 0
                    eraser_mode = False

                    hit = hit_palette(ix, iy)
                    if hit >= 0:
                        # Hovering palette — switch colour, stop drawing
                        color_idx      = hit
                        current_stroke = finalize_stroke(strokes, current_stroke)
                    else:
                        color = PALETTE[color_idx][1]
                        cv2.circle(frame, (ix, iy), BRUSH_THICKNESS // 2 + 2, color, -1)

                        if current_stroke is None:
                            push_undo(undo_stack, strokes)
                            current_stroke = Stroke(color, BRUSH_THICKNESS)
                            current_stroke.add_point((ix, iy))
                        else:
                            last = current_stroke.points[-1]
                            dist = np.hypot(ix - last[0], iy - last[1])
                            if MIN_DRAW_DIST < dist < MAX_DRAW_DIST:
                                current_stroke.add_point((ix, iy))
                            elif dist >= MAX_DRAW_DIST:
                                # Tracking jumped — reset without gap line
                                current_stroke.add_point((ix, iy))

                # ── IDLE ──────────────────────────────────────────────────
                else:
                    fist_count = 0

            else:
                # No hand in frame — finalize anything in progress
                current_stroke  = finalize_stroke(strokes, current_stroke)
                fist_count      = 0
                pinch_prev      = None
                selected_stroke = None
                smooth_buf.clear()

            # ── Render ────────────────────────────────────────────────────
            canvas = render_canvas(strokes, current_stroke, h, w)
            output = blend_canvas(frame, canvas)
            draw_ui(output, color_idx, eraser_mode, fist_count, gesture, selected_stroke)
            cv2.imshow("Air Doodle", output)

            # ── Keyboard ──────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):       # Q or ESC
                break

            elif key == 26:                  # Ctrl+Z
                if undo_stack:
                    strokes         = undo_stack.pop()
                    current_stroke  = None
                    selected_stroke = None
                    print(f"[Undo] {len(undo_stack)} states left")

            elif key == 19:                  # Ctrl+S
                ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"air_doodle_{ts}.png"
                save_canvas = render_canvas(strokes, None, h, w)
                gray_c      = cv2.cvtColor(save_canvas, cv2.COLOR_BGR2GRAY)
                _, smask    = cv2.threshold(gray_c, 1, 255, cv2.THRESH_BINARY)
                save_img    = np.full((h, w, 3), 255, dtype=np.uint8)
                save_img[smask > 0] = save_canvas[smask > 0]
                cv2.imwrite(fname, save_img)
                print(f"[Saved] {fname}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()