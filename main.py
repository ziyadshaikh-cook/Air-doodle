import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from datetime import datetime

# ─── Configuration ─────────────────────────────────────────────────────────────
BRUSH_THICKNESS   = 6
ERASER_THICKNESS  = 50
FIST_CLEAR_FRAMES = 30    # frames of fist held to trigger clear (~1 sec at 30fps)
PINCH_THRESHOLD   = 40    # pixel distance thumb↔index to count as pinch
PINCH_DEAD_ZONE   = 3     # min pixel delta for pinch move to register
MIN_DRAW_DIST     = 4     # min pixel move to draw (kills accidental dots when paused)
MAX_DRAW_DIST     = 80    # max pixel jump allowed (filters tracking noise)
NEUTRAL_SPREAD    = 55    # px between index+middle tips to enter neutral mode
SMOOTH_BUFFER     = 5     # moving average window for fingertip position
MAX_UNDO          = 25

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


# ─── Helpers ───────────────────────────────────────────────────────────────────

def lm_px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)

def finger_up(lms, tip, pip):
    return lms[tip].y < lms[pip].y

def smooth_point(buf, pt):
    buf.append(pt)
    return int(np.mean([p[0] for p in buf])), int(np.mean([p[1] for p in buf]))

def push_undo(stack, canvas):
    stack.append(canvas.copy())
    if len(stack) > MAX_UNDO:
        stack.pop(0)

def detect_gesture(lms, w, h):
    """
    Gesture priority (first match wins):
      FIST    — all fingertips below PIP → hold 1sec to clear
      PINCH   — thumb tip close to index tip → move drawing
      NEUTRAL — index + middle up AND spread wide → safe hover / do nothing
      ERASER  — index + middle up AND close together → duster erase
      DRAW    — only index finger up → draw
      IDLE    — everything else → nothing
    Returns (gesture_str, pinch_center_or_None)
    """
    idx_up   = finger_up(lms, 8,  6)
    mid_up   = finger_up(lms, 12, 10)
    ring_up  = finger_up(lms, 16, 14)
    pinky_up = finger_up(lms, 20, 18)

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
    elif idx_up and mid_up and not ring_up and not pinky_up and spread > NEUTRAL_SPREAD:
        return "NEUTRAL", None
    elif idx_up and mid_up and not ring_up and not pinky_up and spread <= NEUTRAL_SPREAD:
        return "ERASER", None
    elif idx_up and not mid_up:
        return "DRAW", None
    else:
        return "IDLE", None

def hit_palette(tip_x, tip_y):
    if abs(tip_y - PAL_Y) > PAL_R + 8:
        return -1
    for i in range(len(PALETTE)):
        if abs(tip_x - (PAL_X0 + i * PAL_STEP)) <= PAL_R + 8:
            return i
    return -1

def blend_canvas(frame, canvas):
    """Non-black canvas pixels show as drawing; black = camera shows through."""
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    bg = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
    fg = cv2.bitwise_and(canvas, canvas, mask=mask)
    return cv2.add(bg, fg)

def draw_ui(frame, color_idx, eraser_mode, fist_count, gesture):
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
    lbl_key       = "ERASER" if eraser_mode else gesture
    text, col     = labels.get(lbl_key, ("", (255, 255, 255)))
    cv2.putText(frame, text, (w - 270, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

    # Fist clear progress bar
    if fist_count > 0:
        bx, by = w // 2 - 110, h - 40
        bw     = int((fist_count / FIST_CLEAR_FRAMES) * 220)
        cv2.rectangle(frame, (bx,      by), (bx + 220, by + 20), (50,  50,  50),  -1)
        cv2.rectangle(frame, (bx,      by), (bx + bw,  by + 20), (0,   100, 255), -1)
        cv2.rectangle(frame, (bx,      by), (bx + 220, by + 20), (180, 180, 180),  1)
        cv2.putText(frame, "HOLD FIST TO CLEAR", (bx + 28, by - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Gesture guide
    guide = [
        "1 finger  : draw",
        "2 close   : erase",
        "2 spread  : hover",
        "pinch     : move",
        "fist 1sec : clear all",
    ]
    for j, line in enumerate(guide):
        cv2.putText(frame, line, (w - 185, 65 + j * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, (140, 140, 140), 1)

    # Bottom controls hint
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

    h, w       = frame.shape[:2]
    canvas     = np.zeros((h, w, 3), dtype=np.uint8)
    undo_stack = []

    color_idx   = 0
    eraser_mode = False
    drawing     = False
    prev_pt     = None
    fist_count  = 0
    pinch_prev  = None
    smooth_buf  = deque(maxlen=SMOOTH_BUFFER)

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

                # Smooth the index fingertip to reduce jitter
                raw_ix, raw_iy = lm_px(lms[8], w, h)
                ix, iy         = smooth_point(smooth_buf, (raw_ix, raw_iy))

                # ── FIST: hold to clear canvas ────────────────────────────
                if gesture == "FIST":
                    fist_count += 1
                    drawing     = False
                    prev_pt     = None
                    pinch_prev  = None
                    if fist_count >= FIST_CLEAR_FRAMES:
                        push_undo(undo_stack, canvas)
                        canvas[:] = 0
                        fist_count = 0
                        print("[Cleared]")

                # ── PINCH: translate entire drawing ───────────────────────
                elif gesture == "PINCH":
                    fist_count = 0
                    drawing    = False
                    prev_pt    = None
                    if pinch_prev is not None:
                        dx = pinch_pt[0] - pinch_prev[0]
                        dy = pinch_pt[1] - pinch_prev[1]
                        if abs(dx) > PINCH_DEAD_ZONE or abs(dy) > PINCH_DEAD_ZONE:
                            M = np.float32([[1, 0, dx], [0, 1, dy]])
                            canvas = cv2.warpAffine(canvas, M, (w, h))
                    pinch_prev = pinch_pt
                    cv2.circle(frame, pinch_pt, 12, (255, 200, 0), 2)

                # ── NEUTRAL: safe hover — nothing happens ─────────────────
                elif gesture == "NEUTRAL":
                    fist_count  = 0
                    pinch_prev  = None
                    drawing     = False
                    prev_pt     = None
                    eraser_mode = False

                # ── ERASER: drag like a duster across canvas ──────────────
                elif gesture == "ERASER":
                    fist_count  = 0
                    pinch_prev  = None
                    eraser_mode = True
                    mx, my = lm_px(lms[12], w, h)
                    ex     = (ix + mx) // 2
                    ey     = (iy + my) // 2
                    # Show eraser circle outline on the live frame
                    cv2.circle(frame, (ex, ey), ERASER_THICKNESS // 2, (80, 80, 255), 2)
                    if not drawing:
                        push_undo(undo_stack, canvas)
                        drawing = True
                    if prev_pt is not None:
                        dist = np.hypot(ex - prev_pt[0], ey - prev_pt[1])
                        if dist > MIN_DRAW_DIST:
                            cv2.line(canvas, prev_pt, (ex, ey), (0, 0, 0), ERASER_THICKNESS)
                            cv2.circle(canvas, (ex, ey), ERASER_THICKNESS // 2, (0, 0, 0), -1)
                            prev_pt = (ex, ey)
                    else:
                        prev_pt = (ex, ey)

                # ── DRAW ──────────────────────────────────────────────────
                elif gesture == "DRAW":
                    fist_count  = 0
                    pinch_prev  = None
                    eraser_mode = False
                    hit = hit_palette(ix, iy)
                    if hit >= 0:
                        # Hovering over colour palette → switch colour
                        color_idx = hit
                        drawing   = False
                        prev_pt   = None
                    else:
                        color = PALETTE[color_idx][1]
                        cv2.circle(frame, (ix, iy), BRUSH_THICKNESS // 2 + 2, color, -1)
                        if not drawing:
                            push_undo(undo_stack, canvas)
                            drawing = True
                        if prev_pt is not None:
                            dist = np.hypot(ix - prev_pt[0], iy - prev_pt[1])
                            if MIN_DRAW_DIST < dist < MAX_DRAW_DIST:
                                # Normal stroke
                                cv2.line(canvas, prev_pt, (ix, iy), color, BRUSH_THICKNESS)
                                cv2.circle(canvas, (ix, iy), BRUSH_THICKNESS // 2, color, -1)
                                prev_pt = (ix, iy)
                            elif dist >= MAX_DRAW_DIST:
                                # Tracking jumped — reset anchor without drawing
                                prev_pt = (ix, iy)
                            # dist < MIN_DRAW_DIST → paused → do nothing, no dot
                        else:
                            prev_pt = (ix, iy)

                # ── IDLE ──────────────────────────────────────────────────
                else:
                    fist_count = 0
                    pinch_prev = None
                    drawing    = False
                    prev_pt    = None

            else:
                # No hand detected
                drawing    = False
                prev_pt    = None
                fist_count = 0
                pinch_prev = None
                smooth_buf.clear()

            # ── Composite output ──────────────────────────────────────────
            output = blend_canvas(frame, canvas)
            draw_ui(output, color_idx, eraser_mode, fist_count, gesture)
            cv2.imshow("Air Doodle", output)

            # ── Keyboard shortcuts ────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):        # Q or ESC → quit
                break
            elif key == 26:                   # Ctrl+Z → undo
                if undo_stack:
                    canvas  = undo_stack.pop()
                    drawing = False
                    prev_pt = None
                    print(f"[Undo] {len(undo_stack)} states left")
            elif key == 19:                   # Ctrl+S → save PNG
                ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"air_doodle_{ts}.png"
                gray_c = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                _, smask = cv2.threshold(gray_c, 1, 255, cv2.THRESH_BINARY)
                save_img = np.full((h, w, 3), 255, dtype=np.uint8)
                save_img[smask > 0] = canvas[smask > 0]
                cv2.imwrite(fname, save_img)
                print(f"[Saved] {fname}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()