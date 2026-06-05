import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from datetime import datetime
import copy

from core.stroke import Stroke, finalize_stroke
from core.gestures import (detect_gesture, find_stroke_at, apply_eraser, lm_px, get_wrist_angle, ROTATE_DEAD_ZONE)
from core.ui import PALETTE, hit_palette, blend_canvas, render_canvas, draw_ui

# ─── Configuration ─────────────────────────────────────────────────────────────
BRUSH_THICKNESS   = 6
ERASER_THICKNESS  = 50
FIST_CLEAR_FRAMES = 30
PINCH_DEAD_ZONE   = 3
MIN_DRAW_DIST     = 4
MAX_DRAW_DIST     = 80
SMOOTH_BUFFER     = 5
MAX_UNDO          = 25

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils


def smooth_point(buf, pt):
    buf.append(pt)
    return int(np.mean([p[0] for p in buf])), int(np.mean([p[1] for p in buf]))


def push_undo(stack, strokes):
    stack.append(copy.deepcopy(strokes))
    if len(stack) > MAX_UNDO:
        stack.pop(0)


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

    strokes        = []
    current_stroke = None
    undo_stack     = []

    color_idx        = 0
    eraser_mode      = False
    fist_count       = 0
    pinch_prev       = None
    selected_stroke  = None
    smooth_buf       = deque(maxlen=SMOOTH_BUFFER)
    prev_wrist_angle = None
    is_rotating      = False

    with mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame  = cv2.flip(frame, 1)
            result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            all_lms = []
            if result.multi_hand_landmarks:
                for hand_obj in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_obj, mp_hands.HAND_CONNECTIONS)
                    all_lms.append(hand_obj.landmark)

            # ── Hand role assignment ──────────────────────────────────────────
            # dominant = the drawing/pinching hand
            # second   = only assigned when dominant is actively PINCHING
            #
            # When neither hand is pinching, pick dominant by gesture priority
            # so a resting second hand never hijacks drawing.
            dominant_lms = None
            second_lms   = None
            gesture      = "IDLE"
            pinch_pt     = None

            if len(all_lms) == 1:
                dominant_lms = all_lms[0]

            elif len(all_lms) == 2:
                g0, pt0 = detect_gesture(all_lms[0], w, h)
                g1, pt1 = detect_gesture(all_lms[1], w, h)

                if g0 == "PINCH":
                    dominant_lms      = all_lms[0]
                    second_lms        = all_lms[1]
                    gesture, pinch_pt = g0, pt0

                elif g1 == "PINCH":
                    dominant_lms      = all_lms[1]
                    second_lms        = all_lms[0]
                    gesture, pinch_pt = g1, pt1

                else:
                    # Neither pinching — pick dominant by gesture priority.
                    # Higher number = more intentional active gesture.
                    # Second hand is intentionally left as None and ignored.
                    priority = {"DRAW": 5, "ERASER": 4, "FIST": 3,
                                "NEUTRAL": 2, "IDLE": 1}
                    if priority.get(g0, 1) >= priority.get(g1, 1):
                        dominant_lms      = all_lms[0]
                        gesture, pinch_pt = g0, pt0
                    else:
                        dominant_lms      = all_lms[1]
                        gesture, pinch_pt = g1, pt1

            # ── Dominant hand processing ──────────────────────────────────────
            if dominant_lms:
                if gesture == "IDLE" and pinch_pt is None:
                    gesture, pinch_pt = detect_gesture(dominant_lms, w, h)

                raw_ix, raw_iy = lm_px(dominant_lms[8], w, h)
                ix, iy         = smooth_point(smooth_buf, (raw_ix, raw_iy))

                # Finalize in-progress stroke if gesture left draw/erase
                if gesture not in ("DRAW", "ERASER") and current_stroke:
                    if current_stroke.is_eraser:
                        strokes        = apply_eraser(strokes, current_stroke)
                        current_stroke = None
                    else:
                        current_stroke = finalize_stroke(strokes, current_stroke)

                # Deselect stroke and reset rotation if leaving pinch
                if gesture != "PINCH":
                    selected_stroke  = None
                    pinch_prev       = None
                    prev_wrist_angle = None
                    is_rotating      = False

                # ── FIST: hold to clear ───────────────────────────────────────
                if gesture == "FIST":
                    fist_count += 1
                    if fist_count >= FIST_CLEAR_FRAMES:
                        push_undo(undo_stack, strokes)
                        strokes    = []
                        fist_count = 0
                        print("[Cleared]")

                # ── PINCH: move stroke + rotate via second hand wrist ─────────
                elif gesture == "PINCH":
                    fist_count = 0

                    if selected_stroke is None:
                        candidate = find_stroke_at(strokes, pinch_pt[0], pinch_pt[1])
                        if candidate:
                            push_undo(undo_stack, strokes)
                            selected_stroke = candidate
                            pinch_prev      = pinch_pt
                    else:
                        # Move with dominant hand
                        if pinch_prev is not None:
                            dx = pinch_pt[0] - pinch_prev[0]
                            dy = pinch_pt[1] - pinch_prev[1]
                            if abs(dx) > PINCH_DEAD_ZONE or abs(dy) > PINCH_DEAD_ZONE:
                                selected_stroke.translate(dx, dy)
                        pinch_prev = pinch_pt

                        # Rotate with second hand wrist twist
                        if second_lms is not None:
                            cur_angle = get_wrist_angle(second_lms, w, h)
                            if prev_wrist_angle is not None:
                                delta = cur_angle - prev_wrist_angle
                                # Wrap delta into [-π, +π] to handle the ±π boundary
                                if delta >  math.pi:
                                    delta -= 2 * math.pi
                                if delta < -math.pi:
                                    delta += 2 * math.pi
                                if abs(delta) > ROTATE_DEAD_ZONE:
                                    selected_stroke.rotate(delta)
                            prev_wrist_angle = cur_angle
                            is_rotating      = True
                        else:
                            prev_wrist_angle = None
                            is_rotating      = False

                    cv2.circle(frame, pinch_pt, 12, (255, 200, 0), 2)

                # ── NEUTRAL ───────────────────────────────────────────────────
                elif gesture == "NEUTRAL":
                    fist_count  = 0
                    eraser_mode = False

                # ── ERASER ────────────────────────────────────────────────────
                elif gesture == "ERASER":
                    fist_count  = 0
                    eraser_mode = True

                    mx, my = lm_px(dominant_lms[12], w, h)
                    ex     = (ix + mx) // 2
                    ey     = (iy + my) // 2

                    cv2.circle(frame, (ex, ey), ERASER_THICKNESS // 2, (80, 80, 255), 2)

                    if current_stroke is None:
                        push_undo(undo_stack, strokes)
                        current_stroke = Stroke((0, 0, 0), ERASER_THICKNESS, is_eraser=True)
                        current_stroke.add_point((ex, ey))
                    else:
                        last = current_stroke.points[-1]
                        if np.hypot(ex - last[0], ey - last[1]) > MIN_DRAW_DIST:
                            current_stroke.add_point((ex, ey))

                # ── DRAW ──────────────────────────────────────────────────────
                elif gesture == "DRAW":
                    fist_count  = 0
                    eraser_mode = False

                    hit = hit_palette(ix, iy)
                    if hit >= 0:
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
                                current_stroke.add_point((ix, iy))

                # ── IDLE ──────────────────────────────────────────────────────
                else:
                    fist_count = 0

            else:
                # No hands in frame
                if current_stroke and current_stroke.is_eraser:
                    strokes        = apply_eraser(strokes, current_stroke)
                    current_stroke = None
                else:
                    current_stroke = finalize_stroke(strokes, current_stroke)

                fist_count       = 0
                pinch_prev       = None
                selected_stroke  = None
                prev_wrist_angle = None
                is_rotating      = False
                smooth_buf.clear()

            # ── Render ────────────────────────────────────────────────────────
            canvas = render_canvas(strokes, current_stroke, h, w)
            output = blend_canvas(frame, canvas)
            draw_ui(output, color_idx, eraser_mode,
                    fist_count, FIST_CLEAR_FRAMES, gesture, selected_stroke,
                    is_rotating=is_rotating)
            cv2.imshow("Air Doodle", output)

            # ── Keyboard ──────────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                break

            elif key == 26:                  # Ctrl+Z
                if undo_stack:
                    strokes          = undo_stack.pop()
                    current_stroke   = None
                    selected_stroke  = None
                    prev_wrist_angle = None
                    is_rotating      = False
                    print(f"[Undo] {len(undo_stack)} states left")

            elif key == 19:                  # Ctrl+S
                ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname       = f"air_doodle_{ts}.png"
                save_canvas = render_canvas(strokes, None, h, w)
                gray_c      = cv2.cvtColor(save_canvas, cv2.COLOR_BGR2GRAY)
                _, smask    = cv2.threshold(gray_c, 1, 255, cv2.THRESH_BINARY)
                save_img    = np.full((h, w, 3), 255, dtype=np.uint8)
                save_img[smask > 0] = save_canvas[smask > 0]
                cv2.imwrite(fname, save_img)
                print(f"[Saved] {fname}")

    cap.release()
    cv2.destroyAllWindows()


import math

if __name__ == "__main__":
    main()