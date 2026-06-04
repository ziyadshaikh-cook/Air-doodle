import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from datetime import datetime
import copy
import math

from core.stroke import Stroke, finalize_stroke
from core.gestures import (
    detect_gesture, detect_rotate_gesture, find_stroke_at,
    apply_eraser, lm_px, angle_of_point, ROTATE_DEAD_ZONE
)
from core.ui import PALETTE, hit_palette, blend_canvas, render_canvas, draw_ui

# ─── Configuration ──────────────────────────────────────────────────────────────────────────────
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

    color_idx       = 0
    eraser_mode     = False
    fist_count      = 0
    pinch_prev      = None
    selected_stroke = None
    smooth_buf      = deque(maxlen=SMOOTH_BUFFER)

    # ─ Rotation state ───────────────────────────────────────────────────────────────────────
    rotate_stroke      = None   # stroke currently being rotated
    rotate_ref_angle   = None   # angle (radians) of the rotate-gesture centroid from wrist, previous frame
    rotate_total_angle = 0.0    # total rotation applied this session (for display)

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
                # ── Check rotate gesture BEFORE the main gesture detector ────────────────────
                is_rotate, rot_center = detect_rotate_gesture(lms, w, h)

                if is_rotate:
                    gesture = "ROTATE"

                    # Use wrist (lms[0]) as the pivot origin for angle tracking
                    wx, wy = lm_px(lms[0], w, h)
                    cur_angle = angle_of_point(rot_center, (wx, wy))

                    if rotate_stroke is None:
                        # First frame of rotate gesture: find which stroke to rotate
                        candidate = find_stroke_at(strokes, rot_center[0], rot_center[1])
                        if candidate:
                            push_undo(undo_stack, strokes)
                            rotate_stroke      = candidate
                            rotate_total_angle = rotate_stroke._angle
                        rotate_ref_angle = cur_angle

                    else:
                        if rotate_ref_angle is not None:
                            delta = cur_angle - rotate_ref_angle
                            # Wrap delta into [−pi, +pi] to handle crossing ±180°
                            if delta >  math.pi:
                                delta -= 2 * math.pi
                            if delta < -math.pi:
                                delta += 2 * math.pi

                            if abs(delta) > ROTATE_DEAD_ZONE:
                                rotate_stroke.rotate(delta)
                                rotate_total_angle += delta

                        rotate_ref_angle = cur_angle

                    # Visual feedback: draw arc around the rotate gesture centroid
                    cv2.circle(frame, rot_center, 18, (255, 165, 0), 2)
                    cv2.putText(frame, f"{int(math.degrees(rotate_total_angle) % 360)}°",
                                (rot_center[0] + 22, rot_center[1] + 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 165, 0), 2)

                else:
                    # Reset rotation state when gesture is released
                    rotate_stroke    = None
                    rotate_ref_angle = None

                    gesture, pinch_pt = detect_gesture(lms, w, h)

                    raw_ix, raw_iy = lm_px(lms[8], w, h)
                    ix, iy         = smooth_point(smooth_buf, (raw_ix, raw_iy))

                    # Finalize in-progress stroke if gesture left draw/erase
                    if gesture not in ("DRAW", "ERASER") and current_stroke:
                        if current_stroke.is_eraser:
                            strokes        = apply_eraser(strokes, current_stroke)
                            current_stroke = None
                        else:
                            current_stroke = finalize_stroke(strokes, current_stroke)

                    # Deselect stroke if gesture left pinch
                    if gesture != "PINCH":
                        selected_stroke = None
                        pinch_prev      = None

                    # ── FIST: hold to clear all strokes ───────────────────────────────────────
                    if gesture == "FIST":
                        fist_count += 1
                        if fist_count >= FIST_CLEAR_FRAMES:
                            push_undo(undo_stack, strokes)
                            strokes    = []
                            fist_count = 0
                            print("[Cleared]")

                    # ── PINCH: find and move individual stroke ────────────────────────────────
                    elif gesture == "PINCH":
                        fist_count = 0

                        if selected_stroke is None:
                            candidate = find_stroke_at(strokes, pinch_pt[0], pinch_pt[1])
                            if candidate:
                                push_undo(undo_stack, strokes)
                                selected_stroke = candidate
                                pinch_prev      = pinch_pt
                        else:
                            if pinch_prev is not None:
                                dx = pinch_pt[0] - pinch_prev[0]
                                dy = pinch_pt[1] - pinch_prev[1]
                                if abs(dx) > PINCH_DEAD_ZONE or abs(dy) > PINCH_DEAD_ZONE:
                                    selected_stroke.translate(dx, dy)
                            pinch_prev = pinch_pt

                        cv2.circle(frame, pinch_pt, 12, (255, 200, 0), 2)

                    # ── NEUTRAL: safe hover, nothing happens ────────────────────────────────
                    elif gesture == "NEUTRAL":
                        fist_count  = 0
                        eraser_mode = False

                    # ── ERASER: drag two fingers to erase like a duster ─────────────────────
                    elif gesture == "ERASER":
                        fist_count  = 0
                        eraser_mode = True

                        mx, my = lm_px(lms[12], w, h)
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

                    # ── DRAW ───────────────────────────────────────────────────────────────────────────────
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

                    # ── IDLE ──────────────────────────────────────────────────────────────────────────────
                    else:
                        fist_count = 0

            else:
                # No hand in frame — finalize anything in progress
                if current_stroke and current_stroke.is_eraser:
                    strokes        = apply_eraser(strokes, current_stroke)
                    current_stroke = None
                else:
                    current_stroke = finalize_stroke(strokes, current_stroke)

                fist_count      = 0
                pinch_prev      = None
                selected_stroke = None
                rotate_stroke   = None
                rotate_ref_angle = None
                smooth_buf.clear()

            # ── Render ───────────────────────────────────────────────────────────────────────────────
            canvas = render_canvas(strokes, current_stroke, h, w)
            output = blend_canvas(frame, canvas)
            draw_ui(output, color_idx, eraser_mode,
                    fist_count, FIST_CLEAR_FRAMES, gesture, selected_stroke,
                    rotate_stroke=rotate_stroke,
                    rotate_angle=rotate_total_angle if rotate_stroke else None)
            cv2.imshow("Air Doodle", output)

            # ── Keyboard ──────────────────────────────────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                break

            elif key == 26:                  # Ctrl+Z
                if undo_stack:
                    strokes         = undo_stack.pop()
                    current_stroke  = None
                    selected_stroke = None
                    rotate_stroke   = None
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


if __name__ == "__main__":
    main()
