import numpy as np
from core.stroke import Stroke

# ─── Gesture constants ─────────────────────────────────────────────────────────
PINCH_THRESHOLD  = 40    # pixel distance thumb↔index to count as pinch
NEUTRAL_SPREAD   = 55    # px between index+middle tips to enter neutral mode
SELECT_RADIUS    = 70    # px padding around stroke bounding box for pinch selection
ROTATE_DEAD_ZONE = 0.03  # radians — ignore tiny angle changes to reduce jitter


def lm_px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)


def finger_up(lms, tip, pip):
    return lms[tip].y < lms[pip].y


def detect_gesture(lms, w, h):
    """
    Priority order (first match wins):
      FIST    — all fingertips below PIP joint → hold 1 sec to clear
      PINCH   — thumb close to index → move individual stroke
      NEUTRAL — index + middle up, spread wide → safe hover
      ERASER  — index + middle up, close together → duster erase
      DRAW    — only index up → draw
      IDLE    — everything else

    Returns (gesture_str, pinch_center_or_None)
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


def detect_rotate_gesture(lms, w, h):
    """
    Detects the ROTATE gesture on a hand:
      index + thumb + middle all up, ring + pinky down.

    Returns (True, rotation_reference_point) or (False, None).
    The rotation_reference_point is the centroid of the three fingertips.
    The caller computes angle from two consecutive frames to get delta_angle.
    """
    idx_up  = finger_up(lms, 8,  6)
    mid_up  = finger_up(lms, 12, 10)
    ring_up = finger_up(lms, 16, 14)
    pky_up  = finger_up(lms, 20, 18)

    # Thumb: use y-distance from wrist as a rough "up" check
    # lms[4] = thumb tip, lms[2] = thumb MCP
    thumb_up = lms[4].y < lms[2].y

    if idx_up and mid_up and thumb_up and not ring_up and not pky_up:
        tx, ty = lm_px(lms[4],  w, h)   # thumb tip
        ix, iy = lm_px(lms[8],  w, h)   # index tip
        mx, my = lm_px(lms[12], w, h)   # middle tip
        # Also make sure this is NOT a pinch (thumb & index not too close)
        pinch_dist = np.hypot(tx - ix, ty - iy)
        if pinch_dist < 40:
            # Too close — would collide with PINCH; ignore
            return False, None
        cx = (tx + ix + mx) // 3
        cy = (ty + iy + my) // 3
        return True, (cx, cy)

    return False, None


def angle_of_point(pt, origin):
    """Returns angle in radians from origin to pt."""
    return np.arctan2(pt[1] - origin[1], pt[0] - origin[0])


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


def apply_eraser(strokes, eraser_stroke):
    """
    Remove only the points of each stroke that fall within the eraser radius.
    Strokes get split into surviving segments — each continuous run of
    remaining points becomes its own stroke object.
    Eraser strokes are never stored permanently.
    """
    if not eraser_stroke.points:
        return strokes

    radius = eraser_stroke.thickness // 2
    result = []

    for stroke in strokes:
        if stroke.is_eraser:
            continue

        # Mark which points survive (not hit by any eraser point)
        surviving_pts = []
        for sp in stroke.points:
            hit = any(
                np.hypot(ep[0] - sp[0], ep[1] - sp[1]) < radius
                for ep in eraser_stroke.points
            )
            if not hit:
                surviving_pts.append(sp)

        if not surviving_pts:
            continue  # entire stroke erased, drop it

        # Map surviving points back to original indices
        original_indices = [
            i for i, sp in enumerate(stroke.points)
            if sp in surviving_pts
        ]

        # Split into continuous segments (gaps = where erasure happened)
        segment_pts = []
        for k, idx in enumerate(original_indices):
            if k == 0:
                segment_pts.append(stroke.points[idx])
            else:
                prev_idx = original_indices[k - 1]
                if idx - prev_idx == 1:
                    segment_pts.append(stroke.points[idx])
                else:
                    # Gap found — save current segment, start new one
                    if segment_pts:
                        new_stroke = Stroke(stroke.color, stroke.thickness)
                        new_stroke.points = segment_pts
                        result.append(new_stroke)
                    segment_pts = [stroke.points[idx]]

        if segment_pts:
            new_stroke = Stroke(stroke.color, stroke.thickness)
            new_stroke.points = segment_pts
            result.append(new_stroke)

    return result
