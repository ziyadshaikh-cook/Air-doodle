import cv2
import numpy as np
import math


class Stroke:
    def __init__(self, color, thickness, is_eraser=False):
        self.points    = []          # stored as (float, float) to prevent rounding drift
        self.color     = color
        self.thickness = thickness
        self.is_eraser = is_eraser
        self._angle    = 0.0         # cumulative rotation in radians, for UI display

    def add_point(self, pt):
        # Always store as float — prevents int truncation from accumulating
        self.points.append((float(pt[0]), float(pt[1])))

    def bounds(self):
        if not self.points:
            return None
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

    def translate(self, dx, dy):
        self.points = [(p[0] + dx, p[1] + dy) for p in self.points]

    def rotate(self, delta_angle_rad):
        """
        Rotate all points by delta_angle_rad (radians) around their centroid.
        Points stay as floats throughout — no int rounding until draw time.
        This is what prevents the cracking/breaking effect on repeated rotation.
        """
        if not self.points:
            return
        self._angle += delta_angle_rad

        cx = sum(p[0] for p in self.points) / len(self.points)
        cy = sum(p[1] for p in self.points) / len(self.points)

        cos_a = math.cos(delta_angle_rad)
        sin_a = math.sin(delta_angle_rad)

        new_pts = []
        for px, py in self.points:
            rx = px - cx
            ry = py - cy
            new_pts.append((
                cx + rx * cos_a - ry * sin_a,   # pure float — no int()
                cy + rx * sin_a + ry * cos_a
            ))
        self.points = new_pts

    def draw_on(self, canvas):
        if not self.points:
            return
        color = (0, 0, 0) if self.is_eraser else self.color
        # Convert to int ONLY here, at draw time, using round() not truncation
        int_pts = [(int(round(p[0])), int(round(p[1]))) for p in self.points]
        if len(int_pts) == 1:
            cv2.circle(canvas, int_pts[0], self.thickness // 2, color, -1)
            return
        for i in range(1, len(int_pts)):
            cv2.line(canvas, int_pts[i - 1], int_pts[i], color, self.thickness)
        cv2.circle(canvas, int_pts[-1], self.thickness // 2, color, -1)


def finalize_stroke(strokes, current_stroke):
    """Append current_stroke to strokes if it has points. Returns None."""
    if current_stroke and current_stroke.points:
        strokes.append(current_stroke)
    return None