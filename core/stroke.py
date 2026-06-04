import cv2
import numpy as np


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


def finalize_stroke(strokes, current_stroke):
    """Append current_stroke to strokes if it has points. Returns None."""
    if current_stroke and current_stroke.points:
        strokes.append(current_stroke)
    return None