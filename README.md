 
# ✋ Air Doodle

Draw in the air using just your hand and a webcam. No stylus, no touchscreen — just your index finger and computer vision.

Built with Python, OpenCV, and MediaPipe.

![Python](https://img.shields.io/badge/Python-3.10-blue) ![OpenCV](https://img.shields.io/badge/OpenCV-4.11-green) ![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.21-orange)

---

## Demo

> Point your index finger at the camera and start drawing.

---

## Features

| Gesture | Action |
|---|---|
| ☝️ Index finger up | Draw |
| ✌️ Two fingers close together | Erase (drag like a duster) |
| ✌️ Two fingers spread apart | Safe hover — nothing draws |
| 🤏 Pinch (thumb + index) | Move entire drawing |
| ✊ Fist held 1 second | Clear canvas |
| 🖐️ Hover index over palette | Switch colour |

**Keyboard shortcuts:**
- `Ctrl+Z` — Undo
- `Ctrl+S` — Save drawing as PNG (white background)
- `Q` or `ESC` — Quit

**Colour palette:** Red, Orange, Yellow, Green, Blue, Purple, White

---

## Installation

### Prerequisites
- [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- A working webcam

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/ziyadshaikh-cook/Air-doodle.git
cd Air-doodle

# 2. Create and activate conda environment
conda create -p venv python=3.10
conda activate venv/

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python main.py
```

---

## How It Works

- **MediaPipe** detects 21 hand landmarks per frame in real time
- **Gesture logic** checks which fingers are extended and how far apart they are to determine the current mode
- **OpenCV** captures the webcam feed, renders the drawing canvas, and composites everything into a single window
- A **5-frame moving average** smooths the fingertip position so strokes don't jitter
- A **minimum movement threshold** prevents accidental dots when your hand is paused mid-air

---

## Project Structure

```
Air-doodle/
├── main.py           # All application logic
├── requirements.txt  # Python dependencies
└── README.md
```

---

## Dependencies

```
mediapipe==0.10.21
opencv-python==4.11.0.86
numpy==1.26.4
```

> NumPy is pinned to 1.x — MediaPipe 0.10.x is not compatible with NumPy 2.x.

---

## Known Limitations

- Works best in good lighting
- Designed for a single hand (right or left)
- Fast hand movements may briefly drop tracking — the app auto-recovers

---

## Author

**Ziyad Shaikh**
Integrated MSc Data Science — Goa Business School, Goa University

---

## License

MIT License — do whatever you want with it.