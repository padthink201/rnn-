"""
Draw a digit with the mouse and let the trained RNN guess it.

Run rnn_mnist.py first so that rnn_weights.npz exists, then:
    python draw_digit.py

Left mouse button: draw    |    Right mouse button / "Clear": erase everything
The prediction updates every time you release the mouse button.
"""

import os
import tkinter as tk

import numpy as np

from rnn_mnist import RNN

CANVAS = 280          # drawing area in pixels (10x the 28x28 MNIST size)
BRUSH = 11            # brush radius in pixels
PREVIEW_SCALE = 4     # size of each pixel in the 28x28 preview


# ----------------------------------------------------------------------------
# Turn the drawing into an MNIST-style 28x28 image
# ----------------------------------------------------------------------------
def resize_area(img, out):
    """Shrink a square image to (out, out) by averaging blocks of pixels."""
    s = img.shape[0]
    if s < out:  # tiny drawing: enlarge first so every block has pixels
        k = int(np.ceil(out / s))
        img = np.repeat(np.repeat(img, k, axis=0), k, axis=1)
        s = img.shape[0]
    edges = np.linspace(0, s, out + 1).astype(int)[:-1]
    sums = np.add.reduceat(np.add.reduceat(img, edges, axis=0), edges, axis=1)
    sizes = np.diff(np.append(edges, s))
    return sums / np.outer(sizes, sizes)


def to_mnist(canvas_img):
    """Mimic MNIST preprocessing: fit digit in a 20x20 box, center by mass in 28x28."""
    ys, xs = np.nonzero(canvas_img > 0.05)
    if len(ys) == 0:
        return None
    crop = canvas_img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    # Pad to a square so the aspect ratio is kept
    h, w = crop.shape
    side = max(h, w)
    square = np.zeros((side, side), dtype=np.float32)
    square[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = crop

    digit = resize_area(square, 20)
    img = np.zeros((28, 28), dtype=np.float32)
    img[4:24, 4:24] = digit

    # Shift so the center of mass is in the middle (like the original MNIST)
    total = img.sum()
    cy = (img.sum(axis=1) * np.arange(28)).sum() / total
    cx = (img.sum(axis=0) * np.arange(28)).sum() / total
    img = np.roll(img, int(round(14 - cy)), axis=0)
    img = np.roll(img, int(round(14 - cx)), axis=1)

    return np.clip(img / img.max(), 0.0, 1.0)


# ----------------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------------
class App:
    def __init__(self, root, model):
        self.model = model
        self.image = np.zeros((CANVAS, CANVAS), dtype=np.float32)
        self.last = None

        # Pre-computed round brush with soft edges
        r = np.arange(-BRUSH, BRUSH + 1)
        dist = np.sqrt(r[:, None] ** 2 + r[None, :] ** 2)
        self.brush = np.clip(BRUSH + 1 - dist, 0.0, 1.0).astype(np.float32)

        root.title("Draw a digit - RNN guesses it")
        root.resizable(False, False)

        left = tk.Frame(root, padx=10, pady=10)
        left.pack(side=tk.LEFT)
        self.canvas = tk.Canvas(left, width=CANVAS, height=CANVAS, bg="black",
                                cursor="cross", highlightthickness=1)
        self.canvas.pack()
        tk.Button(left, text="Clear", width=12, command=self.clear).pack(pady=8)

        right = tk.Frame(root, padx=10, pady=10)
        right.pack(side=tk.LEFT, fill=tk.Y)
        self.result = tk.Label(right, text="Draw a digit", font=("Arial", 28, "bold"))
        self.result.pack(anchor="w")

        # Probability bars for each digit
        self.bars = tk.Canvas(right, width=230, height=10 * 22)
        self.bars.pack(pady=8)

        tk.Label(right, text="What the network sees (28x28):").pack(anchor="w")
        self.preview = tk.Canvas(right, width=28 * PREVIEW_SCALE, height=28 * PREVIEW_SCALE,
                                 bg="black", highlightthickness=1)
        self.preview.pack(anchor="w")

        self.canvas.bind("<Button-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Button-3>", lambda e: self.clear())

        self.draw_bars(np.zeros(10))

    # ---- drawing ----
    def stamp(self, x, y):
        """Add the brush to the numpy image at (x, y)."""
        x0, y0 = x - BRUSH, y - BRUSH
        x1, y1 = x + BRUSH + 1, y + BRUSH + 1
        bx0, by0 = max(0, -x0), max(0, -y0)
        bx1 = self.brush.shape[1] - max(0, x1 - CANVAS)
        by1 = self.brush.shape[0] - max(0, y1 - CANVAS)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(CANVAS, x1), min(CANVAS, y1)
        if x0 >= x1 or y0 >= y1:
            return
        region = self.image[y0:y1, x0:x1]
        np.maximum(region, self.brush[by0:by1, bx0:bx1], out=region)

    def paint_line(self, x0, y0, x1, y1):
        steps = max(1, int(np.hypot(x1 - x0, y1 - y0) / 2))
        for t in np.linspace(0, 1, steps + 1):
            self.stamp(int(round(x0 + (x1 - x0) * t)), int(round(y0 + (y1 - y0) * t)))
        self.canvas.create_line(x0, y0, x1, y1, fill="white", width=2 * BRUSH,
                                capstyle=tk.ROUND, smooth=True)

    def on_press(self, event):
        self.last = (event.x, event.y)
        self.paint_line(event.x, event.y, event.x, event.y)

    def on_drag(self, event):
        if self.last is not None:
            self.paint_line(*self.last, event.x, event.y)
        self.last = (event.x, event.y)

    def on_release(self, event):
        self.last = None
        self.predict()

    def clear(self):
        self.image[:] = 0
        self.canvas.delete("all")
        self.preview.delete("all")
        self.result.config(text="Draw a digit")
        self.draw_bars(np.zeros(10))

    # ---- prediction ----
    def predict(self):
        img = to_mnist(self.image)
        if img is None:
            return
        probs, _ = self.model.forward(img[None, :, :])   # shape (1, 28, 28)
        probs = probs[0]
        digit = int(probs.argmax())
        self.result.config(text=f"It's a {digit}  ({probs[digit] * 100:.1f}%)")
        self.draw_bars(probs)
        self.draw_preview(img)

    def draw_bars(self, probs):
        self.bars.delete("all")
        best = int(np.argmax(probs)) if probs.max() > 0 else -1
        for d, p in enumerate(probs):
            y = d * 22
            self.bars.create_text(10, y + 10, text=str(d), font=("Arial", 11, "bold"))
            self.bars.create_rectangle(25, y + 3, 175, y + 17, outline="#999")
            color = "#2e7d32" if d == best else "#5b8def"
            self.bars.create_rectangle(25, y + 3, 25 + 150 * p, y + 17, fill=color, outline="")
            self.bars.create_text(182, y + 10, text=f"{p * 100:5.1f}%", anchor="w")

    def draw_preview(self, img):
        self.preview.delete("all")
        s = PREVIEW_SCALE
        for yy in range(28):
            for xx in range(28):
                v = int(img[yy, xx] * 255)
                if v > 0:
                    self.preview.create_rectangle(xx * s, yy * s, xx * s + s, yy * s + s,
                                                  fill=f"#{v:02x}{v:02x}{v:02x}", outline="")


def main():
    weights = "rnn_weights.npz"
    if not os.path.exists(weights):
        raise SystemExit(f"{weights} not found - run 'python rnn_mnist.py' first to train the model.")
    model = RNN.load(weights)
    root = tk.Tk()
    App(root, model)
    root.mainloop()


if __name__ == "__main__":
    main()
