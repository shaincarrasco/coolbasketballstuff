"""Generate the NBA Projects PWA icons (original mark: dark rounded square, two
basketball-seam arcs in white). Re-run: python site/icons/make_icons.py"""
import os
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
BG = (11, 13, 18, 255)        # --bg
INK = (255, 255, 255, 255)


def draw_mark(size, maskable=False, rounded=True):
    S = 8  # supersample
    W = size * S
    img = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    radius = int(W * 0.22) if (rounded and not maskable) else 0
    d.rounded_rectangle([0, 0, W - 1, W - 1], radius=radius, fill=BG)
    # Safe zone: maskable icons keep content inside the central 80%.
    pad = W * (0.22 if maskable else 0.16)
    cx = cy = W / 2
    r = (W / 2) - pad
    stroke = int(W * 0.075)
    # Outer ball circle (thin) + two seam arcs (the classic basketball curves).
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=INK, width=int(stroke * 0.8))
    # Seam arcs: circles offset left/right, clipped to the ball.
    mask = Image.new("L", (W, W), 0)
    ImageDraw.Draw(mask).ellipse([cx - r + stroke * 0.3, cy - r + stroke * 0.3,
                                  cx + r - stroke * 0.3, cy + r - stroke * 0.3], fill=255)
    seams = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    sd = ImageDraw.Draw(seams)
    off = r * 1.15
    rr = r * 1.05
    for sx in (cx - off, cx + off):
        sd.ellipse([sx - rr, cy - rr, sx + rr, cy + rr], outline=INK, width=stroke)
    # Clip the seam strokes to the ball: alpha = stroke alpha x circle mask.
    from PIL import ImageChops
    a = ImageChops.multiply(seams.getchannel("A"), mask)
    seams.putalpha(a)
    img = Image.alpha_composite(img, seams)
    return img.resize((size, size), Image.LANCZOS)


def main():
    outs = {
        "icon-192.png": draw_mark(192),
        "icon-512.png": draw_mark(512),
        "icon-maskable-192.png": draw_mark(192, maskable=True),
        "icon-maskable-512.png": draw_mark(512, maskable=True),
        "apple-touch-icon.png": draw_mark(180, rounded=False),  # iOS applies its own mask
    }
    for name, im in outs.items():
        p = os.path.join(HERE, name)
        im.convert("RGBA").save(p, optimize=True)
        print("wrote", p, im.size)


if __name__ == "__main__":
    main()
