"""MLX Creator app icon — vibrant blue→violet squircle with a clean concave
'AI sparkle' glyph (no text), soft depth. Outputs a 1024px master PNG.
"""
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

S = 1024
OUT = Path(__file__).parent / "icon_1024.png"

# vibrant brand gradient (top-left → bottom-right)
C0 = (95, 162, 255)    # blue
C1 = (124, 110, 255)   # indigo
C2 = (170, 105, 255)   # violet


def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def grad3(t):
    return lerp(C0, C1, t / 0.5) if t < 0.5 else lerp(C1, C2, (t - 0.5) / 0.5)


# ---- diagonal gradient background ----
bg = Image.new("RGBA", (S, S), (0, 0, 0, 0))
px = bg.load()
for y in range(S):
    for x in range(S):
        px[x, y] = grad3((x + y) / (2 * S)) + (255,)

# soft top-left sheen for depth
sheen = Image.new("L", (S, S), 0)
sd = ImageDraw.Draw(sheen)
sd.ellipse([-S * 0.3, -S * 0.55, S * 0.95, S * 0.5], fill=70)
sheen = sheen.filter(ImageFilter.GaussianBlur(120))
bg.alpha_composite(Image.merge("RGBA", (Image.new("L", (S, S), 255),) * 3 + (sheen,)))

# subtle bottom vignette
vig = Image.new("L", (S, S), 0)
ImageDraw.Draw(vig).ellipse([S * 0.1, S * 0.55, S * 1.3, S * 1.35], fill=60)
vig = vig.filter(ImageFilter.GaussianBlur(140))
bg.alpha_composite(Image.merge("RGBA", (Image.new("L", (S, S), 0),) * 3 + (vig,)))


def sparkle(cx, cy, R, valley=0.34, power=1.45, samples=90):
    """Bold 4-point sparkle with gently concave edges and a solid center."""
    pts = []
    for k in range(4):
        spike = k * math.pi / 2
        for j in range(samples + 1):              # spike -> valley
            t = j / samples
            ang = spike + (math.pi / 4) * t
            r = R * (valley + (1 - valley) * ((1 - t) ** power))
            pts.append((cx + r * math.sin(ang), cy - r * math.cos(ang)))
        v = spike + math.pi / 4
        for j in range(samples + 1):              # valley -> next spike
            t = j / samples
            ang = v + (math.pi / 4) * t
            r = R * (valley + (1 - valley) * (t ** power))
            pts.append((cx + r * math.sin(ang), cy - r * math.cos(ang)))
    return pts


# ---- glyph layer (white sparkles) ----
glyph = Image.new("RGBA", (S, S), (0, 0, 0, 0))
gd = ImageDraw.Draw(glyph)
gd.polygon(sparkle(S * 0.46, S * 0.47, S * 0.32), fill=(255, 255, 255, 255))
gd.polygon(sparkle(S * 0.72, S * 0.28, S * 0.115), fill=(255, 255, 255, 240))

# soft drop shadow for depth
shadow = glyph.split()[3].point(lambda a: int(a * 0.5))
shadow = shadow.filter(ImageFilter.GaussianBlur(22))
shadow_rgba = Image.new("RGBA", (S, S), (0, 0, 0, 0))
shadow_rgba.putalpha(shadow)          # black, with the blurred glyph alpha
shifted = Image.new("RGBA", (S, S), (0, 0, 0, 0))
shifted.paste(shadow_rgba, (0, 12))   # nudge down for depth
bg.alpha_composite(shifted)
bg.alpha_composite(glyph)

# ---- squircle mask ----
mask = Image.new("L", (S, S), 0)
ImageDraw.Draw(mask).rounded_rectangle([0, 0, S - 1, S - 1], radius=int(S * 0.225), fill=255)
final = Image.new("RGBA", (S, S), (0, 0, 0, 0))
final.paste(bg, (0, 0), mask)
final.save(OUT)
print("wrote", OUT)
