import math
import random
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1600, 900
random.seed(42)

img = Image.new('RGB', (W, H), '#050816')
draw = ImageDraw.Draw(img)

# Gradient background (blue -> purple)
for y in range(H):
    t = y / (H - 1)
    r = int(8 + 60 * t)
    g = int(20 + 35 * (1 - t))
    b = int(60 + 120 * (1 - 0.4 * t))
    draw.line([(0, y), (W, y)], fill=(r, g, b))

# Add soft neon clouds
cloud = Image.new('RGBA', (W, H), (0, 0, 0, 0))
cdraw = ImageDraw.Draw(cloud)
for _ in range(35):
    x = random.randint(150, W-150)
    y = random.randint(100, H-100)
    rad = random.randint(80, 220)
    color = random.choice([(80, 130, 255, 38), (165, 80, 255, 32), (100, 220, 255, 22)])
    cdraw.ellipse((x-rad, y-rad, x+rad, y+rad), fill=color)
cloud = cloud.filter(ImageFilter.GaussianBlur(30))
img = Image.alpha_composite(img.convert('RGBA'), cloud)

# Brain-like silhouette from two lobes
brain_mask = Image.new('L', (W, H), 0)
bm = ImageDraw.Draw(brain_mask)
center = (W//2, H//2+10)
for i in range(2):
    sign = -1 if i == 0 else 1
    bm.ellipse((center[0] - 380 + sign*70, center[1]-250, center[0]+40+sign*70, center[1]+250), fill=255)
# trim bottom and shape
bm.rectangle((0, center[1]+180, W, H), fill=0)
bm.ellipse((center[0]-160, center[1]+150, center[0]+160, center[1]+330), fill=180)
brain_mask = brain_mask.filter(ImageFilter.GaussianBlur(8))

# Neon rim
rim = Image.new('RGBA', (W, H), (0, 0, 0, 0))
rd = ImageDraw.Draw(rim)
for offset, alpha in [(0, 160), (4, 95), (9, 45)]:
    for i in range(2):
        sign = -1 if i == 0 else 1
        rd.ellipse((center[0]-380+sign*70-offset, center[1]-250-offset,
                    center[0]+40+sign*70+offset, center[1]+250+offset),
                   outline=(145, 210, 255, alpha), width=3)
rim = rim.filter(ImageFilter.GaussianBlur(2))
img = Image.alpha_composite(img, rim)

# Build neuron nodes within mask
pixels = brain_mask.load()
nodes = []
while len(nodes) < 180:
    x = random.randint(center[0]-420, center[0]+420)
    y = random.randint(center[1]-260, center[1]+180)
    if pixels[x, y] > 120:
        nodes.append((x, y))

# Connect to nearby nodes
net = Image.new('RGBA', (W, H), (0, 0, 0, 0))
nd = ImageDraw.Draw(net)
for i, (x1, y1) in enumerate(nodes):
    nearest = sorted(nodes, key=lambda p: (p[0]-x1)**2 + (p[1]-y1)**2)[1:5]
    for x2, y2 in nearest:
        if random.random() < 0.58:
            color = random.choice([(120, 185, 255, 70), (180, 120, 255, 70), (90, 245, 255, 55)])
            nd.line((x1, y1, x2, y2), fill=color, width=1)

# Overlay subtle equations and numbers
math_overlay = Image.new('RGBA', (W, H), (0, 0, 0, 0))
md = ImageDraw.Draw(math_overlay)
try:
    font_small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', 18)
    font_tiny = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', 12)
    font_big = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 118)
except Exception:
    font_small = ImageFont.load_default()
    font_tiny = ImageFont.load_default()
    font_big = ImageFont.load_default()

formulas = [
    '∂L/∂w = (ŷ - y)x',
    'σ(x)=1/(1+e^-x)',
    'aᶫ = Wᶫaᶫ⁻¹ + bᶫ',
    'Δw = -η∇L + λw',
    'softmax(zᵢ)=eᶻⁱ/Σeᶻʲ',
    'E=½(y-ŷ)²'
]
for _ in range(130):
    x, y = random.choice(nodes)
    txt = random.choice(formulas) if random.random() < 0.3 else str(random.choice([3.1415,2.718,0.001,42,7,13,0.987,1.618]))
    md.text((x+random.randint(-26, 18), y+random.randint(-20, 20)), txt,
            font=font_tiny if random.random() < 0.8 else font_small,
            fill=random.choice([(180, 220, 255, 60), (210, 170, 255, 62), (120, 255, 250, 50)]))

math_overlay = math_overlay.filter(ImageFilter.GaussianBlur(0.2))

# Draw node glows
for x, y in nodes:
    r = random.randint(2, 4)
    c = random.choice([(165, 220, 255, 220), (205, 145, 255, 230), (120, 255, 255, 200)])
    nd.ellipse((x-r, y-r, x+r, y+r), fill=c)

net = net.filter(ImageFilter.GaussianBlur(0.3))
img = Image.alpha_composite(img, net)
img = Image.alpha_composite(img, math_overlay)

# DDESONN central branding
text_layer = Image.new('RGBA', (W, H), (0, 0, 0, 0))
td = ImageDraw.Draw(text_layer)
label = 'DDESONN'
bbox = td.textbbox((0, 0), label, font=font_big)
text_w = bbox[2] - bbox[0]
text_h = bbox[3] - bbox[1]
text_x = (W - text_w)//2
text_y = H - text_h - 60
for dx, dy, a in [(0, 0, 220), (0, 0, 130), (2, 2, 80), (-2, -2, 80)]:
    td.text((text_x+dx, text_y+dy), label, font=font_big, fill=(188, 232, 255, a))

img = Image.alpha_composite(img, text_layer)

# Vignette
v = Image.new('L', (W, H), 0)
vd = ImageDraw.Draw(v)
vd.ellipse((-280, -220, W+280, H+260), fill=220)
v = v.filter(ImageFilter.GaussianBlur(80))
black = Image.new('RGBA', (W, H), (0, 0, 0, 110))
img = Image.composite(img, black, v)

img.convert('RGB').save('inst/art/ddesonn_hitech_brain.png', quality=96)
print('Saved inst/art/ddesonn_hitech_brain.png')
