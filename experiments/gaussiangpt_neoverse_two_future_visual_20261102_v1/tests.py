"""Protect ordered video/visual columns against misleading rearrangement."""
from PIL import Image
from visual import panels, arrange


colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]
original = Image.new('RGB', (2240,252))
for i, color in enumerate(colors):
    original.paste(Image.new('RGB', (448,252), color), (448*i,0))
parts = panels(original)
video = arrange(parts, (0,2,4))
motion = arrange(parts, (0,2))
static = arrange(parts, (3,4))
assert [video.getpixel((448*i,0)) for i in range(3)] == [colors[i] for i in (0,2,4)]
assert [motion.getpixel((448*i,0)) for i in range(2)] == [colors[i] for i in (0,2)]
assert [static.getpixel((448*i,0)) for i in range(2)] == [colors[i] for i in (3,4)]
try:
    panels(Image.new('RGB', (2240,251)))
    raise AssertionError('Unexpected frame shape accepted')
except ValueError:
    pass
print('PASS: paired comparison columns and rejection of malformed input frames')
