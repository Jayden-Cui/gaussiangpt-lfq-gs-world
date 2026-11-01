"""Generate labeled contact sheets for human review from existing preview frames."""
from pathlib import Path
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parent
for start in (35, 123):
    frames = [root / f's{start:03d}' / f'frame_{h:02d}.png' for h in range(1, 10)]
    width, height = Image.open(frames[0]).size
    board = Image.new('RGB', (width * 3, (height + 24) * 3), 'white')
    for index, path in enumerate(frames):
        x, y = (index % 3) * width, (index // 3) * (height + 24)
        ImageDraw.Draw(board).text((x + 5, y + 5), f'h{index + 1}', fill='black')
        board.paste(Image.open(path).convert('RGB'), (x, y + 24))
    board.save(root / f's{start:03d}_all_future_montage.png')
