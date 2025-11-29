from PIL import Image
from pathlib import Path

base = Path(r"d:/05.gdrive/codeit/mission7/data/pet_data/modelfiles/SSDLiteMobileNetV3Transfer_Test_partial_1_0_0.001")
files = [base / 'pred_1.png', base / 'pred_2.png']

for i,f in enumerate(files, start=1):
    if f.exists():
        im = Image.open(f)
        im.thumbnail((320,320))
        out = base / f'thumb_pred_{i}.png'
        im.save(out)
        print('Saved', out)
    else:
        print('Missing', f)
