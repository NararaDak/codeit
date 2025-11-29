import os
from pathlib import Path

SRC = r"d:\05.gdrive\codeit\mission7\src\A04.py"

# Load A04.py safely (neutralize Execute_Training calls)
with open(SRC, 'r', encoding='utf-8') as f:
    src = f.read()

lines = src.splitlines()
new_lines = []
for line in lines:
    stripped = line.lstrip()
    if stripped.startswith("Execute_Training("):
        new_lines.append("# " + line)
    else:
        new_lines.append(line)
mod_src = "\n".join(new_lines)

ns = {}
exec(mod_src, ns)

MyMeta = ns.get('MyMeta')
MakeModel = ns.get('MakeModel')
GetLoader = ns.get('GetLoader')

meta = MyMeta()
if meta.df_trainval is None:
    print("No trainval file; aborting")
    raise SystemExit(1)

# Instantiate model
model = MakeModel('SSDLite', meta=meta, gubun='partial', epochs=1, lr=0.001)
print('Model created:', model.getMyName())

transforms = model.get_default_transforms()
train_list, val_list = ns.get('GetTrainValidationSplit')(None)
train_loader, val_loader, test_loader = GetLoader(meta, train_list, val_list, meta.test_list, transforms, batchSize=4)

# Call testModel directly
print('Calling testModel...')
res = model.testModel(test_loader, epoch_index=0, save_images=True, max_images=2)
print('testModel returned mAP=', res)
save_dir = Path(meta.modelfiles_dir) / f"{model.getMyName()}_Test_{model._gubun}_{model._epochs}_0_{model._lr}"
print('Saved images (if any) under:', save_dir)
