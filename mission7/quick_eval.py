import os
import sys
from pathlib import Path

SRC = r"d:\05.gdrive\codeit\mission7\src\A04.py"

# Read and neutralize top-level Execute_Training calls so importing doesn't run full training
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

# Execute modified source in a clean namespace
ns = {}
try:
    exec(mod_src, ns)
except Exception as e:
    print("Failed to load modified A04.py:", e)
    raise

# Helpers from module
MyMeta = ns.get('MyMeta')
MakeModel = ns.get('MakeModel')
GetLoader = ns.get('GetLoader')
GetTrainValidationSplit = ns.get('GetTrainValidationSplit')

if MyMeta is None or MakeModel is None or GetLoader is None:
    print("Required symbols not found in A04.py")
    raise SystemExit(1)

# Create meta and check data availability
meta = MyMeta()
if meta.df_trainval is None:
    print("Train/val file not found or empty. Aborting quick eval.")
    raise SystemExit(1)

# Create model instance (SSDLite chosen to be lightweight)
model = MakeModel('SSDLite', meta=meta, gubun='partial', epochs=1, lr=0.005)
print(f"Created model: {model.getMyName()} on device {meta.device}")

# Prepare transforms and data loaders
transforms = model.get_default_transforms()
train_list, val_list = GetTrainValidationSplit(None)
train_loader, val_loader, test_loader = GetLoader(meta, train_list, val_list, meta.test_list, transforms, batchSize=2)

# Run a single evaluation (validation) to trigger evalModel -> save_Image
print("Running evalModel for 1 validation pass (saving up to 2 images)...")
try:
    mAP = model.evalModel(val_loader, epoch=0, save_images=True, max_images=2)
    print(f"evalModel finished, mAP={mAP}")
except Exception as e:
    print("evalModel raised an exception:", e)
    raise

# Report expected save directory
save_dir = Path(meta.modelfiles_dir) / f"{model.getMyName()}_Validation_{model._gubun}_{model._epochs}_0_{model._lr}"
print("Expected saved images under:", save_dir)
print("Quick eval done.")
