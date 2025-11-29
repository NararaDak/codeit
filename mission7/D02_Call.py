"""Command-line wrapper to call D01.Execute_Training with optional parameters.

Usage examples (PowerShell):

# 1) Default: runs YOLOv8n with recommended defaults
#    (equivalent to nothing specified)
# d:; cd 'd:\01.project\CodeIt\mission7\src'; & 'D:\01.project\CodeIt\.venv\Scripts\python.exe' D02_Call.py

# 2) SSD (recommended): epochs=10, lr=0.001, batchSize=8
# d:; cd 'd:\01.project\CodeIt\mission7\src'; & 'D:\01.project\CodeIt\.venv\Scripts\python.exe' D02_Call.py --model_name SSD --epochs 10 --lr 0.001 --batchSize 8

# 3) Faster R-CNN ResNet50 (recommended): epochs=5, lr=0.005, batchSize=4
# d:; cd 'd:\01.project\CodeIt\mission7\src'; & 'D:\01.project\CodeIt\.venv\Scripts\python.exe' D02_Call.py --model_name FasterRCNN --epochs 5 --lr 0.005 --batchSize 4

# 4) RetinaNet ResNet50 (recommended, freeze backbone): epochs=5, lr=0.001, batchSize=8
# d:; cd 'd:\01.project\CodeIt\mission7\src'; & 'D:\01.project\CodeIt\.venv\Scripts\python.exe' D02_Call.py --model_name RetinaNet --gubun freeze --epochs 5 --lr 0.001 --batchSize 8

# 5) SSDLite MobileNetV3 (recommended for mobile): epochs=30, lr=0.001, batchSize=16
# d:; cd 'd:\01.project\CodeIt\mission7\src'; & 'D:\01.project\CodeIt\.venv\Scripts\python.exe' D02_Call.py --model_name SSDLite --epochs 30 --lr 0.001 --batchSize 16

# 6) YOLOv8n (recommended default): epochs=20, lr=0.01, backbone_lr_ratio=0.05, batchSize=16
# d:; cd 'd:\01.project\CodeIt\mission7\src'; & 'D:\01.project\CodeIt\.venv\Scripts\python.exe' D02_Call.py --model_name YOLOv8n --epochs 20 --lr 0.01 --backbone_lr_ratio 0.05 --batchSize 16

# 7) Run all default trainings sequentially (if available in D01):
# d:; cd 'd:\01.project\CodeIt\mission7\src'; & 'D:\01.project\CodeIt\.venv\Scripts\python.exe' D02_Call.py --all

python D:\05.gdrive\codeit\mission7\src\D02_Call.py --model_name YOLOv8n --epochs 20 --lr 0.001 --backbone_lr_ratio 0.05 --batchSize 16


"""
import argparse
import D01 as d1


def Execute_Training(model_name=None, gubun=None, epochs=None, lr=None, backbone_lr_ratio=None, batchSize=None):
    """Proxy to D01.Execute_Training. Passes None for unspecified params so D01 can apply its defaults."""
    d1.Execute_Training(
        model_name=model_name,
        gubun=gubun,
        epochs=epochs,
        lr=lr,
        backbone_lr_ratio=backbone_lr_ratio,
        batchSize=batchSize,
    )


def _parse_args():
    parser = argparse.ArgumentParser(description="Call Execute_Training in D01 with optional parameters. Defaults to YOLOv8n if model not provided.")
    parser.add_argument('--model_name', type=str, help='Model name: SSD | FasterRCNN | RetinaNet | SSDLite | YOLOv8n')
    parser.add_argument('--gubun', type=str, help='Training strategy: partial | freeze | full')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--lr', type=float, help='Base learning rate')
    parser.add_argument('--backbone_lr_ratio', type=float, help='Backbone LR ratio (used for partial)')
    parser.add_argument('--batchSize', type=int, help='Batch size')
    parser.add_argument('--all', action='store_true', help='Run all default model trainings sequentially (calls Trains_All in D01 if present)')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    # If user asked to run all, call Trains_All if available
    if getattr(args, 'all', False):
        if hasattr(d1, 'Trains_All'):
            d1.Trains_All()
        else:
            print('Trains_All not available in D01. Falling back to Execute_Training()')
            Execute_Training()
    else:
        # Pass None for values not provided so D01 will fill defaults
        Execute_Training(
            model_name=args.model_name,
            gubun=args.gubun,
            epochs=args.epochs,
            lr=args.lr,
            backbone_lr_ratio=args.backbone_lr_ratio,
            batchSize=args.batchSize,
        )
