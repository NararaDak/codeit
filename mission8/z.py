import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, auc

# 예측 결과

cat_preds = [(0.95, True), (0.90, True), (0.85, False), (0.80, True), (0.70, False)]

dog_preds = [(0.88, True), (0.75, False), (0.65, True), (0.60, True), (0.50, False)]

# confidence와 label 분리

cat_confidences, cat_labels = zip(*cat_preds)

dog_confidences, dog_labels = zip(*dog_preds)

# Precision-Recall 계산

cat_precision, cat_recall, _ = precision_recall_curve(cat_labels, cat_confidences)

dog_precision, dog_recall, _ = precision_recall_curve(dog_labels, dog_confidences)

# AP 계산

cat_ap = auc(cat_recall, cat_precision)

dog_ap = auc(dog_recall, dog_precision)

# mAP 계산

mAP = (cat_ap + dog_ap) / 2

# 시각화

plt.figure(figsize=(7,7))

plt.plot(cat_recall, cat_precision, marker='o', label=f'cat (AP={cat_ap:.3f})', color='blue')

plt.plot(dog_recall, dog_precision, marker='s', label=f'dog (AP={dog_ap:.3f})', color='green')

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.title(f'Precision-Recall Curve (mAP = {mAP:.3f})')

plt.legend()

plt.grid(True)

plt.show()

