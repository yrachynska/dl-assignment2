# Архів експериментів

Фінальний вибір описаний у [README](../README.md) та [звіті](../REPORT.md).

## experiments/

| Ноутбук | Етап |
|---|---|
| `cnn_2.ipynb` | Single-fold ConvNeXt та pipeline за зразком ResNet |
| `cnn_2_inference.ipynb` | Single-fold inference |
| `cnn_2_tta.ipynb` | Original / horizontal-flip TTA |
| `cnn_2_cv.ipynb` | П'ять фолдів 224, OOF та пороги |
| `cnn_2_ensemble.ipynb` | Test ensemble моделей 224 |
| `cnn_2_cv_320.ipynb` | Повний 320 fine-tuning, raw/EMA, до шести епох |
| `cnn_2_ensemble_320.ipynb` | Історичний original/TTA inference 320 |
| `cnn_2_blend_224_320.ipynb` | Історичний original/TTA blend |
| `cnn_2_vote_resnet.ipynb` | Три бінарні голоси ConvNeXt/ResNet |

Історичні notebooks можуть очікувати старі Output-папки та TTA-поля. Для
фінального original-only fast-run використовувати `cnn_2_final.ipynb` у корені.

## reference/

`ResNet.ipynb` — оригінальний ноутбук колеги, перенесений без зміни вмісту.
Наш pipeline запозичив підхід до preprocessing, augmentation та дисбалансу
класів, але зберіг backbone ConvNeXt.

## results/

`convnext_resnet_vote` містить voting CSV, контрольне відтворення нашого blend
та діагностику. Відмінності від baseline не є числом виправлених помилок:
validation-прогнозів ResNet для оцінки цього голосування немає.
