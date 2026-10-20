# ConvNeXt → ResNet: явна схема ансамблю

Поточний фінальний результат ConvNeXt — **0.95039**. CSV:
[submission_convnext_best.csv](../results/submission_convnext_best.csv).

## Формула в окремому файлі

[ensemble_convnext_resnet.py](../ensemble_convnext_resnet.py) виконує:

```python
p_convnext = (p_224 + p_320) / 2.0
p_final = (p_convnext + p_resnet) / 2.0
```

Ваги — 25% ConvNeXt-224, 25% ConvNeXt-320, 50% ResNet. Кожен вхід — raw
sigmoid probabilities у порядку `Cat, Dog, Moris, Motya, Biatrix`. CSV містять
`image_id` та ці п'ять колонок. Рядки вирівнюються за ID; перевіряються дублікати,
пропуски та діапазон. Суцільно бінарний ResNet CSV відхиляється.

```powershell
python ensemble_convnext_resnet.py --convnext224 test224.csv --convnext320 test320.csv --resnet test_resnet_probabilities.csv --output ensemble_probabilities.csv
```

CLI зберігає **ймовірності**, не готовий thresholded submission. Пороги фінального
ансамблю потрібно калібрувати на його спільних held-out validation-прогнозах.

## Що потрібно від колеги

Є лише `submission (6).csv` із 0/1. Ймовірнісний ансамбль ще не запущений.
Для нього потрібен один із варіантів:

- найкращий Lightning `.ckpt` та фактичний `folds.csv` його навчання;
- raw validation/test probabilities з точними ID, targets і достовірними
  відомостями про split та checkpoint.

Checkpoint сам по собі не доводить, які фото були held out. У
[ноутбуці колеги](../archive/reference/ResNet.ipynb) відкладений fold — 0.

## Експорт та запуск

1. Відкрити [resnet_export_for_ensemble.ipynb](../resnet_export_for_ensemble.ipynb)
   у Kaggle з GPU. Додати competition, checkpoint і split. Вказати
   `CHECKPOINT_PATH`, `FOLDS_PATH`, фактичний `RESNET_VAL_FOLD`.
2. Виконати інференс без навчання. Зберегти `resnet_for_ensemble.zip` з
   `validation_predictions.csv`, `test_probabilities.csv`, `resnet_export.json`.
3. Додати bundle, 224 backup та фінальний fast-320 backup до
   [cnn_2_ensemble_resnet.ipynb](../cnn_2_ensemble_resnet.ipynb).
4. Вказати `FINE_CV_DIR` → `convnext_cv_320_fast`,
   `FINE_PREDICTIONS_DIR` → `convnext_cv_320_fast_ensemble`.
5. Виконати CPU-ансамбль. Результат:
   `/kaggle/working/convnext_resnet_ensemble/submission_convnext_resnet.csv`.

Експорт відтворює Resize 224, ImageNet normalization та sigmoid без TTA або
`keep_one`. `pretrained=False` у loader означає повне відновлення навчених ваг.
Архітектура ResNet — `resnet152.a1_in1k`.

Ноутбук ансамблю вимагає однаковий початковий split, перевіряє маніфести та
оцінює тільки спільний held-out fold ResNet із відповідними ConvNeXt OOF.
Пороги підбираються на цьому fold, тому fitted F1 оптимістичний, особливо для
рідкісних класів. Це не незалежний full-CV score.

## Binary-only експеримент

[Архівний voting notebook](../archive/experiments/cnn_2_vote_resnet.ipynb)
застосовує більшість із трьох голосів 224 / 320 / binary ResNet. Це інша схема,
а не усереднення probability arrays. Результати —
[archive/results/convnext_resnet_vote/](../archive/results/convnext_resnet_vote/).

Перевірено 18 673 ID та точне відтворення ConvNeXt baseline. Змінено 72 мітки
на 64 фото. Kaggle score голосування — **0.94854**. Він не покращив наш blend
0.95039, тому залишений в архіві експериментів. Ймовірнісний ансамбль 25/25/50
ще не перевірений і є окремим методом.

## Повернення backup у Kaggle

Input → Upload → завантажити ZIP → створити Private Dataset → підключити через
Add Input → Your Datasets, якщо не додався автоматично. У наступній сесії
підключати той самий Dataset. Старий текстовий output клітинки не доводить,
що файли з попередньої сесії збереглися.
