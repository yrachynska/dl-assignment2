# ConvNeXt + ResNet: запуск експерименту

## Доступний зараз варіант: голосування за готовим CSV

Після скороченого запуску 320 наш новий найкращий public score — **0.95039**
(`submission_blend_224_320 (1).csv`). Для ResNet поки є лише
`submission (6).csv` із готовими 0/1, без validation-прогнозів і навчених ваг.

`cnn_2_vote_resnet.ipynb` відтворює наш blend для контролю джерел та створює
окремий експеримент: для кожної мітки більшість із трьох голосів — ConvNeXt-224,
ConvNeXt-320-fast та бінарний ResNet. Пороги наших гілок беруться з їхніх
original-only OOF-конфігурацій. Рівні 0/1 ResNet не видаються за ймовірності.
Моріс і Мотя можуть залишатися позитивними одночасно; `keep_one` не додається.

Готовий файл на фактичних збережених прогнозах:
`artifacts/convnext_resnet_vote/submission_vote_convnext_resnet.csv`.
Початковий submission із 0.95039 відтворено точно, перевірено всі 18 673 ID та
порядок рядків. Новий варіант відрізняється на 64 фото / 72 мітках. Це відмінності,
а не число виправлених помилок: validation F1 цього ансамблю невідомий.
Правило голосування зафіксоване без підбору за public leaderboard.

Для відтворення в Kaggle додати обидва бекапи та CSV колеги, відкрити
`cnn_2_vote_resnet.ipynb` і виконати Run All. GPU не потрібен.
Для цього конкретного експерименту файл колеги перевіряється за SHA-256.
Якщо пізніше з'являться її ваги або ймовірності, використати повний
ймовірнісний ансамбль, описаний далі.

## Ймовірнісний ансамбль після отримання ваг ResNet

Наш поточний найкращий submission — `submission_blend_224_320.csv`, public score
**0.95039** після скороченого запуску (попередній запуск — 0.94996).
Це blend 224 + 320 без TTA. Навчальні ноутбуки та `ResNet.ipynb`
цим експериментом не змінюються.

## 1. Повернути бекапи у Kaggle

У правій панелі **Input → Upload** завантажити обидва архіви:

- `convnext_backup.zip` — попередні моделі 224 та їхні test probabilities;
- `convnext_320_backup_….zip` — моделі 320, результати їхнього ансамблю та blend,
  якщо ці кроки вже завершилися на момент архівування.

Створити **Private Dataset**, наприклад `cats-dogs-model-backups`. Після обробки
перевірити його в Input. Якщо його не додано автоматично: **Add Input → Your
Datasets → назва датасету → Add**. Датасет можна повторно підключати у нових сесіях;
знову тренувати моделі не потрібно. Новий API key для цього не потрібен.

Папки під `/kaggle/input` доступні лише для читання, що підходить для інференсу.
Нові файли ноутбуки записують у `/kaggle/working`.

Для цього CPU-ансамблю потрібні:

```text
convnext_cv/ensemble_config.json
convnext_cv/oof_predictions.csv
convnext_cv_ensemble/ensemble_config.json
convnext_cv_ensemble/test_probabilities_original.csv
convnext_cv_320/ensemble_config.json
convnext_cv_320/oof_predictions.csv
convnext_cv_320_ensemble/ensemble_config.json
convnext_cv_320_ensemble/test_probabilities_original.csv
```

Якщо архів був створений до інференсу, спочатку запустити відповідний
`cnn_2_ensemble.ipynb` / `cnn_2_ensemble_320.ipynb` із відновленими вагами. Це
створює прогнози, не запускаючи навчання.

## 2. Експортувати ймовірності ResNet

Запустити `resnet_export_for_ensemble.ipynb` у Kaggle з GPU, додавши competition,
найкращий **Lightning `.ckpt` колеги** та **її фактичний `folds.csv`**.
Вказати `CHECKPOINT_PATH` та `FOLDS_PATH`. У наявному `ResNet.ipynb` відкладений
fold — 0; `RESNET_VAL_FOLD` повинен відповідати навчанню саме цих ваг.

Ні назва checkpoint, ні високий validation score не доводять, який split він
використовував. Це має підтвердити власниця checkpoint. Передавати notebook із
навчанням або тренувати модель повторно не потрібно.

Інференс відтворює Resize 224, ImageNet normalization та sigmoid із її ноутбука.
`pretrained=False` використовується лише тому, що повністю відновлюються навчені
ваги. Архітектура — `resnet152.a1_in1k`. Ні TTA, ні `keep_one` не додаються.

Зберегти папку або готовий `resnet_for_ensemble.zip`:

```text
resnet_for_ensemble/
    validation_predictions.csv
    test_probabilities.csv
    resnet_export.json
```

Передати всю папку та додати її в Input ноутбука ансамблю як private Dataset.
Одного submission із 0/1 для цього експерименту недостатньо. Якщо в колеги вже
є ймовірності в пам'яті, їх можна експортувати в цей самий формат, але обов'язково
з точними `image_id`, split, checkpoint і метаданими; не вигадувати ці поля.

## 3. Ансамбль

Запустити `cnn_2_ensemble_resnet.ipynb`. GPU та W&B не потрібні.
За замовчуванням ноутбук шукає відомі папки в working, а потім підключені
Input. Якщо версій кілька, вказати їх явно в першій клітинці.

Якщо 320 відновлюється через скорочений `cnn_2_fast_320.ipynb`, у ноутбуці
ансамблю вказати `FINE_CV_DIR` на `convnext_cv_320_fast`, а
`FINE_PREDICTIONS_DIR` — на `convnext_cv_320_fast_ensemble`. Після підключення
бекапу в новій сесії це будуть відповідні папки під `/kaggle/input`.
Скорочений запуск створює потрібні original-only OOF і test probabilities;
додатково запускати старий TTA-ансамбль не потрібно. Він є новим експериментом
із меншим числом епох, тому попередній score 0.94996 не гарантується.

Вага ансамблю зафіксована до оцінки: **25% ConvNeXt-224 + 25% ConvNeXt-320 +
50% ResNet**. Усереднюються ймовірності, потім застосовуються пороги.

Validation порівнюється тільки на спільному відкладеному fold ResNet із
відповідними OOF-прогнозами ConvNeXt. Інші training-рядки ResNet не оцінюються.
Поточна реалізація вимагає однаковий оригінальний split у всіх гілок.

Пороги ансамблю підбираються на цьому спільному validation fold. Тому його
`fitted_common_validation` F1 оптимістичний; рідкісних позитивних прикладів мало.
Це не незалежна оцінка всього CV та не гарантія кращого Kaggle score.

Результати в `/kaggle/working/convnext_resnet_ensemble`:

- **`submission_convnext_resnet.csv`** — новий експеримент для подачі на Kaggle;
- `submission_convnext_baseline.csv` — відтворення нашого найкращого ConvNeXt blend;
- `validation_comparison.csv`, `validation_per_label.csv` — діагностика;
- ймовірності, спільні validation-прогнози та `ensemble_config.json` — для відтворення.

Автоматичної подачі на Kaggle немає. Після результату submission можна обрати
варіант для остаточного ноутбука.

Локально перевірено CPU-пайплайн на синтетичних даних і малій PyTorch-мережі:
експорт checkpoint, SHA, відповідність validation/test ID, змішування,
відтворення baseline, schema submission, відхилення несумісних файлів та
мультиміткові передбачення. Реальний timm ResNet + Albumentations GPU-інференс
не запускався локально; його потрібно виконати на Kaggle.

Kaggle: [Datasets](https://www.kaggle.com/docs/datasets),
[Notebooks та підключення Input](https://www.kaggle.com/docs/notebooks).
