# Cats and Dogs++: фінальний ConvNeXt pipeline

Найкращий підтверджений результат нашої моделі — **Kaggle public score 0.95039**.
Це рівне змішування п'ятифолдового ConvNeXt-224 та п'ятифолдового ConvNeXt-320
**без TTA**. Архітектура — `convnext_tiny.fb_in1k`; початкові pretrained weights —
тільки ImageNet-1k.

Готовий submission: [results/submission_convnext_best.csv](../results/submission_convnext_best.csv).
Перевірено всі 18 673 ID, порядок рядків та бінарні мітки. Нове навчання може
дати інший результат; 0.95039 стосується саме збереженого файлу.

## Основні файли

| Файл | Призначення |
|---|---|
| [EDA.ipynb](../EDA.ipynb) | Готовий початковий EDA, без змін |
| [Validation.ipynb](../Validation.ipynb) | Готовий grouped split, без змін |
| [cnn_2_final.ipynb](../cnn_2_final.ipynb) | Фінальний запуск: 320 fine-tuning → OOF → test → blend → backup |
| [ensemble_convnext_resnet.py](../ensemble_convnext_resnet.py) | Явна формула ConvNeXt → ResNet на ймовірностях |
| [cnn_2_ensemble_resnet.ipynb](../cnn_2_ensemble_resnet.ipynb) | Калібрування та submission для ансамблю з ResNet |
| [resnet_export_for_ensemble.ipynb](../resnet_export_for_ensemble.ipynb) | Відновлення ResNet з `.ckpt` та експорт probabilities |
| [REPORT.md](../REPORT.md) | Звіт від готових EDA та validation до фінального результату |
| [results/](../results) | Найкращий CSV, метрики, історія та походження результату |
| [archive/](../archive) | Попередні експерименти та незмінений ноутбук колеги |

## Запуск фінальної моделі в Kaggle

1. Відкрити **`cnn_2_final.ipynb`**, увімкнути GPU.
2. Додати competition **Cats and Dogs++** і Dataset із попереднім
   `convnext_backup.zip`. В Input мають бути `convnext_cv` та `convnext_cv_ensemble`.
   Перший містить п'ять моделей 224, split та OOF; другий — test probabilities.
3. Для W&B надати доступ до Secret `WANDB_API_KEY`. Entity —
   `yrachynska-kyiv-school-of-economics`, project — `cats-and-dogs`.
   Для запуску без W&B встановити `USE_WANDB = False` до старту.
4. Виконати **Run All**. За неоднозначного автопошуку явно вказати
   `SOURCE_CV_DIR` та `SOURCE_PREDICTIONS_DIR` з Input.
5. Основний результат:
   `/kaggle/working/convnext_blend_224_320_fast/submission_blend_224_320.csv`.

Збережено п'ять початкових фолдів. Кожна модель 320 стартує лише зі свого
224-checkpoint. Максимум чотири епохи, EMA-only validation, м'який бюджет
навчання 95 хвилин і резерв 25 хвилин на подальші кроки. Поточна епоха завжди
завершується. Це планування часу, не гарантія двох годин на будь-якому GPU/I/O.

Для відтворення початкових моделей 224 дивись
[single-fold pipeline](../archive/experiments/cnn_2.ipynb) та
[224 CV](../archive/experiments/cnn_2_cv.ipynb). Фінальний notebook очікує їхній
готовий Output і не запускає початкове навчання заново.

Успішний фінальний запуск записав `torch=2.11.0+cu128`, `timm=1.0.29`,
`albumentations=2.0.8`. Інші залежності: NumPy, pandas, scikit-learn, Pillow,
tqdm, matplotlib, IPython та W&B. Збережені параметри —
[results/final_run/cv_config.json](../results/final_run/cv_config.json).

## Збереження та відновлення

Завантажити `convnext_320_fast_inference_backup.zip` із Kaggle Output **до
завершення сесії** та перевірити наявність усіх п'яти `best.pt`.
Створення ZIP у `/kaggle/working` саме по собі не зберігає його поза сесією.
Можна також зберегти версію Kaggle notebook із включеними Output files.

Бекап містить best weights, split, OOF, test probabilities та конфігурації.
`last.pt` зі станами optimizer/EMA/scheduler зберігається в поточній сесії,
але виключений із компактного ZIP. У новій сесії вказати `PREVIOUS_CV_DIR`
на збережений `convnext_cv_320_fast` та залишити ті самі параметри й версії.
Завершені фолди будуть пропущені; для продовження незавершеного фолду потрібен
його `last.pt`. Ваги, ZIP, датасет і API keys не зберігаються в Git.

## Ансамбль з ResNet

Запропонована двоетапна формула реалізована в окремому файлі:

```python
p_convnext = (p_224 + p_320) / 2
p_final = (p_convnext + p_resnet) / 2
```

Ваги — **25% / 25% / 50%**. [ensemble_convnext_resnet.py](../ensemble_convnext_resnet.py)
явно показує обидва кроки та вирівнює CSV за `image_id`.

Від колеги поки є лише binary submission. Для цього ансамблю потрібні її
validation/test probabilities або checkpoint та оригінальний split. Він **ще
не виконаний на реальних ймовірностях ResNet** і не має score. Код відхиляє
суцільно бінарний ResNet CSV, щоб не приписати йому впевненість 0/100%.

Попередній hard-voting experiment знаходиться в
[archive/experiments/cnn_2_vote_resnet.ipynb](../archive/experiments/cnn_2_vote_resnet.ipynb),
його CSV — у [archive/results/convnext_resnet_vote/](../archive/results/convnext_resnet_vote).
Змінено 72 мітки на 64 фото; результат Kaggle цього експерименту ще не наданий.
Підтверджений фінальний score 0.95039 належить нашому ConvNeXt blend.

Детальніше: [інструкція ансамблю](ENSEMBLE_RESNET.md).

## Перевірки

```powershell
python -m unittest discover -s tests -v
```

Перевіряються ваги 25/25/50, незалежність від порядку CSV, ID, відхилення binary
ResNet та неправильних входів. Попередні CPU-тести навчального пайплайна
використовували малу PyTorch-мережу; це не GPU-бенчмарк. Фактичні Kaggle метрики
та історія успішного запуску збережені в `results/`.
