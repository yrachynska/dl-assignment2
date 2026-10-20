# Cats and Dogs++

Multi-label image classification. A photo can contain a **Cat**, a **Dog**, and/or one of three individual pets: **Moris**, **Motya**, **Biatrix**. The pets are very rare (about 40-55 photos each), the metric is macro F1.

## Notebooks

| Notebook | What it does |
|---|---|
| `EDA.ipynb` | Looks at the data: labels, class balance, how labels go together, image sizes and colours, sample photos of the rare pets, and copies of the same photo (inside train and between train and test). |
| `Validation.ipynb` | Builds 5 validation folds and saves them to `folds.csv`. |
| `ResNet.ipynb` | Trains the model ResNet152, tunes thresholds, shows wrong predictions, writes `submission.csv`. |

### EDA
- Unique labels, positives per label, labels per image.
- Pets always come with a Cat or Dog label.
- All images have the same size, train and test look alike.
- Copies of one photo are found with a pretrained ResNet18: photos with almost the same embedding (cosine similarity > 0.95) form one group.

### Validation
- Copies of one photo get the same group, so they never end up in different folds.
- `StratifiedGroupKFold` with 5 folds, stratified by the rarest pet on the photo.
- The notebook checks that no photo is split between folds and prints the labels per fold.

### ResNet
ResNet152 is a good fit because it let a deep network train stably and give strong pretrained features for fine details such as fur colour and texture, but with about 60M parameters it is slower than EfficientNet or ConvNeXt-Tiny and can overfit on pets that have only about 40 photos, so we start from ImageNet weights and use a small learning rate and light augmentations.
- ResNet152 pretrained on ImageNet (`timm`), PyTorch Lightning, logging to Weights & Biases.
- Multi-label head: `BCEWithLogitsLoss` with a weight `sqrt(neg / pos)` for every label, so the rare pets are not ignored.
- Light augmentations: flip, small shift, small brightness / contrast.
- `lr = 1e-4` with cosine decay, the best checkpoint is chosen by validation F1.
- One threshold per label is tuned on the validation fold.
- If both Moris and Motya are predicted and at least one of them is below 0.9, only the more probable one is kept.


---

## ConvNeXt: наш pipeline та звіт

Інструкція запуску: [docs/CONVNEXT_PIPELINE.md](docs/CONVNEXT_PIPELINE.md). Окрема копія звіту: [REPORT.md](REPORT.md). Фінальний notebook: [cnn_2_final.ipynb](cnn_2_final.ipynb).

# Звіт: ConvNeXt для Cats and Dogs++

## Результат і межі виконаної роботи

Від готових EDA та validation ми побудували multilabel CNN pipeline на
`convnext_tiny.fb_in1k`, перевірили навчання, налаштували W&B, пройшли від
однієї моделі до п'ятифолдового ансамблю та змішування роздільностей 224/320.
Найкращий підтверджений результат нашого submission — **0.95039 на public
leaderboard Kaggle**. Готовий файл —
[results/submission_convnext_best.csv](results/submission_convnext_best.csv).

Фінальний notebook — [cnn_2_final.ipynb](cnn_2_final.ipynb). Ціль 0.97 на
submission не досягнута. Підготовлений ймовірнісний ансамбль із ResNet ще не
оцінений: від колеги доступний лише binary submission. Генерацію експериментального
голосування не зараховуємо як підтверджене покращення.

Збережені метрики та історія взяті з фактичних Output-архівів Kaggle. Public
scores у цьому звіті зафіксовані зі скріншотів користувача, а не обчислені
локально: test ground truth нам недоступний.

## 1. Початкова база: готові EDA та validation

[EDA.ipynb](EDA.ipynb) і [Validation.ipynb](Validation.ipynb) вже існували до
роботи над CNN. Їх не переписували. EDA містить перевірки таблиць, характеристик
зображень, розподілу класів, структури міток, рідкісних класів і дублікатів.
Групування та split перевикористані з validation notebook.

Збережений `folds.csv` містить **12 260 train-зображень** та **12 186 груп**
схожих фото. Test містить **18 673 ID**. Є п'ять міток:

| Мітка | Позитивних train-прикладів |
|---|---:|
| Cat | 5077 |
| Dog | 5025 |
| Moris | 56 |
| Motya | 42 |
| Biatrix | 41 |

Це **multilabel** задача: одна фотографія може мати кілька міток або жодної.
Є 2162 train-фото з усіма нульовими мітками та 10 фото, де одночасно позитивні
Moris і Motya. Суми позитивних класів не є числом унікальних фото.

Тому початкову вимогу CrossEntropyLoss узгоджено замінили на
**BCEWithLogitsLoss**, а вихід моделі — п'ять незалежних logits. Використання
softmax/argmax або примусової взаємовиключності рідкісних тварин суперечило б
наявним міткам. Ручне перемаркування та зовнішні навчальні дані не додавались.

### Групи й фолди

Оригінальна схема `Validation.ipynb`:

1. Витягти embeddings моделлю `resnet18.a1_in1k` із ImageNet-1k weights.
2. Знайти cosine-neighbours із similarity близько 0.95 та вище і побудувати
   connected components — групи копій/схожих фото.
3. Побудувати stratum: спочатку найрідкісніша позитивна pet-мітка, інакше Cat,
   Dog, їхня комбінація або None.
4. Застосувати `StratifiedGroupKFold(5, shuffle=True, random_state=21)`.
5. Перевірити, що жодна група не перетинає train/validation у межах fold.

| Fold | Фото | Cat | Dog | Moris | Motya | Biatrix |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2448 | 1006 | 982 | 14 | 7 | 9 |
| 1 | 2448 | 1027 | 1007 | 12 | 7 | 7 |
| 2 | 2452 | 1024 | 1002 | 12 | 14 | 5 |
| 3 | 2458 | 1031 | 1003 | 7 | 4 | 14 |
| 4 | 2454 | 989 | 1031 | 11 | 10 | 6 |

У фінальних CV/fine-tuning notebooks новий split не генерується: читається
оригінальний `folds.csv`, перевіряються IDs, labels, group separation та SHA-256
fingerprint split. Це зберігає порівнюваність експериментів.

## 2. Перша CNN і перевірка всього циклу

Основна архітектура весь час залишалась **`convnext_tiny.fb_in1k`** з timm.
Початкове створення моделі використовувало `pretrained=True`, ImageNet-1k
weights та класифікаційну голову на п'ять класів. Варіант із ImageNet-22k не
використовувався. Модель переносилася на CUDA за доступності.

Почали з короткого запуску, щоб перевірити дані, forward/backward, loss,
validation і збереження результатів. Далі виконали повне single-fold навчання
та окремий inference на test. Перший зафіксований submission із фіксованим
порогом 0.5 отримав public score **0.87808**.

У процесі розв'язали операційні проблеми: неправильний шлях до Kaggle Input,
необхідність перезапуску kernel після зміни пакетів, помилки cleanup
multiprocessing DataLoader, відмінність між серверним `/kaggle/working` і
локальними шляхами VS Code. Для стабільного notebook-запуску залишили
`NUM_WORKERS = 0` та звичайний tqdm. Це не змінює сам split або метрики.

## 3. Покращення за зразком ResNet колеги

Оригінальний [ResNet.ipynb](archive/reference/ResNet.ipynb) колеги перенесено
без редагування. У ньому використовується `resnet152.a1_in1k`, fold 0,
224-pixel full-frame resize, mild augmentation, weighted BCE та cosine LR.
Ми використали ці ідеї для **нашого ConvNeXt**, не замінюючи backbone.

Підсумкові спільні принципи preprocessing/training:

- повнокадровий `Resize`, без випадкового відсікання частини тварини;
- horizontal flip із p=0.5;
- affine translation до приблизно ±10% із p=0.5;
- brightness/contrast до приблизно ±10% із p=0.5;
- ImageNet normalization: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`;
- train BCE з `pos_weight = sqrt(negative_count / positive_count)`, обчисленим
  **тільки на training-частині відповідного fold**;
- AdamW, weight decay 0.01, cosine LR та CUDA AMP;
- найкращий checkpoint за validation macro F1 при порозі 0.5, tie-break за loss.

Validation loss у нашому pipeline — unweighted BCE. Тому його абсолютне
значення не слід напряму порівнювати з weighted validation loss ResNet.

Покращений single-fold pipeline отримав **0.93519**. Це результат сукупності
змін; окремих контрольованих абляцій кожної аугментації/optimizer не проводили,
тому причинний внесок одного параметра не визначений.

## 4. Метрики, W&B і пороги

Основна метрика — **macro F1**: F1 обчислюється окремо для кожної мітки, потім
береться просте середнє. Рідкісні Moris/Motya/Biatrix мають таку саму вагу, як
Cat/Dog. Кілька помилок на рідкісному класі можуть помітно змінити macro F1.

Додатково зберігаються:

- train loss — середня weighted BCE за train-фото;
- validation loss — середня unweighted BCE за відкладеними фото;
- validation accuracy — **exact match**: усі п'ять міток фото правильні;
- per-label F1, precision, recall, false positives / false negatives;
- macro ROC-AUC, learning rate, час епохи та early-stopping counter.

У W&B `batch/loss` — loss поточного minibatch, `batch/running_loss` — накопичене
середнє від початку поточної епохи. `val_f1` та `epoch_metrics/macro_f1` у
відповідних runs стосуються validation, а не train. Для спільного порівняння з
колегою додані aliases `val_f1`, `val_loss`, `train_loss_epoch`, `lr`; entity —
`yrachynska-kyiv-school-of-economics`, project — `cats-and-dogs`.
Ключ читається з Kaggle Secret, його значення не записується до репозиторію.

Після вибору checkpoint підбираємо п'ять окремих порогів по grid 0.05–0.95 із
кроком 0.01. За однакового F1 обирається поріг, найближчий до 0.5; якщо у
калібрувальних targets немає обох outcomes, залишається 0.5. Test labels у
підборі не використовуються.

## 5. TTA та п'ятифолдовий CV на 224

TTA тут означає середнє **sigmoid probabilities** оригінального фото та його
горизонтального flip. Це не нове навчання. Single-fold TTA отримав **0.93720**,
трохи вище за 0.93519 без TTA.

Далі навчали п'ять незалежних моделей. Для кожного фото OOF-прогноз створювала
тільки та модель, яка не бачила відповідний fold у train. На test усереднювали
ймовірності всіх п'яти моделей. Fold 0 можна було повторно використати лише
після перевірки його validation IDs та конфігурації; інші fold-моделі не
стартували з уже навчених ваг чужого fold.

| 224 OOF | Macro F1 при 0.5 | Macro F1 після підбору порогів |
|---|---:|---:|
| Original | 0.976454 | 0.978271 |
| TTA | 0.969678 | 0.974188 |

На public leaderboard результати були **0.93994 original** та **0.94275 TTA**.
Отже, ранжування за OOF і public score не завжди збігалося. TTA не можна
вважати гарантованим покращенням для всіх наступних конфігурацій.

## 6. Fine-tuning на 320 і змішування роздільностей

Наступний етап — 320×320 full-frame resize, той самий ConvNeXt та той самий
split. Кожна модель 320 відновлювала **власний matching-fold checkpoint 224**.
`pretrained=False` у цьому loader означає, що всю навчену state_dict завантажено
з checkpoint; випадкова ініціалізація не використовується для fine-tuning.

Застосували малий backbone LR, більший head LR, warmup і exponential moving
average ваг (EMA). EMA — не усереднення test-прогнозів і не TTA: після кожного
успішного optimizer step оновлюється згладжена копія параметрів моделі.
У повному попередньому run порівнювали raw/EMA, максимум шість епох на fold.
Після test inference змішали 224 та 320 probabilities 50/50 та підібрали
пороги на відповідному OOF mean. Цей original-only blend отримав **0.94996**.

Простий resize старих ваг на 320 без донавчання погіршував validation F1.
Тому додаткова роздільність сама по собі не була готовим покращенням.

## 7. Фінальний скорочений pipeline

Через бюджет близько двох годин підготували коротший окремий run, який став
фінальним [cnn_2_final.ipynb](cnn_2_final.ipynb). Архітектуру, фолди,
аугментації, loss та визначення метрик збережено.

| Параметр | Фінальний варіант |
|---|---|
| Image size | 320×320 |
| Fold-моделей | 5 |
| Максимум епох на fold | 4 |
| Early-stopping patience | 2 |
| Batch / accumulation | 16 / 2, effective batch близько 32 |
| Backbone / head peak LR | 1e-5 / 5e-5 |
| Warmup | 1 епоха, далі cosine per optimizer step |
| Backbone minimum LR | 2e-7 |
| EMA decay | 0.99 |
| Optimizer / weight decay | AdamW / 0.01 |
| AMP / memory format | CUDA AMP / channels_last |
| Workers | 0 |
| TTA | вимкнено |

Після кожної епохи оцінюється тільки EMA, що прибирає другий validation pass.
Epoch 0 також залишається кандидатом. Best checkpoint містить validation
probabilities і точні IDs, тому OOF можна експортувати без повторного forward.
На test кожна модель виконує один original-only pass; 224 test probabilities
читаються з бекапу. Soft budget розподіляє залишок часу між фолдами та
завершує поточну епоху перед зупинкою.

Фактичний успішний run завершив **19 епох загалом**. Сума записаних
`epoch_seconds` — **56.60 хвилини**. Це не повний wall time: поза ним залишилися
setup, збереження checkpoint, test inference, ZIP та завантаження.

| Fold | F1 вихідної моделі 224 при 0.5 | Найкращий F1 320 при 0.5 | Best epoch |
|---:|---:|---:|---:|
| 0 | 0.972387 | 0.972481 | 4 |
| 1 | 0.974429 | 0.981466 | 1 |
| 2 | 0.989132 | 0.990119 | 2 |
| 3 | 0.989916 | 0.989627 | 3 |
| 4 | 0.967023 | 0.962025 | 4 |

Це не покращення кожного fold: 3 і 4 трохи/помітно нижчі від своїх джерел 224.
У короткому run raw-модель окремо не оцінюється, тому він не доводить перевагу
EMA над raw для кожної епохи.

## 8. Фінальний ансамбль та результати

Спочатку рівно усереднюються fold-моделі всередині кожної роздільності, далі
224 та 320 утворюють одну ConvNeXt-гілку:

```python
# p_224_folds і p_320_folds: sigmoid probabilities, shape (5, N, 5).
p_224 = np.mean(p_224_folds, axis=0)
p_320 = np.mean(p_320_folds, axis=0)
p_convnext = (p_224 + p_320) / 2
predictions = p_convnext >= thresholds
```

Фінальні пороги `[Cat, Dog, Moris, Motya, Biatrix]`:
**`[0.58, 0.42, 0.56, 0.50, 0.50]`**.

| Original-only варіант | OOF F1 при 0.5 | OOF F1 після підбору порогів | ROC-AUC |
|---|---:|---:|---:|
| 224 CV | 0.976454 | 0.978271 | 0.999723 |
| Fast 320 CV | 0.976476 | 0.978790 | 0.999757 |
| Фінальний 224/320 blend | 0.979278 | 0.981114 | 0.999829 |

Exact-match accuracy фінального blend після порогів — **0.993475** на OOF.
Збережені таблиці — [results/final_run/](results/final_run/).

| Етап | Kaggle public score |
|---|---:|
| Початковий single-fold, threshold 0.5 | 0.87808 |
| Покращений single-fold | 0.93519 |
| Single-fold + TTA | 0.93720 |
| CV 224 без TTA | 0.93994 |
| CV 224 + TTA | 0.94275 |
| Попередній 224/320 blend без TTA | 0.94996 |
| **Фінальний fast 224/320 blend без TTA** | **0.95039** |
| Голосування ConvNeXt-224 / ConvNeXt-320 / ResNet | 0.94854 |

Окремий reference колеги ResNet мав reported public score 0.94888. Він не є
результатом нашого ConvNeXt або вже виконаного ймовірнісного ансамблю.

Високий fitted OOF F1 не означає такий самий test score. Checkpoint selection і
пороги використовують validation labels; для повністю незалежної оцінки потрібен
окремий holdout або nested evaluation. Мала кількість rare-label positives
підсилює нестабільність. У попередньому blend notebook також перевіряли
cross-fitted thresholds, але це не усуває весь bias від вибору checkpoint.

## 9. ResNet: що зроблено і що ще потрібно

Підготовлено окремий явний код [ensemble_convnext_resnet.py](ensemble_convnext_resnet.py):

```python
p_convnext = (p_224 + p_320) / 2.0
p_final = (p_convnext + p_resnet) / 2.0
```

Це ваги 25% / 25% / 50%. До них потрібно застосувати нові пороги, підібрані
на відповідних спільних validation-прогнозах. Для цього є
[probability ensemble notebook](cnn_2_ensemble_resnet.ipynb) та
[ResNet exporter](resnet_export_for_ensemble.ipynb). Loader використовує trained
`resnet152.a1_in1k` checkpoint, strict state loading, її preprocessing та sigmoid.
Не потрібні повторне навчання або інша pretrained-модель, якщо ваги збережені.

Отриманий `submission (6).csv` колеги має 18 673 унікальних IDs, правильні
колонки, без пропусків, але лише 0/1. Із них не відновлюються ймовірності чи ваги.
Усереднення таких значень із нашими probabilities при порозі 0.5 переважно
відтворювало б binary ResNet. Тому основний probability-код відхиляє цей вхід.

Як окремий експеримент створили majority vote трьох гілок 224 / 320 / binary
ResNet. Наш baseline 0.95039 перед цим відтворено точно. Голосування змінює
72 рішення щодо міток на 64 фото; без ResNet validation-прогнозів ці зміни
не можна вважати виправленими помилками. Public score голосування — **0.94854**,
тобто на 0.00185 нижче за наш blend 0.95039. Голосування не обране фінальним.
Код та CSV збережені в `archive/`, а не обрані фінальною моделлю.

## 10. Збереження, відтворення та перевірки

Після втрати незбережених Output попереднього 320-run відновилися зі старого
224 backup та повторили тільки етап 320. Це виявило різницю між текстовими
output клітинок, тимчасовими файлами Kaggle-сесії та реально збереженим ZIP/Dataset.

Фінальний notebook оновлює компактний inference backup після кожного завершеного
fold, а після test додає probabilities, configs та submissions. ZIP містить best
weights, але не великі optimizer states чи W&B logs. Для незавершеної епохи
потрібен `last.pt`; для відтворення completed inference достатньо best weights
і маніфестів. Сам ZIP у working все ще треба завантажити або включити до Output
збереженої Kaggle-версії перед закриттям сесії.

Локально перевіряли pipeline на малій CPU PyTorch-мережі з doubles для відсутніх
timm/Albumentations: accumulation, EMA, OOF correspondence, checkpoint hashes,
epoch-boundary resume, test caches, time-budget handling, ZIP і schema submission.
Реальні ConvNeXt GPU результати в цьому звіті — із виконаного користувачем Kaggle-run,
а не з цих CPU-тестів.

Для окремої формули ансамблю додані постійні unit tests на арифметику 25/25/50,
перестановки IDs, CSV round-trip, shape/range/NaN та відхилення binary ResNet.
EDA, validation і перенесений reference ResNet перевірені за SHA-256 на відсутність
змін. Ваги, ZIP та API keys виключені з Git; код, звіт, історія, конфігурації й
найкращий submission зберігаються в репозиторії. Джерела зафіксовані в
[results/provenance.json](results/provenance.json).


### Інтеграція з командною main

Після підготовки фінального pipeline до нашої гілки інтегровано актуальну main. Файли EDA.ipynb, Validation.ipynb, ResNet.ipynb та ai_usage.txt залишені точно як у main колеги. Її README збережений незмінним початковим блоком; цей звіт доданий після нього. Інструкцію запуску ConvNeXt винесено до docs/CONVNEXT_PIPELINE.md.
