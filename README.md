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

