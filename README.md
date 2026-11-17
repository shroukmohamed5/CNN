<img width="1920" height="940" alt="image" src="https://github.com/user-attachments/assets/6f854915-ce4c-4579-8764-eda745a52efc" />

# Vegetable Image Classification with a CNN (From Scratch)

A Convolutional Neural Network built **from scratch** with Keras `Sequential` to classify vegetable images into **15 classes**.
No pretrained models and no transfer learning are used.

This project was completed as a Deep Learning / Computer Vision assignment and is designed to run as a **Kaggle Notebook**.

## Dataset

[Kaggle Vegetable Image Dataset](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset)

```
Vegetable Images/
├── train/
├── validation/
└── test/
```

**15 classes:** Bean, Bitter Gourd, Bottle Gourd, Brinjal, Broccoli, Cabbage, Capsicum, Carrot, Cauliflower, Cucumber, Papaya, Potato, Pumpkin, Radish, Tomato.

Kaggle path used in the notebook:
`/kaggle/input/vegetable-image-dataset/Vegetable Images/`

## Repository contents

| File | Description |
|---|---|
| `vegetable_cnn_classification.ipynb` | The complete notebook (all 12 assignment parts) |
| `README.md` | This file |

## How to run

1. Open [Kaggle](https://www.kaggle.com) and create a new notebook (**File → Import Notebook**, then upload the `.ipynb` file).
2. Click **Add Input** and add the dataset `misrakahmed/vegetable-image-dataset`.
3. In the notebook settings, enable a **GPU** accelerator (recommended; training is much faster).
4. Click **Run All**.

All counts, accuracy values, and loss values are produced by running the notebook. Nothing is hard-coded.

## Pipeline (the 12 parts)

1. Import libraries
2. Define dataset paths (`train_dir`, `test_dir`, `validation_dir`)
3. Count images per class (reusable function returning a dictionary)
4. Build the Train / Test / Validation DataFrames and merge them into one table
5. Grouped bar chart of class distribution and a balance analysis
6. `ImageDataGenerator` for preprocessing and augmentation
7. Load images with `flow_from_directory()` (`train_ds`, `val_ds`)
8. Build the CNN
9. Compile the model and print `model.summary()`
10. Train with EarlyStopping
11. Evaluate on training and validation data
12. Plot accuracy and loss curves

## Model architecture

| Block | Layers |
|---|---|
| Convolution Block 1 | Conv2D(32, 3×3, ReLU, input 224×224×3) → MaxPooling2D |
| Convolution Block 2 | Conv2D(64, 3×3, ReLU) → MaxPooling2D |
| Convolution Block 3 | Conv2D(128, 3×3, ReLU) → MaxPooling2D |
| Classifier | Flatten → Dense(512, ReLU) → Dropout(0.5) → Dense(15, Softmax) |

**Compile:** Adam optimizer, categorical crossentropy loss, accuracy metric.  
**Training:** at most 20 epochs, `EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)`.

## Data preparation

| Generator | Settings |
|---|---|
| Training | rescale 1/255, rotation 20°, width shift 0.1, height shift 0.1, shear 0.1, zoom 0.1, horizontal flip, validation split 0.2 |
| Validation | rescale 1/255 and validation split 0.2 only (no augmentation) |

`flow_from_directory` settings: `target_size=(224, 224)`, `batch_size=32`, `class_mode="categorical"`, `seed=42`.

### Avoiding data leakage

The assignment requires `validation_split` with `subset="training"` / `subset="validation"`, but the dataset also ships with its own `validation` folder.
To keep both requirements without mixing them:

- `train_ds` and `val_ds` both read from `train_dir` with the same `validation_split`, so they are two **disjoint** subsets (80% / 20%). The notebook checks that the class mappings are identical and that no file appears in both subsets.
- The original `validation` folder is never used for training or for the split. It is counted in the analysis and used only as an extra independent check in Part 11.
- The `test` folder is not used for training or model selection.

## Results

Run the notebook to generate the final numbers (training/validation accuracy and loss, learning curves, and the overfitting analysis). You can paste your own results here after running it.

## Requirements

- Python 3
- TensorFlow / Keras
- NumPy, pandas, Matplotlib, Pillow

All of these are already installed in Kaggle Notebooks.
