# Real-Time Turkish Sign Language Recognition

Recognizes Turkish Sign Language (TİD) signs from a webcam in real time. MediaPipe Holistic
extracts skeletal landmarks from each frame, a two-layer LSTM classifies the resulting temporal
sequence, and a Streamlit app displays live top-3 predictions with confidence scores.

Graduation project — Istanbul Topkapı University, Software Engineering, 2026.

<!-- Buraya demo GIF'ini ekle:  ![demo](docs/demo.gif)  -->

## How it works

```
webcam frame
   → MediaPipe Holistic      42 hand landmarks + 23 upper-body pose landmarks
   → feature vector          195 dims per frame (126 hand + 69 pose)
   → sequence buffer         30 frames, length-normalized by linear interpolation
   → LSTM classifier         2 layers (128 → 64) + Dropout + BatchNorm + L2
   → softmax                 top-3 predictions with confidence
```

Training uses Adam with categorical cross-entropy, EarlyStopping and ReduceLROnPlateau.
Data augmentation applies Gaussian noise and scaling (3×). Seed is fixed at 42 for reproducibility.

## Results

Trained on a 5-class subset of AUTSL (*ağaç, bebek, ev, söz, yatak*), randomly selected with a
fixed seed. The final model reaches **96.47% test accuracy** (98.8% top-3, macro F1 0.965)
with 220,549 parameters.

| Experiment | Model | Augmentation | Pose | Test acc | Top-3 | Params | Train time |
|---|---|---|---|---|---|---|---|
| **baseline (final)** | LSTM | yes | yes | **96.47%** | 98.8% | 220,549 | 27.4s |
| gru | GRU | yes | yes | 96.47% | 100.0% | 167,301 | 33.9s |
| bilstm | BiLSTM | yes | yes | 94.12% | 100.0% | 506,245 | 39.2s |
| no_aug | LSTM | no | yes | 95.29% | 100.0% | 220,549 | 23.3s |
| no_pose | LSTM | yes | no | 90.59% | 98.8% | 220,549 | 34.3s |

The ablations show that upper-body pose landmarks matter most — removing them costs roughly
6 points of accuracy, since several signs are distinguished by where the hands sit relative to
the body rather than by hand shape alone. GRU matches LSTM with 24% fewer parameters.
BiLSTM adds parameters without helping, which is expected for short single-gesture sequences.

Confusion matrices and training curves for every experiment are in [`results/`](results/).

Live demo runs at roughly 7.8 FPS on CPU.

## Dataset

[AUTSL](https://cvml.ankara.edu.tr/datasets/) (Ankara University Turkish Sign Language) — 226 sign
classes recorded by 43 signers. The dataset is not redistributed here; download it from the
official source and place it under `data/raw/`. The pipeline works with any subset size via
`--n_classes`.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python src/kurulum_kontrol.py   # verifies the installation
```

Requires Python 3.10–3.12.

## Running the pipeline

```bash
# 1. Select an N-class subset (seed=42)
python src/sinif_sec.py --n_classes 5

# 2. Extract landmarks from videos (writes .npy files)
python src/landmark_cikar.py --split train --n_classes 5
python src/landmark_cikar.py --split val   --n_classes 5
python src/landmark_cikar.py --split test  --n_classes 5

# 3. Validate the extracted arrays (shape / NaN / all-zero checks)
python src/dogrulama.py --n_classes 5

# 4. Train
python src/egit_model.py --n_classes 5

# 5. Live webcam demo
streamlit run src/demo_app.py
```

To reproduce the ablation table: `python src/deneyleri_calistir.py`

A pre-trained 5-class model ships in `models/`, so the demo can be run without retraining.

## Project structure

```
src/
  sinif_sec.py          class subset selection
  landmark_cikar.py     MediaPipe landmark extraction
  ozellik.py            feature vector construction
  dogrulama.py          data integrity checks
  egit_model.py         model definition and training
  deneyleri_calistir.py ablation experiments
  demo_app.py           Streamlit live demo
  kesif.py              dataset exploration
models/                 trained weights + metadata
results/                confusion matrices, training curves, comparison table
data/splits/            class lists and train/val/test splits
```

## Limitations

- Trained on 5 of the 226 AUTSL classes. Accuracy drops as the class count grows; the pipeline
  supports larger subsets but they need retraining.
- Signs are classified in isolation — there is no continuous sentence-level segmentation.
- Good lighting and a plain background noticeably improve landmark detection.
- Semantically close signs are the main source of error (for example *sen* / *siz*, which differ
  only slightly in form).

## Tech

Python · TensorFlow/Keras · MediaPipe · OpenCV · Streamlit · scikit-learn · NumPy · pandas
