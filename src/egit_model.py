"""
LSTM/GRU/BiLSTM Model Egitim Script'i (AUTSL - Cok Parametrik)
==============================================================
Sinif sayisini, model tipini ve ozellik kombinasyonlarini komut satirindan
secebilirsiniz. Vize raporu sozleri: Dropout + BatchNorm + Veri artirma + EarlyStopping.

KULLANIM ORNEKLERI:
    # Final model (LSTM, augmentation acik, el+poz birlikte)
    python src/egit_model.py --n_classes 20

    # GRU mimari karsilastirmasi
    python src/egit_model.py --n_classes 5 --model_type GRU --exp_name gru

    # BiLSTM (cift yonlu)
    python src/egit_model.py --n_classes 5 --model_type BiLSTM --exp_name bilstm

    # Augmentation kapali (vize sozunun ampirik kaniti)
    python src/egit_model.py --n_classes 5 --no_aug --exp_name no_aug

    # Sadece el (poz noktalari sifirlanir - poz katkisinin olcumu)
    python src/egit_model.py --n_classes 5 --no_pose --exp_name no_pose
"""
import os
import json
import time
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (LSTM, GRU, Bidirectional, Dense, Dropout,
                                      BatchNormalization, Masking, Input)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau, CSVLogger)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- SABITLER ----
SEQUENCE_LENGTH = 30
FEATURE_DIM = 195
POSE_DIM = 69                 # son 69 boyut = ust govde pozu
SPLITS_DIR = os.path.join("data", "splits")
LANDMARKS_DIR = os.path.join("data", "landmarks")
MODELS_DIR = "models"
RESULTS_DIR = "results"
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ---------------------------------------------------------------- veri yukleme
def label_haritasi(n_classes):
    """secili_siniflar_{N}.csv -> {orijinal_class_id: 0..N-1}"""
    path = os.path.join(SPLITS_DIR, f"secili_siniflar_{n_classes}.csv")
    secili = pd.read_csv(path)["class_id"].tolist()
    return {c: i for i, c in enumerate(sorted(secili))}, sorted(secili)


def load_split(split, n_classes, harita):
    csv_path = os.path.join(SPLITS_DIR, f"{split}_{n_classes}.csv")
    df = pd.read_csv(csv_path)
    base = os.path.join(LANDMARKS_DIR, str(n_classes), split)
    X, y = [], []
    for _, row in df.iterrows():
        sample, cid = str(row["sample"]), int(row["class_id"])
        npy = os.path.join(base, f"{sample}.npy")
        if not os.path.exists(npy):
            continue
        arr = np.load(npy)
        if arr.shape != (SEQUENCE_LENGTH, FEATURE_DIM):
            continue
        X.append(arr)
        y.append(harita[cid])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# ---------------------------------------------------------------- augmentation
def augment(X, y, kat=3):
    """Gauss gurultusu + olcekleme ile veri artirma (3 kat)."""
    X_aug, y_aug = [X], [y]
    rng = np.random.RandomState(SEED)
    for k in range(kat - 1):
        gurultu = rng.normal(0, 0.01, X.shape).astype(np.float32)
        olcek = rng.uniform(0.95, 1.05, (X.shape[0], 1, 1)).astype(np.float32)
        X_aug.append(X * olcek + gurultu)
        y_aug.append(y)
    return np.concatenate(X_aug, axis=0), np.concatenate(y_aug, axis=0)


def poz_sifirla(X):
    """Son POSE_DIM boyutu sifirlar (sadece el bilgisiyle calismak icin)."""
    X = X.copy()
    X[:, :, FEATURE_DIM - POSE_DIM:] = 0.0
    return X


# ---------------------------------------------------------------- model
def build_model(n_classes, model_type="LSTM"):
    model = Sequential(name=f"{model_type}_{n_classes}sinif")
    model.add(Input(shape=(SEQUENCE_LENGTH, FEATURE_DIM)))
    model.add(Masking(mask_value=0.0))

    def rnn(birim, return_seq):
        if model_type == "GRU":
            return GRU(birim, return_sequences=return_seq, kernel_regularizer=l2(1e-4))
        layer = LSTM(birim, return_sequences=return_seq, kernel_regularizer=l2(1e-4))
        if model_type == "BiLSTM":
            return Bidirectional(layer)
        return layer

    model.add(rnn(128, True))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(rnn(64, False))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu", kernel_regularizer=l2(1e-4)))
    model.add(Dropout(0.3))
    model.add(Dense(n_classes, activation="softmax"))
    return model


# ---------------------------------------------------------------- gorseller
def egitim_egrisi(history, out_png):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history.history["loss"], label="train")
    ax[0].plot(history.history["val_loss"], label="val")
    ax[0].set_title("Kayip"); ax[0].set_xlabel("Epoch"); ax[0].legend()
    ax[1].plot(history.history["accuracy"], label="train")
    ax[1].plot(history.history["val_accuracy"], label="val")
    ax[1].set_title("Dogruluk"); ax[1].set_xlabel("Epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)


def karisiklik_matrisi(y_true, y_pred, n_classes, out_png):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    fig, ax = plt.subplots(figsize=(max(6, n_classes * 0.6),
                                    max(5, n_classes * 0.5)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Tahmin"); ax.set_ylabel("Gercek")
    ax.set_title("Karisiklik Matrisi")
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=8)
    fig.colorbar(im); fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)
    return cm


# ---------------------------------------------------------------- ana akis
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_classes", type=int, default=20)
    ap.add_argument("--model_type", choices=["LSTM", "GRU", "BiLSTM"], default="LSTM")
    ap.add_argument("--no_aug", action="store_true")
    ap.add_argument("--no_pose", action="store_true")
    ap.add_argument("--exp_name", default="")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ek = f"_{args.exp_name}" if args.exp_name else ""
    etiket = f"{args.n_classes}class{ek}"

    harita, secili = label_haritasi(args.n_classes)
    print(f"Siniflar (orijinal id) -> 0..{args.n_classes-1}: {secili}")

    print("Veri yukleniyor...")
    X_train, y_train = load_split("train", args.n_classes, harita)
    X_val, y_val = load_split("val", args.n_classes, harita)
    X_test, y_test = load_split("test", args.n_classes, harita)
    print(f"  train={X_train.shape}  val={X_val.shape}  test={X_test.shape}")

    if args.no_pose:
        print("--no_pose: poz noktalari sifirlaniyor (sadece el).")
        X_train, X_val, X_test = map(poz_sifirla, (X_train, X_val, X_test))

    if not args.no_aug:
        print("Veri artirma (3 kat) uygulaniyor...")
        X_train, y_train = augment(X_train, y_train, kat=3)
        print(f"  artirilmis train={X_train.shape}")
    else:
        print("--no_aug: veri artirma KAPALI.")

    y_train_c = to_categorical(y_train, args.n_classes)
    y_val_c = to_categorical(y_val, args.n_classes)
    y_test_c = to_categorical(y_test, args.n_classes)

    model = build_model(args.n_classes, args.model_type)
    model.compile(optimizer=Adam(1e-3),
                  loss="categorical_crossentropy",
                  metrics=["accuracy",
                           TopKCategoricalAccuracy(k=3, name="top3")])
    model.summary()

    best_path = os.path.join(MODELS_DIR, f"best_model_{etiket}.keras")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5),
        ModelCheckpoint(best_path, monitor="val_accuracy", save_best_only=True),
        CSVLogger(os.path.join(RESULTS_DIR, f"egitim_log_{etiket}.csv")),
    ]

    t0 = time.time()
    history = model.fit(X_train, y_train_c,
                        validation_data=(X_val, y_val_c),
                        epochs=args.epochs, batch_size=args.batch_size,
                        callbacks=callbacks, verbose=2)
    egitim_sn = time.time() - t0

    # degerlendirme
    test_loss, test_acc, test_top3 = model.evaluate(X_test, y_test_c, verbose=0)
    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
    rapor = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    macro_f1 = rapor["macro avg"]["f1-score"]

    print("\n" + "=" * 60)
    print(f"SONUC ({etiket}, {args.model_type})")
    print(f"  Test dogruluk : %{test_acc*100:.2f}")
    print(f"  Top-3         : %{test_top3*100:.2f}")
    print(f"  Macro F1      : {macro_f1:.3f}")
    print(f"  Egitim suresi : {egitim_sn:.1f} sn")
    print("=" * 60)

    # gorseller
    egitim_egrisi(history, os.path.join(RESULTS_DIR, f"training_curves_{etiket}.png"))
    karisiklik_matrisi(y_test, y_pred, args.n_classes,
                       os.path.join(RESULTS_DIR, f"confusion_matrix_{etiket}.png"))

    # meta JSON (deneyleri_calistir.py bunu okur)
    meta = {
        "etiket": etiket,
        "n_classes": args.n_classes,
        "model_type": args.model_type,
        "augmentation": (not args.no_aug),
        "use_pose": (not args.no_pose),
        "test_accuracy": float(test_acc),
        "test_top3_accuracy": float(test_top3),
        "macro_f1": float(macro_f1),
        "total_params": int(model.count_params()),
        "training_time_sec": float(egitim_sn),
        "secili_siniflar": secili,
    }
    with open(os.path.join(MODELS_DIR, f"model_meta_{etiket}.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nModel  -> {best_path}")
    print(f"Meta   -> models/model_meta_{etiket}.json")
    print(f"Gorsel -> results/confusion_matrix_{etiket}.png, training_curves_{etiket}.png")


if __name__ == "__main__":
    main()
