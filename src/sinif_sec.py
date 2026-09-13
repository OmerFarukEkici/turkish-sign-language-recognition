"""
Sinif Secim Script'i
====================
AUTSL'in 226 sinifindan, sabit seed (42) ile YANSIZ rastgele bir alt kume secer
ve her split icin filtrelenmis CSV uretir. seed=42 -> tekrarlanabilirlik.

KULLANIM:
    python src/sinif_sec.py --n_classes 20
    python src/sinif_sec.py --n_classes 5

CIKTI:
    data/splits/secili_siniflar_{N}.csv      -> secilen class_id listesi
    data/splits/{split}_{N}.csv              -> filtrelenmis ornekler
"""
import os
import argparse
import numpy as np
import pandas as pd

RAW_DIR = os.path.join("data", "raw")
SPLITS_DIR = os.path.join("data", "splits")
SEED = 42
SPLITS = ["train", "val", "test"]


def csv_oku(split):
    path = os.path.join(RAW_DIR, f"{split}_labels.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} bulunamadi. AUTSL CSV'leri data/raw/ icine konmali.")
    df = pd.read_csv(path, header=None)
    df = df.iloc[:, :2]
    df.columns = ["sample", "class_id"]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_classes", type=int, default=20)
    ap.add_argument("--ids", type=str, default="",
                    help="Acik sinif id listesi, orn: --ids 7,68,99,121,132,145,174,211")
    args = ap.parse_args()

    os.makedirs(SPLITS_DIR, exist_ok=True)
    np.random.seed(SEED)

    # train uzerinden mevcut tum siniflari bul
    train_df = csv_oku("train")
    tum_siniflar = sorted(train_df["class_id"].unique())
    print(f"Toplam sinif: {len(tum_siniflar)}")

    if args.ids.strip():
        istenen = sorted({int(x) for x in args.ids.replace(",", " ").split()})
        eksik = [c for c in istenen if c not in set(tum_siniflar)]
        if eksik:
            raise ValueError(f"Bu id'ler veri setinde yok: {eksik}")
        secili = istenen
        args.n_classes = len(secili)
        print(f"ACIK secim ({args.n_classes} sinif): {secili}")
    else:
        if args.n_classes > len(tum_siniflar):
            raise ValueError(f"{args.n_classes} > mevcut {len(tum_siniflar)} sinif")
        secili = sorted(np.random.choice(tum_siniflar, size=args.n_classes, replace=False).tolist())
        print(f"Secilen {args.n_classes} sinif (seed={SEED}): {secili}")

    # secili sinif listesini kaydet
    sec_path = os.path.join(SPLITS_DIR, f"secili_siniflar_{args.n_classes}.csv")
    pd.DataFrame({"class_id": secili}).to_csv(sec_path, index=False)
    print(f"-> {sec_path}")

    # her split'i filtrele
    for split in SPLITS:
        df = csv_oku(split)
        filt = df[df["class_id"].isin(secili)].reset_index(drop=True)
        out = os.path.join(SPLITS_DIR, f"{split}_{args.n_classes}.csv")
        filt.to_csv(out, index=False)
        print(f"[{split}] {len(filt)} ornek -> {out}")

    print("\nTamam. Sonraki adim: python src/landmark_cikar.py --split train --n_classes",
          args.n_classes)


if __name__ == "__main__":
    main()
