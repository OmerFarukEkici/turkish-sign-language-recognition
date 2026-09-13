"""
Dogrulama Script'i
==================
Cikarilan .npy landmark dosyalarinin sekil, NaN ve "tumu sifir" kontrolunu yapar.
Egitime gecmeden once veri butunlugunu garanti eder.

KULLANIM:
    python src/dogrulama.py --n_classes 20
"""
import os
import argparse
import glob
import numpy as np

LANDMARKS_DIR = os.path.join("data", "landmarks")
SPLITS = ["train", "val", "test"]
EXPECTED_SHAPE = (30, 195)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_classes", type=int, default=20)
    args = ap.parse_args()

    total_files = 0
    shape_problems = 0
    nan_problems = 0
    all_zero_videos = 0

    for split in SPLITS:
        d = os.path.join(LANDMARKS_DIR, str(args.n_classes), split)
        files = glob.glob(os.path.join(d, "*.npy"))
        print(f"\n[{split}] {len(files)} dosya")
        bad_shape = 0
        for f in files:
            total_files += 1
            try:
                arr = np.load(f)
            except Exception as e:
                print(f"  YUKLENEMEDI: {os.path.basename(f)} ({e})")
                shape_problems += 1
                continue
            if arr.shape != EXPECTED_SHAPE:
                bad_shape += 1
                shape_problems += 1
            if np.isnan(arr).any():
                nan_problems += 1
            if not np.any(arr):
                all_zero_videos += 1
        if bad_shape:
            print(f"  !! Yanlis sekilde {bad_shape} dosya var")
        else:
            print(f"  Tum dosyalar dogru sekilde: {EXPECTED_SHAPE}")

    print(f"\n{'=' * 60}")
    print(f"GENEL:  Toplam dosya: {total_files}")
    print(f"        Sekil sorunu: {shape_problems}, NaN: {nan_problems}, "
          f"Tumu sifir: {all_zero_videos}")
    print(f"{'=' * 60}")

    if shape_problems == 0 and nan_problems == 0:
        print("\nHER SEY YOLUNDA, EGITIME GECEBILIRIZ.")
        print("Sonraki adim: python src/egit_model.py --n_classes", args.n_classes)
    else:
        print("\n!!! Sorun var. Ilgili videolarin landmark cikarimini tekrar deneyin.")


if __name__ == "__main__":
    main()
