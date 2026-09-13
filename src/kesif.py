"""
AUTSL Kesif Script'i
====================
Indirilen AUTSL veri setinin CSV etiketlerini ve video formatini analiz eder.
Veri kumesi yapisini anlamak ve sonraki adimlara hazirlanmak icin calistirilir.

KULLANIM:
    python src/kesif.py
"""
import os
import glob
import pandas as pd

RAW_DIR = os.path.join("data", "raw")
SPLITS = ["train", "val", "test"]


def csv_analiz():
    print("=" * 60)
    print("ETIKET CSV ANALIZI")
    print("=" * 60)
    for split in SPLITS:
        csv_path = os.path.join(RAW_DIR, f"{split}_labels.csv")
        if not os.path.exists(csv_path):
            print(f"[{split}] CSV bulunamadi: {csv_path}")
            continue
        # AUTSL CSV'leri genelde basliksiz: sample_name, class_id
        df = pd.read_csv(csv_path, header=None)
        if df.shape[1] >= 2:
            df.columns = ["sample", "class_id"] + list(df.columns[2:])
        n_classes = df["class_id"].nunique()
        print(f"[{split}] {len(df)} ornek | {n_classes} farkli sinif")
        print(f"        Sinif basina ortalama: {len(df) / max(n_classes,1):.1f} ornek")


def video_analiz():
    print("\n" + "=" * 60)
    print("VIDEO FORMAT ANALIZI")
    print("=" * 60)
    try:
        import cv2
    except ImportError:
        print("OpenCV yok, video analizi atlandi.")
        return

    for split in SPLITS:
        vid_dir = os.path.join(RAW_DIR, split)
        vids = glob.glob(os.path.join(vid_dir, "*.mp4"))
        if not vids:
            print(f"[{split}] video bulunamadi: {vid_dir}")
            continue
        ornek = vids[0]
        cap = cv2.VideoCapture(ornek)
        fps = cap.get(cv2.CAP_PROP_FPS)
        n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        print(f"[{split}] {len(vids)} video | ornek: {os.path.basename(ornek)}")
        print(f"        {w}x{h}, {fps:.1f} FPS, ~{n_frame} kare")


if __name__ == "__main__":
    csv_analiz()
    video_analiz()
    print("\nKesif tamam. Sonraki adim: python src/sinif_sec.py --n_classes 20")
