"""
Landmark Cikarim Script'i
=========================
AUTSL videolarindan MediaPipe Holistic ile kare basina 195 boyutlu ozellik
cikarir, dogrusal interpolasyon ile 30 kareye normalize eder ve .npy kaydeder.
Yarida kesilirse kaldigi yerden devam eder (resume).

KULLANIM:
    python src/landmark_cikar.py --split train --n_classes 20
    python src/landmark_cikar.py --split val   --n_classes 20
    python src/landmark_cikar.py --split test  --n_classes 20

CIKTI:
    data/landmarks/{n_classes}/{split}/{sample}.npy   -> sekil (30, 195)
"""
import os
import argparse
import numpy as np
import pandas as pd

from ozellik import extract_keypoints, interpolate_to_length, SEQUENCE_LENGTH, FEATURE_DIM

RAW_DIR = os.path.join("data", "raw")
SPLITS_DIR = os.path.join("data", "splits")
LANDMARKS_DIR = os.path.join("data", "landmarks")


def video_to_sequence(holistic, video_path):
    """Bir videoyu (30, 195) diziye donusturur."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        frames.append(extract_keypoints(results))
    cap.release()
    if not frames:
        return np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)
    return interpolate_to_length(np.array(frames), SEQUENCE_LENGTH)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--n_classes", type=int, default=20)
    args = ap.parse_args()

    import mediapipe as mp

    csv_path = os.path.join(SPLITS_DIR, f"{args.split}_{args.n_classes}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} yok. Once: python src/sinif_sec.py --n_classes "
                                f"{args.n_classes}")
    df = pd.read_csv(csv_path)

    out_dir = os.path.join(LANDMARKS_DIR, str(args.n_classes), args.split)
    os.makedirs(out_dir, exist_ok=True)

    vid_dir = os.path.join(RAW_DIR, args.split)
    mp_holistic = mp.solutions.holistic

    toplam = len(df)
    atlanan = 0
    islenen = 0
    hata = 0

    print(f"[{args.split}] {toplam} ornek islenecek -> {out_dir}")

    with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:

        for i, row in df.iterrows():
            sample = str(row["sample"])
            out_path = os.path.join(out_dir, f"{sample}.npy")

            # resume: zaten varsa ve sekli dogruysa atla
            if os.path.exists(out_path):
                try:
                    arr = np.load(out_path)
                    if arr.shape == (SEQUENCE_LENGTH, FEATURE_DIM):
                        atlanan += 1
                        continue
                except Exception:
                    pass

            # AUTSL dosyalari "_color" ekiyle gelir: signer0_sample1_color.mp4
            video_path = os.path.join(vid_dir, f"{sample}_color.mp4")
            if not os.path.exists(video_path):
                alt = os.path.join(vid_dir, f"{sample}.mp4")  # ek olmadan da dene
                if os.path.exists(alt):
                    video_path = alt
                else:
                    hata += 1
                    continue
            try:
                seq = video_to_sequence(holistic, video_path)
                np.save(out_path, seq)
                islenen += 1
            except Exception as e:
                print(f"  HATA {sample}: {e}")
                hata += 1

            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{toplam} | islenen={islenen} atlanan={atlanan} hata={hata}")

    print(f"\n[{args.split}] BITTI -> islenen={islenen}, atlanan(resume)={atlanan}, hata={hata}")
    print("Sonraki adim: python src/dogrulama.py --n_classes", args.n_classes)


if __name__ == "__main__":
    main()
