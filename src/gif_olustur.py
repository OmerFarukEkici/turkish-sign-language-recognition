"""
İşaret GIF Üretici
==================
Seçili her sınıf için, AUTSL veri setindeki bir örnek videoyu küçük ve döngülü
bir GIF'e çevirir. Demo bu GIF'leri "İşaretler nasıl yapılır?" bölümünde gösterir.
Telif sorunu yoktur; modelin eğitildiği gerçek veriyi kullanır.

KULLANIM:
    python src/gif_olustur.py --n_classes 5

ÇIKTI:
    data/gifs/{class_id}.gif
"""
import os
import argparse
import pandas as pd

RAW_DIR = os.path.join("data", "raw")
SPLITS_DIR = os.path.join("data", "splits")
GIFS_DIR = os.path.join("data", "gifs")

GENISLIK = 240        # GIF genişliği (piksel); yükseklik orana göre
HEDEF_KARE = 30       # GIF'te yaklaşık kaç kare olsun
KARE_SURE = 80        # her karenin süresi (ms)


def video_bul(sample, splits=("train", "val", "test")):
    for sp in splits:
        for ad in (f"{sample}_color.mp4", f"{sample}.mp4"):
            p = os.path.join(RAW_DIR, sp, ad)
            if os.path.exists(p):
                return p
    return None


def gif_yap(video_path, out_path):
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(video_path)
    kareler = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        kareler.append(frame)
    cap.release()
    if not kareler:
        return False

    # Kareleri seyrelt (yaklaşık HEDEF_KARE adet)
    step = max(1, len(kareler) // HEDEF_KARE)
    secili = kareler[::step][:HEDEF_KARE]

    pil_kareler = []
    for f in secili:
        h, w = f.shape[:2]
        yeni_h = int(h * GENISLIK / w)
        f = cv2.resize(f, (GENISLIK, yeni_h))
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        pil_kareler.append(Image.fromarray(rgb))

    pil_kareler[0].save(
        out_path, save_all=True, append_images=pil_kareler[1:],
        duration=KARE_SURE, loop=0, optimize=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_classes", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(GIFS_DIR, exist_ok=True)

    sec_path = os.path.join(SPLITS_DIR, f"secili_siniflar_{args.n_classes}.csv")
    siniflar = pd.read_csv(sec_path)["class_id"].tolist()

    # Türkçe adlar (sadece bilgi amaçlı yazdırma)
    isim = {}
    sl = os.path.join(SPLITS_DIR, "signlist_TR.csv")
    if os.path.exists(sl):
        df = pd.read_csv(sl)
        isim = dict(zip(df["class_id"], df["isim_tr"]))

    # train listesinden her sınıfa bir örnek seç
    train_df = pd.read_csv(os.path.join(SPLITS_DIR, f"train_{args.n_classes}.csv"))

    print(f"{len(siniflar)} sınıf için GIF üretiliyor...\n")
    for cid in siniflar:
        ad = isim.get(cid, f"Sinif_{cid}")
        ornekler = train_df[train_df["class_id"] == cid]["sample"].tolist()
        yapildi = False
        for sample in ornekler[:5]:        # ilk birkaç örnekten ilk bulunanı kullan
            vp = video_bul(str(sample))
            if vp:
                out = os.path.join(GIFS_DIR, f"{cid}.gif")
                if gif_yap(vp, out):
                    boyut = os.path.getsize(out) // 1024
                    print(f"  ✓ {ad:<10} (id {cid}) -> {out}  [{boyut} KB]")
                    yapildi = True
                    break
        if not yapildi:
            print(f"  ✗ {ad:<10} (id {cid}) -> video bulunamadı")

    print(f"\nBitti. GIF'ler: {GIFS_DIR}")


if __name__ == "__main__":
    main()
