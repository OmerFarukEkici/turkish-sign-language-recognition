"""
Karsilastirma Deneyleri Wrapper
===============================
egit_model.py'yi 4 farkli ayarla sirayla calistirir ve rapora hazir
karsilastirma tablosu uretir.

Deneyler:
    1. GRU       - Mimari karsilastirma (basit alternatif)
    2. BiLSTM    - Mimari karsilastirma (cift yonlu)
    3. No-Aug    - Augmentation kapali (vize sozunun kaniti)
    4. No-Pose   - Poz noktalari sifirli (sadece el - poz katkisinin olcumu)

KULLANIM:
    python src/deneyleri_calistir.py --n_classes 5

CIKTI:
    results/karsilastirma_tablosu.md    <- Rapor icin hazir markdown
    results/karsilastirma_tablosu.csv   <- Excel'de acilabilir
"""
import os
import sys
import json
import time
import argparse
import subprocess

MODELS_DIR = "models"
RESULTS_DIR = "results"

DENEYLER = [
    {"ad": "gru",     "aciklama": "GRU mimari (LSTM yerine)",
     "args": ["--model_type", "GRU", "--exp_name", "gru"]},
    {"ad": "bilstm",  "aciklama": "BiLSTM (cift yonlu)",
     "args": ["--model_type", "BiLSTM", "--exp_name", "bilstm"]},
    {"ad": "no_aug",  "aciklama": "Augmentation kapali",
     "args": ["--no_aug", "--exp_name", "no_aug"]},
    {"ad": "no_pose", "aciklama": "Poz noktalari sifirli (sadece el)",
     "args": ["--no_pose", "--exp_name", "no_pose"]},
]


def meta_oku(etiket):
    path = os.path.join(MODELS_DIR, f"model_meta_{etiket}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def satir(ad, aciklama, meta):
    if meta is None:
        return {"deney": ad, "aciklama": aciklama, "model": "?",
                "aug": "?", "poz": "?", "test_acc": 0.0, "top3": 0.0,
                "params": 0, "sure": 0.0}
    return {"deney": ad, "aciklama": aciklama,
            "model": meta.get("model_type", "?"),
            "aug": "VAR" if meta.get("augmentation", True) else "YOK",
            "poz": "VAR" if meta.get("use_pose", True) else "YOK",
            "test_acc": meta["test_accuracy"], "top3": meta["test_top3_accuracy"],
            "params": meta.get("total_params", 0),
            "sure": meta.get("training_time_sec", 0.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_classes", type=int, default=5)
    args = ap.parse_args()
    N = args.n_classes
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print(f"KARSILASTIRMA DENEYLERI ({len(DENEYLER)} adet, {N} sinif)")
    print("=" * 70)

    egit_py = os.path.join(os.path.dirname(__file__), "egit_model.py")

    # 0) Baseline
    baseline = meta_oku(f"{N}class")
    if baseline is None:
        print(f"UYARI: Baseline (model_meta_{N}class.json) yok.")
        print(f"Once calistir: python src/egit_model.py --n_classes {N}")
        sys.exit(1)
    sonuclar = [satir("baseline", "LSTM (final model)", baseline)]

    t0 = time.time()
    for i, d in enumerate(DENEYLER, 1):
        print(f"\n{'#'*70}\nDENEY {i}/{len(DENEYLER)}: {d['aciklama']}\n{'#'*70}")
        cmd = [sys.executable, egit_py, "--n_classes", str(N)] + d["args"]
        print("Komut:", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"  !! Deney patladi ({d['ad']}): {e} — devam ediliyor.")
        etiket = f"{N}class_{d['ad']}"
        sonuclar.append(satir(d["ad"], d["aciklama"], meta_oku(etiket)))

    # tablo uret
    md = ["| Deney | Aciklama | Model | Aug | Poz | Test Acc | Top-3 | Params | Sure(s) |",
          "|---|---|---|---|---|---|---|---|---|"]
    csv = ["deney,aciklama,model,aug,poz,test_acc,top3,params,sure_sn"]
    for s in sonuclar:
        md.append(f"| {s['deney']} | {s['aciklama']} | {s['model']} | {s['aug']} | "
                  f"{s['poz']} | %{s['test_acc']*100:.2f} | %{s['top3']*100:.1f} | "
                  f"{s['params']:,} | {s['sure']:.1f} |")
        csv.append(f"{s['deney']},{s['aciklama']},{s['model']},{s['aug']},{s['poz']},"
                   f"{s['test_acc']:.4f},{s['top3']:.4f},{s['params']},{s['sure']:.1f}")

    with open(os.path.join(RESULTS_DIR, "karsilastirma_tablosu.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(RESULTS_DIR, "karsilastirma_tablosu.csv"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(csv) + "\n")

    print("\n" + "\n".join(md))
    print(f"\nToplam sure: {(time.time()-t0)/60:.1f} dk")
    print("-> results/karsilastirma_tablosu.md  (rapora yapistirmaya hazir)")


if __name__ == "__main__":
    main()
