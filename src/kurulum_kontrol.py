"""
Kurulum Kontrol Script'i
========================
Python / TensorFlow / GPU / MediaPipe / OpenCV surumlerini dogrular.

KULLANIM:
    python src/kurulum_kontrol.py
"""
import sys
import platform


def baslik(s):
    print("\n" + "=" * 60)
    print(s)
    print("=" * 60)


def main():
    baslik("KURULUM KONTROL")
    print(f"Python      : {platform.python_version()}")
    print(f"Platform    : {platform.system()} {platform.release()}")

    sorun = []

    # NumPy
    try:
        import numpy as np
        print(f"NumPy       : {np.__version__}")
    except Exception as e:
        sorun.append(f"NumPy yuklenemedi: {e}")

    # TensorFlow
    try:
        import tensorflow as tf
        print(f"TensorFlow  : {tf.__version__}")
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            print(f"GPU         : {len(gpus)} adet -> {[g.name for g in gpus]}")
        else:
            print("GPU         : YOK (CPU ile calisilacak - egitim yine de hizli)")
    except Exception as e:
        sorun.append(f"TensorFlow yuklenemedi: {e}")

    # MediaPipe
    try:
        import mediapipe as mp
        print(f"MediaPipe   : {mp.__version__}")
    except Exception as e:
        sorun.append(f"MediaPipe yuklenemedi: {e}")

    # OpenCV
    try:
        import cv2
        print(f"OpenCV      : {cv2.__version__}")
    except Exception as e:
        sorun.append(f"OpenCV yuklenemedi: {e}")

    # scikit-learn
    try:
        import sklearn
        print(f"scikit-learn: {sklearn.__version__}")
    except Exception as e:
        sorun.append(f"scikit-learn yuklenemedi: {e}")

    # Streamlit
    try:
        import streamlit
        print(f"Streamlit   : {streamlit.__version__}")
    except Exception as e:
        sorun.append(f"Streamlit yuklenemedi: {e}")

    baslik("SONUC")
    if sorun:
        print("!!! SORUNLAR:")
        for s in sorun:
            print(f"  - {s}")
        print("\n-> 'pip install -r requirements.txt' calistirin.")
        sys.exit(1)
    else:
        print("Her sey yolunda. Pipeline'a gecebilirsiniz.")


if __name__ == "__main__":
    main()
