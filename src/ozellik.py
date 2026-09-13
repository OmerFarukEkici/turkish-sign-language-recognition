"""
Ortak Ozellik Cikarim Modulu
============================
Hem landmark_cikar.py hem demo_app.py bu modulu kullanir; boylece
egitim ve canli demo BIREBIR ayni 195 boyutlu ozellik tanimini paylasir.

Ozellik vektoru (kare basina, 195 boyut):
    - Sol el : 21 nokta x 3 (x,y,z) = 63
    - Sag el : 21 nokta x 3           = 63
    - Ust govde pozu: 23 nokta x 3    = 69   (MediaPipe Pose 0..22)
    Toplam = 63 + 63 + 69 = 195
"""
import numpy as np

N_POSE_UST = 23          # ust govde (MediaPipe pose 0..22)
FEATURE_DIM = 195
SEQUENCE_LENGTH = 30


def extract_keypoints(results):
    """MediaPipe Holistic sonucundan 195 boyutlu ozellik vektoru uretir.
    Gorunmeyen el/poz icin sifir doldurulur."""
    # Sol el (21 x 3)
    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z]
                       for lm in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21 * 3)

    # Sag el (21 x 3)
    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z]
                       for lm in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21 * 3)

    # Ust govde pozu (ilk 23 nokta x 3)
    if results.pose_landmarks:
        pose_all = results.pose_landmarks.landmark
        pose = np.array([[pose_all[i].x, pose_all[i].y, pose_all[i].z]
                         for i in range(N_POSE_UST)]).flatten()
    else:
        pose = np.zeros(N_POSE_UST * 3)

    return np.concatenate([lh, rh, pose]).astype(np.float32)  # 195


def interpolate_to_length(seq, target_len=SEQUENCE_LENGTH):
    """Degisken uzunluktaki (T, 195) diziyi dogrusal interpolasyon ile
    sabit target_len uzunluguna normalize eder."""
    seq = np.asarray(seq, dtype=np.float32)
    T = seq.shape[0]
    if T == target_len:
        return seq
    if T == 0:
        return np.zeros((target_len, FEATURE_DIM), dtype=np.float32)
    if T == 1:
        return np.repeat(seq, target_len, axis=0)
    eski = np.linspace(0, 1, T)
    yeni = np.linspace(0, 1, target_len)
    out = np.zeros((target_len, seq.shape[1]), dtype=np.float32)
    for d in range(seq.shape[1]):
        out[:, d] = np.interp(yeni, eski, seq[:, d])
    return out
