"""
Gerçek Zamanlı Türk İşaret Dili Tanıma — Streamlit Canlı Demo
=============================================================
Web kamerasından MediaPipe Holistic ile 195 boyutlu özellik çıkarılır, 30 karelik
tampon dolunca LSTM modeli tahmin yapar. Tahmin güven eşiğini geçerse kameranın
yanındaki panelde gösterilir; geçmezse panel boş kalır.

Çalıştırma:
    streamlit run src/demo_app.py
"""
import os
import json
import time
import threading
import collections
import numpy as np
import streamlit as st

from ozellik import extract_keypoints, SEQUENCE_LENGTH

st.set_page_config(page_title="Türk İşaret Dili Tanıma", page_icon="🤟",
                   layout="wide", initial_sidebar_state="expanded")

MODELS_DIR = "models"
SPLITS_DIR = os.path.join("data", "splits")
GIFS_DIR = os.path.join("data", "gifs")
_GUVEN_ESIGI = [0.70]   # kamera iş parçacığı ile paylaşılan eşik
# Zamansal yumuşatma: tek karelik anlık tahminler yerine son birkaç tahmin
# birlikte değerlendirilir. Böylece işaret YAPILIRKEN oluşan geçiş/yarım
# hareketler ekrana yansımaz; tahmin ancak hareket KARARLI hale gelince gösterilir.
SMOOTH_WINDOW = 10   # son kaç tahminin penceresi üzerinde yumuşatma yapılacağı
MIN_VOTES = 6        # baskın sınıfın bu pencerede en az kaç kez tekrar etmesi gerektiği


@st.cache_resource
def model_ve_etiketleri_yukle(n_classes):
    import tensorflow as tf
    etiket = f"{n_classes}class"
    model = tf.keras.models.load_model(
        os.path.join(MODELS_DIR, f"best_model_{etiket}.keras"))
    with open(os.path.join(MODELS_DIR, f"model_meta_{etiket}.json"),
              encoding="utf-8") as f:
        meta = json.load(f)
    secili = meta["secili_siniflar"]
    test_acc = meta.get("test_accuracy", 0.0)
    isim_map = {}
    sl = os.path.join(SPLITS_DIR, "signlist_TR.csv")
    if os.path.exists(sl):
        import pandas as pd
        df = pd.read_csv(sl)
        isim_map = dict(zip(df["class_id"], df["isim_tr"]))
    etiketler = [str(isim_map.get(cid, f"Sinif_{cid}")) for cid in secili]
    return model, etiketler, test_acc, secili


# ----------------------------------------------------------- Kenar çubuğu: ayarlar
n_classes = 5   # Proje 5 işaret üzerine sabit
with st.sidebar:
    st.header("⚙️ Ayarlar")
    esik = st.slider(
        "Güven eşiği", 0.0, 1.0, 0.70, 0.05,
        help="Tahminin gösterilmesi için gereken en düşük güven oranı. Eşiğin "
             "altındaki belirsiz hareketlerde tahmin gösterilmez.")
    _GUVEN_ESIGI[0] = esik

# ----------------------------------------------------------- Başlık (sade)
st.title("🤟 Gerçek Zamanlı Türk İşaret Dili Tanıma")
st.caption("İstanbul Topkapı Üniversitesi · Yazılım Mühendisliği Mezuniyet Projesi")

# Modeli yükle
try:
    model, etiketler, test_acc, secili = model_ve_etiketleri_yukle(n_classes)
except Exception as e:
    st.error(f"Model yüklenemedi: {e}")
    st.info(f"Önce modeli eğitin:  python src/egit_model.py --n_classes {n_classes}")
    st.stop()

# ----------------------------------------------------------- Kenar çubuğu: bilgi
with st.sidebar:
    st.metric("Test doğruluğu", f"%{test_acc*100:.1f}")
    st.markdown("**Tanıyabildiği işaretler**")
    st.markdown("  ·  ".join(etiketler))
    with st.expander("❓ Yardım", expanded=False):
        st.markdown(
            "**Bu uygulama ne yapar?**\n"
            "Web kameranızdan görüntü alıp yaptığınız Türk İşaret Dili işaretini "
            "gerçek zamanlı olarak tanır ve ekranda gösterir.\n\n"
            "**Nasıl çalışır?**\n"
            "Her kareden MediaPipe Holistic ile el ve üst gövde anahtar noktaları "
            "çıkarılır; 30 karelik dizi LSTM tabanlı bir model ile sınıflandırılır.\n\n"
            "**Nasıl kullanılır?**\n"
            "1. **Başlat**'a basın.\n"
            "2. Kamera izni isteyince **İzin Ver**.\n"
            "3. Birden fazla kamera varsa **Cihaz seç** ile kullanacağınız kamerayı seçin.\n"
            "4. Elleriniz ve üst gövdeniz kameraya tam görünür olsun.\n"
            "5. İşareti yapıp kısa süre bekleyin; tahmin **soldaki panelde** çıkar.\n"
            "6. Hareket net değilse panel boş kalır.\n\n"
            "💡 İyi ışık ve sade arka plan doğruluğu artırır.")
    st.divider()
    st.caption("MediaPipe Holistic (iskelet çıkarımı) + LSTM (zamansal "
               "sınıflandırma) ile gerçek zamanlı tanıma.")

# İşaret referans GIF'leri (data/gifs/{id}.gif) — nasıl yapıldığını gösterir
with st.expander("📖 İşaretler nasıl yapılır? (göster / gizle)", expanded=False):
    if any(os.path.exists(os.path.join(GIFS_DIR, f"{c}.gif")) for c in secili):
        gcols = st.columns(len(secili))
        for col, cid, ad in zip(gcols, secili, etiketler):
            with col:
                gp = os.path.join(GIFS_DIR, f"{cid}.gif")
                if os.path.exists(gp):
                    st.image(gp, width="stretch")
                st.markdown(f"<div style='text-align:center'><b>{ad}</b></div>",
                            unsafe_allow_html=True)
    else:
        st.info("GIF'ler henüz oluşturulmadı. Şu komutu çalıştırın:  "
                "python src/gif_olustur.py --n_classes 5")

# ----------------------------------------------------------- WebRTC
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import mediapipe as mp
import cv2

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils


class IsaretProcessor(VideoProcessorBase):
    def __init__(self):
        self.holistic = mp_holistic.Holistic(
            model_complexity=1, min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.buffer = collections.deque(maxlen=SEQUENCE_LENGTH)
        self.hand_hist = collections.deque(maxlen=SEQUENCE_LENGTH)
        self.lock = threading.Lock()
        self.ana = ""
        self.top3 = []
        self.buffer_len = 0
        self.gecmis = collections.deque(maxlen=10)
        self.prob_hist = collections.deque(maxlen=SMOOTH_WINDOW)  # yumuşatma için son tahminler

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.holistic.process(rgb)
        rgb.flags.writeable = True

        if results.left_hand_landmarks:
            mp_draw.draw_landmarks(img, results.left_hand_landmarks,
                                   mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_draw.draw_landmarks(img, results.right_hand_landmarks,
                                   mp_holistic.HAND_CONNECTIONS)
        if results.pose_landmarks:
            mp_draw.draw_landmarks(img, results.pose_landmarks,
                                   mp_holistic.POSE_CONNECTIONS)

        self.buffer.append(extract_keypoints(results))
        hand_present = bool(results.left_hand_landmarks or results.right_hand_landmarks)
        self.hand_hist.append(hand_present)

        # --- Anlık tahmin: tampon dolu VE eller yeterince görünüyorsa olasılıkları üret
        yeterli_el = sum(self.hand_hist) >= 12   # 30 karenin en az ~%40'ında el
        if len(self.buffer) == SEQUENCE_LENGTH and yeterli_el:
            seq = np.expand_dims(np.array(self.buffer), axis=0)
            probs = model.predict(seq, verbose=0)[0]
            self.prob_hist.append(probs)
        else:
            # El yoksa / tampon dolmadıysa geçmişi temizle ki eski tahmin takılı kalmasın
            self.prob_hist.clear()

        # --- Zamansal yumuşatma: tahmin ancak son karelerde KARARLI hale gelince gösterilir
        ana, top3 = "", []
        if len(self.prob_hist) >= MIN_VOTES:
            arr = np.array(self.prob_hist)            # (pencere, sinif_sayisi)
            ort_probs = arr.mean(axis=0)              # pencere boyunca ortalama olasılık
            oylar = arr.argmax(axis=1)                # her karenin tahmini
            siniflar, sayilar = np.unique(oylar, return_counts=True)
            baskin = int(siniflar[sayilar.argmax()])  # en sık tahmin edilen sınıf
            baskin_oy = int(sayilar.max())
            idx3 = ort_probs.argsort()[-3:][::-1]
            top3 = [(etiketler[i], float(ort_probs[i])) for i in idx3]
            # Baskın sınıf yeterince tekrar etmiş VE ortalama güveni eşiği geçiyorsa kabul et
            if baskin_oy >= MIN_VOTES and ort_probs[baskin] >= _GUVEN_ESIGI[0]:
                ana = etiketler[baskin]

        with self.lock:
            self.ana = ana
            self.top3 = top3 if ana else []
            self.buffer_len = len(self.buffer)
            if ana and (not self.gecmis or self.gecmis[-1] != ana):
                self.gecmis.append(ana)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# Ana satır: solda tahmin paneli, sağda kamera — sayfanın üst kısmında
col_panel, col_video = st.columns([1, 1.25], gap="large")

with col_panel:
    st.markdown("##### 🔎 Tahmin")
    ph_ana = st.empty()
    ph_top3 = st.empty()
    st.markdown("##### 🕘 Son tanınanlar")
    ph_gecmis = st.empty()
    ph_buf = st.empty()

with col_video:
    st.markdown("##### 📷 Kamera")
    _webrtc_args = dict(
        key="tid",
        video_processor_factory=IsaretProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )
    # Kamera bileşeninin düğmelerini Türkçeleştir (Başlat/Durdur/Cihaz seç).
    _tr = {
        "start": "Başlat",
        "stop": "Durdur",
        "select_device": "Cihaz seç",
        "media_api_not_available": "Medya API'si kullanılamıyor",
        "device_ask_permission": "Kamera izni bekleniyor…",
        "device_not_available": "Kamera bulunamadı",
        "device_access_denied": "Kamera erişimi reddedildi",
    }
    try:
        ctx = webrtc_streamer(**_webrtc_args, translations=_tr)
    except TypeError:
        # Eski streamlit-webrtc sürümü 'translations' desteklemiyorsa düğmeler İngilizce kalır
        ctx = webrtc_streamer(**_webrtc_args)

ph_ana.markdown("<p style='color:#888'>Kamerayı başlatmak için Başlat'a basın.</p>",
                unsafe_allow_html=True)
ph_gecmis.markdown("—")

if ctx.state.playing:
    while ctx.state.playing:
        vp = ctx.video_processor
        if vp is None:
            break
        with vp.lock:
            ana = vp.ana
            top3 = list(vp.top3)
            blen = vp.buffer_len
            gecmis = list(vp.gecmis)
        if ana:
            ph_ana.markdown(f"<h1 style='color:#16a34a;margin:0'>{ana}</h1>",
                            unsafe_allow_html=True)
            with ph_top3.container():
                for ad, p in top3:
                    st.progress(min(max(p, 0.0), 1.0), text=f"{ad} — %{p*100:.0f}")
        else:
            ph_ana.markdown(
                "<p style='color:#888;font-size:1.2rem'>İşaret bekleniyor…</p>",
                unsafe_allow_html=True)
            ph_top3.empty()
        ph_gecmis.markdown("  →  ".join(gecmis) if gecmis else "—")
        ph_buf.caption(f"Tampon: {blen}/{SEQUENCE_LENGTH} kare")
        time.sleep(0.2)
