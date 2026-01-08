import streamlit as st
import cv2
import numpy as np
import matplotlib
# --- ВАЖНЫЙ ФИКС: Отключаем интерактивный режим Matplotlib ---
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from PIL import Image
import os


# --- 1. CONFIG & STYLE ---
st.set_page_config(layout="wide", page_title="DeepFake Hunter", page_icon="🛡️")

# --- 1. CONFIG & STYLE ---
st.set_page_config(layout="wide", page_title="DeepFake Hunter: Pro", page_icon="🛡️")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    h1, h2, h3 { color: #FF4B4B !important; font-family: 'Helvetica', sans-serif; }
    .stAlert { background-color: #262730; border: 1px solid #FF4B4B; color: #FAFAFA; }
    div[data-testid="stMetricValue"] { font-size: 2rem; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ DEEPFAKE HUNTER // ANALYZER")
st.markdown("Загрузи изображение для проверки на спектральные аномалии (FFT + Walsh).")

# --- 2. MATH CORE ---

@st.cache_data
def generate_walsh_matrix(n_dim):
    # Генерация матрицы Уолша (Sequency Order)
    def hadamard_recursive(n):
        if n == 1: return np.array([[1]])
        h_n = hadamard_recursive(n // 2)
        return np.vstack((np.hstack((h_n, h_n)), np.hstack((h_n, -h_n))))

    H = hadamard_recursive(n_dim)
    k = int(np.log2(n_dim))
    
    def gray_code(m):
        if m == 1: return ['0', '1']
        g_prev = gray_code(m - 1)
        return ['0' + s for s in g_prev] + ['1' + s for s in g_prev[::-1]]

    g = gray_code(k)
    revers = [int(s[::-1], 2) for s in g]
    return H[:, np.array(revers)]

class ArtifactHunter:
    @staticmethod
    def get_log_profile(image):
        # 1D профиль частот
        img_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        f = np.fft.fft2(img_gray)
        fshift = np.fft.fftshift(f)
        
        rows, cols = img_gray.shape
        crow, ccol = rows // 2, cols // 2
        y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
        r = np.hypot(x, y).astype(int)
        
        psd = np.abs(fshift) ** 2
        tbin = np.bincount(r.ravel(), weights=psd.ravel(), minlength=r.max()+1)
        nr = np.bincount(r.ravel(), minlength=r.max()+1)
        nr[nr == 0] = 1
        return np.log(tbin / nr + 1e-9)[1:]

    @staticmethod
    def analyze_walsh_sequency(image):
        target_size = 512
        img_gray = np.array(image.convert('L').resize((target_size, target_size))).astype(np.float64)
        W = generate_walsh_matrix(target_size)
        wht = np.dot(np.dot(W, img_gray), W) / target_size
        wht_log = np.log(np.abs(wht) + 1)
        wht_norm = cv2.normalize(wht_log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Энергия высоких частот (правый нижний угол)
        h, w = wht_norm.shape
        roi = wht_norm[int(h*0.75):, int(w*0.75):] 
        return wht_norm, np.mean(roi)

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Настройки детектора")
    st.info("Измените пороги, если много ложных срабатываний.")
    drop_threshold = st.slider("Порог FFT (Резкость падения)", 0.05, 0.4, 0.15)
    energy_threshold = st.slider("Порог Walsh (Текстура)", 5.0, 50.0, 18.0)

# --- 4. MAIN ---
uploaded_file = st.file_uploader("Перетащите изображение сюда", type=["jpg", "png", "jpeg", "webp"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        
        # --- ANALYSIS ---
        with st.spinner('Сканирование спектра...'):
            # FFT Analysis
            log_profile = ArtifactHunter.get_log_profile(image)
            # Logic for FFT Drop
            skip_start = int(len(log_profile) * 0.2)
            skip_end = int(len(log_profile) * 0.95)
            roi_profile = log_profile[skip_start:skip_end]
            diffs = np.convolve(np.diff(roi_profile), np.ones(3)/3, mode='valid')
            
            min_drop = np.min(diffs) if len(diffs) > 0 else 0
            fft_fake_prob = 1 if min_drop < -drop_threshold else 0
            
            # Walsh Analysis
            walsh_img, walsh_energy = ArtifactHunter.analyze_walsh_sequency(image)
            walsh_fake_prob = 1 if walsh_energy < energy_threshold else 0

        # --- VERDICT SYSTEM ---
        # Простая система очков: 0 - чисто, 1 - подозрительно, 2 - точно фейк
        risk_score = fft_fake_prob + walsh_fake_prob
        
        col_res1, col_res2, col_res3 = st.columns([1, 2, 1])
        with col_res2:
            if risk_score == 2:
                st.error(f"🚨 ВЫСОКАЯ ВЕРОЯТНОСТЬ AI (Score: {risk_score}/2)")
            elif risk_score == 1:
                st.warning(f"⚠️ ПОДОЗРИТЕЛЬНО (Score: {risk_score}/2)")
            else:
                st.success(f"✅ ВЫГЛЯДИТ ЧИСТЫМ (Score: {risk_score}/2)")

        st.divider()

        # --- VISUALIZATION ---
        t1, t2 = st.tabs(["📉 Спектральный анализ (FFT)", "🏁 Текстура (Walsh)"])
        
        with t1:
            col1, col2 = st.columns([3, 1])
            with col1:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(log_profile, color='#00ff41' if fft_fake_prob == 0 else '#ff4b4b')
                ax.set_facecolor('black')
                fig.patch.set_facecolor('#0E1117')
                ax.grid(color='#333', linestyle='--')
                st.pyplot(fig)
            with col2:
                st.metric("Макс. падение", f"{min_drop:.3f}")
                if fft_fake_prob:
                    st.caption("❌ Резкий обрыв частот (характерно для GAN/Upscale)")
                else:
                    st.caption("✅ Естественное затухание")

        with t2:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.image(walsh_img, use_container_width=True, caption="Спектр Уолша")
            with col2:
                st.metric("Энергия ВЧ", f"{walsh_energy:.1f}")
                if walsh_fake_prob:
                    st.caption("❌ Мало деталей в ВЧ (Размытие/AI)")
                else:
                    st.caption("✅ Хорошая микро-текстура")

    except Exception as e:
        st.error(f"Ошибка чтения файла: {e}")

else:
    # Заглушка, когда файл не выбран
    st.info("👆 Загрузите изображение, чтобы начать анализ.")
