import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

st.set_page_config(page_title="Analisador de Experimentos", layout="wide")
st.title("📊 Analisador de Experimentos Online v2.1")

PASTA_RAIZ = "experimentos"

def ler_txt_vallen(caminho_arquivo):
    if not os.path.exists(caminho_arquivo):
        return None, None
    sample_rate = None
    dados = []
    lendo = False
    with open(caminho_arquivo, "r", encoding="utf-8", errors="ignore") as f:
        for linha in f:
            linha = linha.strip()
            if linha.startswith("SampleRate[Hz]:"):
                try:
                    sample_rate = float(linha.split(":")[1].strip())
                except: pass
            elif linha == "[DATA]":
                lendo = True
                continue
            elif linha == "[ENDDATA]":
                break
            elif lendo and linha:
                try:
                    dados.append(float(linha))
                except: pass
    return sample_rate, np.array(dados)

def calcular_amortecimento(dados, fs):
    picos, _ = signal.find_peaks(dados)
    if len(picos) < 2: return 0.0
    val_picos = dados[picos][:2]
    if val_picos[0] <= 0 or val_picos[1] <= 0: return 0.0
    log_dec = np.log(val_picos[0] / val_picos[1])
    zeta = log_dec / np.sqrt((2 * np.pi)**2 + log_dec**2)
    return zeta

# --- MENU LATERAL ---
if not os.path.exists(PASTA_RAIZ):
    st.error(f"Pasta '{PASTA_RAIZ}' não encontrada.")
else:
    subpastas = sorted([f for f in os.listdir(PASTA_RAIZ) if os.path.isdir(os.path.join(PASTA_RAIZ, f))])
    
    modo = st.sidebar.radio("Selecione o Modo:", ["🔬 Visão Detalhada", "🆚 Comparação"])
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Filtro Digital")
    usar_filtro = st.sidebar.checkbox("Ativar filtro passa-baixa", value=True)
    freq_corte = st.sidebar.slider("Frequência de corte (Hz)", 100, 10000, 2000) if usar_filtro else None

    if modo == "🔬 Visão Detalhada":
        experimento_sel = st.sidebar.selectbox("Escolha o Experimento:", subpastas)
        caminho_exp = os.path.join(PASTA_RAIZ, experimento_sel)
        arquivos = os.listdir(caminho_exp)
        prefixos = sorted(list(set([f[0] for f in arquivos if f.endswith(".txt") and len(f) >= 2 and f[0].isdigit()])))
        escolha_par = st.sidebar.radio("Escolha o par:", prefixos, format_func=lambda x: f"Par {x}1 e {x}2")

        if st.sidebar.button("📉 Gerar Análise"):
            fs, x = ler_txt_vallen(os.path.join(caminho_exp, f"{escolha_par}1.txt"))
            _, y = ler_txt_vallen(os.path.join(caminho_exp, f"{escolha_par}2.txt"))

            if x is not None and y is not None:
                x, y = x - np.mean(x), y - np.mean(y)
                if usar_filtro:
                    sos = signal.butter(4, freq_corte, fs=fs, btype='low', output='sos')
                    x, y = signal.sosfilt(sos, x), signal.sosfilt(sos, y)
                
                st.subheader(f"📊 {experimento_sel}")
                fig, axs = plt.subplots(2, 2, figsize=(14, 8))
                axs[0,0].plot(x, color="blue"); axs[0,0].set_title("Canal 1 - Tempo")
                axs[0,1].plot(y, color="green"); axs[0,1].set_title("Canal 2 - Tempo")
                
                X_f = np.abs(np.fft.rfft(x)); Y_f = np.abs(np.fft.rfft(y))
                freq = np.fft.rfftfreq(len(x), d=1/fs)
                axs[1,0].plot(freq, X_f, label="C1"); axs[1,0].plot(freq, Y_f, label="C2")
                axs[1,0].set_title("FFT"); axs[1,0].set_xlim(0, fs/4); axs[1,0].legend()
                
                axs[1,1].plot(x[:len(y)], y[:len(x)], color="purple", alpha=0.6)
                axs[1,1].set_title("Lissajous (C1 vs C2) - Gancho")
                
                for ax in axs.flat: ax.grid(True)
                plt.tight_layout()
                st.pyplot(fig)

    else:  # MODO COMPARAÇÃO
        exp_a = st.sidebar.selectbox("Material A:", subpastas, index=0)
        exp_b = st.sidebar.selectbox("Material B:", subpastas, index=min(1, len(subpastas)-1))
        
        if st.sidebar.button("⚡ Comparar"):
            def pegar_dados(subpasta):
                caminho = os.path.join(PASTA_RAIZ, subpasta)
                txts = sorted([f for f in os.listdir(caminho) if f.endswith(".txt") and f[0].isdigit()])
                if not txts: return None, None, None
                fs, x = ler_txt_vallen(os.path.join(caminho, txts[0]))
                _, y = ler_txt_vallen(os.path.join(caminho, txts[1])) if len(txts)>1 else (fs, x)
                return fs, x, y

            fs_a, x_a, y_a = pegar_dados(exp_a)
            fs_b, x_b, y_b = pegar_dados(exp_b)

            if x_a is not None and x_b is not None:
                x_a, x_b = x_a - np.mean(x_a), x_b - np.mean(x_b)
                if usar_filtro:
                    sos_a = signal.butter(4, freq_corte, fs=fs_a, btype='low', output='sos')
                    x_a = signal.sosfilt(sos_a, x_a)
                    sos_b = signal.butter(4, freq_corte, fs=fs_b, btype='low', output='sos')
                    x_b = signal.sosfilt(sos_b, x_b)

                st.subheader(f"⚔️ Comparação: {exp_a} vs {exp_b}")
                fig, axs = plt.subplots(2, 2, figsize=(14, 8))
                
                axs[0,0].plot(x_a, label=exp_a); axs[0,0].plot(x_b, label=exp_b)
                axs[0,0].set_title("Sinais no Tempo"); axs[0,0].legend()
                
                axs[0,1].plot(x_a[:len(y_a)], y_a[:len(x_a)], label=f"Gancho {exp_a}")
                axs[0,1].plot(x_b[:len(y_b)], y_b[:len(x_b)], label=f"Gancho {exp_b}")
                axs[0,1].set_title("Lissajous Comparativo"); axs[0,1].legend()

                max_len = max(len(x_a), len(x_b))
                X_a_f = np.abs(np.fft.rfft(x_a, n=max_len)); X_b_f = np.abs(np.fft.rfft(x_b, n=max_len))
                freq = np.fft.rfftfreq(max_len, d=1/fs_a)
                axs[1,0].plot(freq, X_a_f, label=exp_a); axs[1,0].plot(freq, X_b_f, label=exp_b)
                axs[1,0].set_title("FFT Comparativa"); axs[1,0].set_xlim(0, fs_a/4); axs[1,0].legend()

                st.pyplot(fig)