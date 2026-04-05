import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

st.set_page_config(page_title="Analisador de Experimentos", layout="wide")
st.title("📊 Analisador de Experimentos Online v2.0")

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
                except:
                    pass
            elif linha == "[DATA]":
                lendo = True
                continue
            elif linha == "[ENDDATA]":
                break
            elif lendo and linha:
                try:
                    dados.append(float(linha))
                except:
                    pass
    return sample_rate, np.array(dados)

def calcular_amortecimento(dados, fs):
    # Encontra picos sucessivos para calcular decréscimo logarítmico
    picos, _ = signal.find_peaks(dados)
    if len(picos) < 2:
        return 0.0
    
    val_picos = dados[picos][:2]  # Pega os dois primeiros picos
    if val_picos[0] <= 0 or val_picos[1] <= 0:
        return 0.0
        
    log_dec = np.log(val_picos[0] / val_picos[1])
    zeta = log_dec / np.sqrt((2 * np.pi)**2 + log_dec**2)
    return zeta

# --- MENU LATERAL ---
if not os.path.exists(PASTA_RAIZ):
    st.error(f"Pasta '{PASTA_RAIZ}' não encontrada.")
else:
    subpastas = sorted([f for f in os.listdir(PASTA_RAIZ) if os.path.isdir(os.path.join(PASTA_RAIZ, f))])
    
    modo = st.sidebar.radio("Selecione o Modo:", ["🔬 Visão Detalhada", "🆚 Comparação"])

    if modo == "🔬 Visão Detalhada":
        experimento_sel = st.sidebar.selectbox("Escolha o Experimento:", subpastas)
        caminho_exp = os.path.join(PASTA_RAIZ, experimento_sel)
        arquivos = os.listdir(caminho_exp)
        prefixos = sorted(list(set([f[0] for f in arquivos if f.endswith(".txt") and len(f) >= 2 and f[0].isdigit()])))
        escolha_par = st.sidebar.radio("Escolha o par:", prefixos, format_func=lambda x: f"Par {x}1 e {x}2")

        if st.sidebar.button("📉 Gerar Análise Completa"):
            fs, x = ler_txt_vallen(os.path.join(caminho_exp, f"{escolha_par}1.txt"))
            _, y = ler_txt_vallen(os.path.join(caminho_exp, f"{escolha_par}2.txt"))

            if x is not None and y is not None:
                x = x - np.mean(x)
                y = y - np.mean(y)
                
                zeta_x = calcular_amortecimento(x, fs)
                zeta_y = calcular_amortecimento(y, fs)

                st.subheader(f"📊 {experimento_sel} - Par {escolha_par}")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("🔵 Amortecimento C1 (ζ)", f"{zeta_x:.4f}")
                col2.metric("🟢 Amortecimento C2 (ζ)", f"{zeta_y:.4f}")
                col3.metric("🔵 RMS Canal 1", f"{np.sqrt(np.mean(x**2)):.4f}")
                col4.metric("🟢 RMS Canal 2", f"{np.sqrt(np.mean(y**2)):.4f}")

                fig, axs = plt.subplots(2, 2, figsize=(14, 8))
                axs[0,0].plot(x, color="blue"); axs[0,0].set_title("Canal 1 - Tempo")
                axs[0,1].plot(y, color="green"); axs[0,1].set_title("Canal 2 - Tempo")
                
                X_f = np.abs(np.fft.rfft(x))
                Y_f = np.abs(np.fft.rfft(y))
                freq = np.fft.rfftfreq(len(x), d=1/fs)
                axs[1,0].plot(freq, X_f, color="blue"); axs[1,0].plot(freq, Y_f, color="green")
                axs[1,0].set_title("FFT Sobreposta"); axs[1,0].set_xlim(0, fs/4)
                
                axs[1,1].plot(x[:len(y)], y[:len(x)], color="purple", alpha=0.6)
                axs[1,1].set_title("Lissajous (C1 vs C2)")
                
                for ax in axs.flat: ax.grid(True)
                plt.tight_layout()
                st.pyplot(fig)

    else:  # MODO COMPARAÇÃO
        st.sidebar.subheader("Experimento A")
        exp_a = st.sidebar.selectbox("Material A:", subpastas)
        st.sidebar.subheader("Experimento B")
        exp_b = st.sidebar.selectbox("Material B:", subpastas)
        
        if st.sidebar.button("⚡ Comparar"):
            fs_a, x_a = ler_txt_vallen(os.path.join(PASTA_RAIZ, exp_a, "11.txt"))
            fs_b, x_b = ler_txt_vallen(os.path.join(PASTA_RAIZ, exp_b, "11.txt"))

            if x_a is not None and x_b is not None:
                x_a = x_a - np.mean(x_a)
                x_b = x_b - np.mean(x_b)

                st.subheader(f"⚔️ Comparando: {exp_a} vs {exp_b} (Canais 1)")

                fig, axs = plt.subplots(2, 1, figsize=(12, 10))
                
                # Gráfico Tempo
                axs[0].plot(x_a, label=exp_a, alpha=0.7); axs[0].plot(x_b, label=exp_b, alpha=0.7)
                axs[0].set_title("Sinais no Tempo - Sobrepostos"); axs[0].legend(); axs[0].grid(True)

                # Gráfico FFT
                max_len = max(len(x_a), len(x_b))
                X_a_f = np.abs(np.fft.rfft(x_a, n=max_len))
                X_b_f = np.abs(np.fft.rfft(x_b, n=max_len))
                freq = np.fft.rfftfreq(max_len, d=1/fs_a)
                
                axs[1].plot(freq, X_a_f, label=f"FFT {exp_a}"); axs[1].plot(freq, X_b_f, label=f"FFT {exp_b}")
                axs[1].set_title("Espectro de Frequência Comparativo"); axs[1].legend(); axs[1].grid(True)
                axs[1].set_xlim(0, fs_a/4)

                plt.tight_layout()
                st.pyplot(fig)