import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd

st.set_page_config(page_title="Ensaio de Emissão Acústica", layout="wide")
st.title("📡 Ensaio de Emissão Acústica")

PASTA_RAIZ = "experimentos"

# =========================
# LEITURA DOS ARQUIVOS
# =========================
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

# =========================
# AMORTECIMENTO
# =========================
def calcular_amortecimento(dados):
    picos, _ = signal.find_peaks(np.abs(dados))
    if len(picos) < 2:
        return 0.0
    p1, p2 = np.abs(dados[picos[:2]])
    if p1 == 0 or p2 == 0:
        return 0.0
    log_dec = np.log(p1/p2)
    return log_dec / np.sqrt((2*np.pi)**2 + log_dec**2)

# =========================
# TABELA 1
# =========================
def calcular_tabela(sinal, fs, nome):
    rms = np.sqrt(np.mean(sinal**2))
    pp = np.max(sinal) - np.min(sinal)
    energia = np.sum(sinal**2)
    X_f = np.abs(np.fft.rfft(sinal))
    freq = np.fft.rfftfreq(len(sinal), d=1/fs)
    freq_dom = freq[np.argmax(X_f[1:])+1]/1000
    return {
        "Canal": nome,
        "RMS": round(rms,5),
        "Pico a Pico": round(pp,5),
        "Energia": f"{energia:.2e}",
        "Freq (kHz)": round(freq_dom,1),
        "ζ": round(calcular_amortecimento(sinal),5)
    }

# =========================
# FILTRO (suave)
# =========================
def aplicar_filtro(x, fs, fmin, fmax):
    if fs is None: return x
    nyq = fs/2
    if fmax >= nyq: return x
    sos = signal.butter(2, [fmin, fmax], fs=fs, btype='bandpass', output='sos')
    return signal.sosfilt(sos, x)

# =========================
# VERIFICAÇÃO DOS PARES
# =========================
def verificar_par(x, y, fs):
    min_len = min(len(x), len(y))
    x = x[:min_len]
    y = y[:min_len]

    # Correlação
    corr = np.corrcoef(x, y)[0,1]

    # atraso (lag)
    cross = signal.correlate(x, y, mode='full')
    lag = np.argmax(cross) - len(x)

    atraso_us = lag / fs * 1e6 if fs else 0

    return corr, atraso_us

# =========================
# INTERFACE
# =========================
if not os.path.exists(PASTA_RAIZ):
    st.error("Pasta não encontrada")
else:
    subpastas = sorted([f for f in os.listdir(PASTA_RAIZ) if os.path.isdir(os.path.join(PASTA_RAIZ,f))])

    st.sidebar.subheader("Filtro")
    usar_filtro = st.sidebar.checkbox("Ativar filtro", value=False)
    fmin = st.sidebar.number_input("Freq Min", value=300000)
    fmax = st.sidebar.number_input("Freq Max", value=600000)

    exp = st.sidebar.selectbox("Experimento", subpastas)
    caminho = os.path.join(PASTA_RAIZ, exp)
    arquivos = os.listdir(caminho)

    pares = sorted(list(set([f[0] for f in arquivos if f.endswith(".txt")])))
    par = st.sidebar.selectbox("Par", pares)

    if st.sidebar.button("Gerar"):

        fs, x = ler_txt_vallen(os.path.join(caminho,f"{par}1.txt"))
        _, y = ler_txt_vallen(os.path.join(caminho,f"{par}2.txt"))

        if x is None:
            st.error("Erro leitura")
        else:
            x -= np.mean(x)
            y -= np.mean(y)

            if usar_filtro:
                x = aplicar_filtro(x, fs, fmin, fmax)
                y = aplicar_filtro(y, fs, fmin, fmax)

            # =========================
            # VERIFICAÇÃO AUTOMÁTICA
            # =========================
            corr, atraso = verificar_par(x,y,fs)

            st.subheader("✅ Verificação do Par")
            c1,c2 = st.columns(2)
            c1.metric("Correlação", f"{corr:.3f}")
            c2.metric("Atraso (µs)", f"{atraso:.2f}")

            if corr < 0.3:
                st.warning("⚠️ Baixa correlação: possível erro de pareamento")
            else:
                st.success("Par consistente ✅")

            # =========================
            # TABELA
            # =========================
            st.subheader("Tabela 1")
            df = pd.DataFrame([
                calcular_tabela(x,fs,"Canal 1"),
                calcular_tabela(y,fs,"Canal 2")
            ])
            st.dataframe(df, use_container_width=True)

            # =========================
            # GRÁFICOS
            # =========================
            t = np.arange(len(x))/fs*1e6

            fig,axs = plt.subplots(2,2,figsize=(14,8))

            axs[0,0].plot(t,x)
            axs[0,0].set_title("Canal 1")
            axs[0,0].set_xlabel("µs")

            axs[0,1].plot(t,y,color="green")
            axs[0,1].set_title("Canal 2")
            axs[0,1].set_xlabel("µs")

            Xf = np.abs(np.fft.rfft(x))
            Yf = np.abs(np.fft.rfft(y))
            f = np.fft.rfftfreq(len(x),1/fs)/1000

            axs[1,0].plot(f,Xf,label="C1")
            axs[1,0].plot(f,Yf,label="C2")
            axs[1,0].set_title("FFT (kHz)")
            axs[1,0].legend()

            axs[1,1].plot(x,y,color="purple")
            axs[1,1].set_title("Lissajous")

            plt.tight_layout()
            st.pyplot(fig)