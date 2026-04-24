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
# FUNÇÃO: Busca arquivos que começam com o prefixo (ex: "11")
# =========================
def buscar_arquivo(diretorio, prefixo):
    arquivos = os.listdir(diretorio)
    for f in arquivos:
        if f.startswith(prefixo) and f.endswith(".txt"):
            return os.path.join(diretorio, f)
    return None

# =========================
# LEITURA DOS ARQUIVOS
# =========================
def ler_txt_vallen(caminho_arquivo):
    if not caminho_arquivo or not os.path.exists(caminho_arquivo):
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

def calcular_amortecimento(dados):
    picos, _ = signal.find_peaks(np.abs(dados), height=np.max(np.abs(dados))*0.2)
    if len(picos) < 2: return 0.0
    p1, p2 = np.abs(dados[picos[:2]])
    log_dec = np.log(p1/p2)
    return log_dec / np.sqrt((2*np.pi)**2 + log_dec**2)

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

def aplicar_filtro(x, fs, fmin, fmax):
    if fs is None: return x
    nyq = fs/2
    if fmax >= nyq: return x
    sos = signal.butter(2, [fmin, fmax], fs=fs, btype='bandpass', output='sos')
    return signal.sosfilt(sos, x)

def verificar_par(x, y, fs):
    min_len = min(len(x), len(y))
    x, y = x[:min_len], y[:min_len]
    corr = np.corrcoef(x, y)[0,1]
    cross = signal.correlate(x, y, mode='full')
    lag = np.argmax(cross) - len(x)
    atraso_us = lag / fs * 1e6 if fs else 0
    return corr, atraso_us

# =========================
# INTERFACE
# =========================
if not os.path.exists(PASTA_RAIZ):
    st.error(f"Pasta '{PASTA_RAIZ}' não encontrada.")
else:
    subpastas = sorted([f for f in os.listdir(PASTA_RAIZ) if os.path.isdir(os.path.join(PASTA_RAIZ,f))])

    st.sidebar.subheader("🔧 Filtro")
    usar_filtro = st.sidebar.checkbox("Ativar filtro", value=False)
    fmin = st.sidebar.number_input("Freq Min (Hz)", value=300000)
    fmax = st.sidebar.number_input("Freq Max (Hz)", value=600000)

    exp = st.sidebar.selectbox("📁 Experimento", subpastas)
    caminho_dir = os.path.join(PASTA_RAIZ, exp)
    
    # Extrai prefixos únicos de arquivos que terminam em 1.txt ou 2.txt
    arquivos_no_dir = os.listdir(caminho_dir)
    prefixos = sorted(list(set([f[0] for f in arquivos_no_dir if f.endswith(".txt") and f[0].isdigit()])))
    
    par_id = st.sidebar.selectbox("🔢 Selecione o Par", prefixos, format_func=lambda x: f"Par {x}1 e {x}2")

    if st.sidebar.button("📉 Gerar Análise"):
        # Busca automática pelos arquivos reais
        path_x = buscar_arquivo(caminho_dir, f"{par_id}1")
        path_y = buscar_arquivo(caminho_dir, f"{par_id}2")

        fs, x = ler_txt_vallen(path_x)
        _, y = ler_txt_vallen(path_y)

        if x is None or y is None:
            st.error(f"Erro ao ler arquivos do par {par_id}. Verifique se {par_id}1 e {par_id}2 existem.")
        else:
            x -= np.mean(x)
            y -= np.mean(y)

            if usar_filtro:
                x = aplicar_filtro(x, fs, fmin, fmax)
                y = aplicar_filtro(y, fs, fmin, fmax)

            # Verificação
            corr, atraso = verificar_par(x,y,fs)
            st.subheader("✅ Verificação do Par")
            col1, col2 = st.columns(2)
            col1.metric("Correlação", f"{corr:.3f}")
            col2.metric("Atraso (µs)", f"{atraso:.2f}")

            if corr < 0.3: st.warning("⚠️ Baixa correlação detectada.")
            else: st.success("Par consistente ✅")

            # Tabela
            st.subheader("📋 Tabela 1 — Parâmetros")
            df = pd.DataFrame([
                calcular_tabela(x,fs,"Canal 1"),
                calcular_tabela(y,fs,"Canal 2")
            ])
            st.dataframe(df, use_container_width=True)

            # Gráficos
            t = np.arange(len(x))/fs*1e6
            fig, axs = plt.subplots(2,2,figsize=(14,8))
            axs[0,0].plot(t,x); axs[0,0].set_title("Canal 1 (µs)"); axs[0,0].grid(True)
            axs[0,1].plot(t,y,color="green"); axs[0,1].set_title("Canal 2 (µs)"); axs[0,1].grid(True)
            
            Xf = np.abs(np.fft.rfft(x))
            Yf = np.abs(np.fft.rfft(y))
            freqs = np.fft.rfftfreq(len(x),1/fs)/1000
            axs[1,0].plot(freqs,Xf,label="C1"); axs[1,0].plot(freqs,Yf,label="C2")
            axs[1,0].set_title("FFT (kHz)"); axs[1,0].legend(); axs[1,0].grid(True)
            
            axs[1,1].plot(x,y,color="purple", alpha=0.6); axs[1,1].set_title("Lissajous"); axs[1,1].grid(True)
            
            plt.tight_layout()
            st.pyplot(fig)