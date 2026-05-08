import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
import re

st.set_page_config(page_title="Ensaio de Emissão Acústica", layout="wide")

# =========================
# LOGIN
# =========================
USUARIOS = {
    "Cermat1": "acustica2026"
}

if "logado" not in st.session_state:
    st.session_state.logado = False

if not st.session_state.logado:
    st.title("🔒 Acesso Restrito")
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.subheader("Entre com suas credenciais")
        usuario = st.text_input("👤 Usuário")
        senha = st.text_input("🔑 Senha", type="password")
        if st.button("Entrar", use_container_width=True):
            if usuario in USUARIOS and USUARIOS[usuario] == senha:
                st.session_state.logado = True
                st.session_state.usuario = usuario
                st.rerun()
            else:
                st.error("❌ Usuário ou senha incorretos.")
    st.stop()

st.sidebar.markdown(f"👤 **{st.session_state.usuario}**")
if st.sidebar.button("🚪 Sair"):
    st.session_state.logado = False
    st.rerun()

st.title("📡 Ensaio de Emissão Acústica")

PASTA_RAIZ = "experimentos"

# =========================
# FUNÇÕES DE LEITURA E FILTRO
# =========================
def buscar_arquivo(diretorio, prefixo):
    arquivos = os.listdir(diretorio)
    for f in arquivos:
        if f.startswith(prefixo) and f.endswith(".txt"):
            return os.path.join(diretorio, f)
    return None

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
                try: sample_rate = float(linha.split(":")[1].strip())
                except: pass
            elif linha == "[DATA]":
                lendo = True
                continue
            elif linha == "[ENDDATA]":
                break
            elif lendo and linha:
                try: dados.append(float(linha))
                except: pass
    return sample_rate, np.array(dados)

def aplicar_filtro(x, fs, fmin, fmax):
    if fs is None: return x
    nyq = fs/2
    if fmax >= nyq: return x
    sos = signal.butter(2, [fmin, fmax], fs=fs, btype='bandpass', output='sos')
    return signal.sosfilt(sos, x)

# =========================
# NOVOS CÁLCULOS (PARAMETROS DE EA)
# =========================
def calcular_parametros_ea(sinal, fs, nome, threshold=0.1):
    # Removendo DC e calculando envelopes
    sinal_abs = np.abs(sinal)
    rms = np.sqrt(np.mean(sinal**2))
    pp = np.max(sinal) - np.min(sinal)
    energia = np.sum(sinal**2)
    
    # Counts: Cruzamentos acima do threshold (limiar)
    # Vamos usar 10% do pico máximo como threshold padrão se não for definido
    thresh_real = np.max(sinal_abs) * threshold
    counts = np.sum((sinal_abs[:-1] < thresh_real) & (sinal_abs[1:] >= thresh_real))
    
    # Duration: Tempo entre o primeiro e último cruzamento (em µs)
    indices_acima = np.where(sinal_abs > thresh_real)[0]
    if len(indices_acima) > 0:
        duration = (indices_acima[-1] - indices_acima[0]) / fs * 1e6
    else:
        duration = 0.0

    # Frequência Dominante
    X_f = np.abs(np.fft.rfft(sinal))
    freq = np.fft.rfftfreq(len(sinal), d=1/fs)
    freq_dom = freq[np.argmax(X_f[1:])+1]/1000

    return {
        "Canal": nome,
        "RMS (V)": round(rms, 5),
        "Pico (V)": round(np.max(sinal_abs), 5),
        "Counts": int(counts),
        "Duration (µs)": round(duration, 1),
        "Energia (V²s)": f"{energia:.2e}",
        "Freq. Dom. (kHz)": round(freq_dom, 1)
    }

def verificar_par(x, y, fs):
    min_len = min(len(x), len(y))
    x, y = x[:min_len], y[:min_len]
    corr = np.corrcoef(x, y)[0,1]
    cross = signal.correlate(x, y, mode='full')
    lag = np.argmax(cross) - len(x)
    atraso_us = lag / fs * 1e6 if fs else 0
    return corr, atraso_us

# =========================
# INTERFACE PRINCIPAL
# =========================
if not os.path.exists(PASTA_RAIZ):
    st.error(f"Pasta '{PASTA_RAIZ}' não encontrada.")
else:
    subpastas = sorted([f for f in os.listdir(PASTA_RAIZ) if os.path.isdir(os.path.join(PASTA_RAIZ,f))])

    st.sidebar.subheader("🔧 Configurações")
    exp = st.sidebar.selectbox("📁 Experimento", subpastas)
    
    # Filtro
    usar_filtro = st.sidebar.checkbox("Ativar filtro band-pass", value=False)
    fmin = st.sidebar.number_input("Freq Min (Hz)", value=300000)
    fmax = st.sidebar.number_input("Freq Max (Hz)", value=600000)
    
    # Threshold para Counts/Duration
    st.sidebar.markdown("---")
    thresh_p = st.sidebar.slider("Threshold p/ Counts (% do pico)", 5, 50, 10) / 100

    caminho_dir = os.path.join(PASTA_RAIZ, exp)
    arquivos_no_dir = os.listdir(caminho_dir)

    # CORREÇÃO DOS NÚMEROS ACIMA DE 100
    # Pegamos tudo que vem antes do '1.txt' ou '2.txt'
    lista_ids = []
    for f in arquivos_no_dir:
        match = re.match(r"(\d+)[12]\.txt", f)
        if match:
            lista_ids.append(match.group(1))
    
    prefixos = sorted(list(set(lista_ids)), key=int)

    par_id = st.sidebar.selectbox("🔢 Selecione o Par", prefixos, format_func=lambda x: f"Par {x}1 e {x}2")

    if st.sidebar.button("📉 Gerar Análise Completa"):
        path_x = buscar_arquivo(caminho_dir, f"{par_id}1")
        path_y = buscar_arquivo(caminho_dir, f"{par_id}2")

        fs, x = ler_txt_vallen(path_x)
        _, y = ler_txt_vallen(path_y)

        if x is None or y is None:
            st.error(f"Erro ao ler arquivos do par {par_id}.")
        else:
            x -= np.mean(x)
            y -= np.mean(y)

            if usar_filtro:
                x = aplicar_filtro(x, fs, fmin, fmax)
                y = aplicar_filtro(y, fs, fmin, fmax)

            # Verificação e Resultados
            corr, atraso = verificar_par(x, y, fs)
            st.subheader("✅ Verificação de Sincronismo")
            c1, c2 = st.columns(2)
            c1.metric("Correlação", f"{corr:.3f}")
            c2.metric("Atraso Temporal", f"{atraso:.2f} µs")

            st.subheader("📋 Tabela de Parâmetros de Emissão Acústica")
            df = pd.DataFrame([
                calcular_parametros_ea(x, fs, "Canal 1", thresh_p),
                calcular_parametros_ea(y, fs, "Canal 2", thresh_p)
            ])
            st.dataframe(df, use_container_width=True)

            # Gráficos de análise
            t = np.arange(len(x))/fs*1e6
            fig, axs = plt.subplots(2, 2, figsize=(14, 8))

            axs[0,0].plot(t, x, color="dodgerblue", lw=0.7); axs[0,0].set_title("C1 - Tempo (µs)"); axs[0,0].grid(alpha=0.3)
            axs[0,1].plot(t, y, color="forestgreen", lw=0.7); axs[0,1].set_title("C2 - Tempo (µs)"); axs[0,1].grid(alpha=0.3)

            Xf = np.abs(np.fft.rfft(x))
            Yf = np.abs(np.fft.rfft(y))
            fr = np.fft.rfftfreq(len(x), 1/fs)/1000
            axs[1,0].plot(fr, Xf, label="C1", alpha=0.8); axs[1,0].plot(fr, Yf, label="C2", alpha=0.8)
            axs[1,0].set_title("Espectro de Frequência (kHz)"); axs[1,0].legend(); axs[1,0].grid(alpha=0.3)

            axs[1,1].plot(x, y, color="purple", alpha=0.5, lw=0.5); axs[1,1].set_title("Gráfico de Lissajous"); axs[1,1].grid(alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)