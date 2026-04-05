import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

st.set_page_config(page_title="Analisador de Experimentos", layout="wide")
st.title("📊 Analisador de Experimentos Online")

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

if not os.path.exists(PASTA_RAIZ):
    st.error(f"Pasta '{PASTA_RAIZ}' não encontrada.")
else:
    subpastas = sorted([f for f in os.listdir(PASTA_RAIZ) if os.path.isdir(os.path.join(PASTA_RAIZ, f))])

    if not subpastas:
        st.warning("Nenhuma subpasta encontrada dentro de 'experimentos'.")
    else:
        experimento_sel = st.sidebar.selectbox("🔬 Escolha o Experimento:", subpastas)
        caminho_exp = os.path.join(PASTA_RAIZ, experimento_sel)

        arquivos = os.listdir(caminho_exp)
        prefixos = sorted(list(set([f[0] for f in arquivos if f.endswith(".txt") and len(f) >= 2 and f[0].isdigit()])))

        if not prefixos:
            st.warning("Nenhum par de arquivos encontrado nessa pasta.")
        else:
            escolha_par = st.sidebar.radio(
                "📁 Escolha o par:",
                prefixos,
                format_func=lambda x: f"Par {x}1 e {x}2"
            )

            # Filtro opcional
            st.sidebar.markdown("---")
            usar_filtro = st.sidebar.checkbox("🔧 Aplicar filtro passa-baixa", value=False)
            freq_corte = st.sidebar.slider("Frequência de corte (Hz)", 10, 5000, 1000) if usar_filtro else None

            if st.sidebar.button("📈 Gerar Gráfico"):
                arq_x = os.path.join(caminho_exp, f"{escolha_par}1.txt")
                arq_y = os.path.join(caminho_exp, f"{escolha_par}2.txt")

                fs1, x = ler_txt_vallen(arq_x)
                fs2, y = ler_txt_vallen(arq_y)

                if x is None or y is None or len(x) == 0 or len(y) == 0:
                    st.error("Arquivos não encontrados ou vazios.")
                else:
                    fs = fs1 if fs1 else 1.0

                    # Remove média
                    x = x - np.mean(x)
                    y = y - np.mean(y)

                    # Aplica filtro se ativado
                    if usar_filtro and freq_corte:
                        try:
                            sos = signal.butter(4, freq_corte, fs=fs, btype='low', output='sos')
                            x = signal.sosfilt(sos, x)
                            y = signal.sosfilt(sos, y)
                        except:
                            st.warning("Não foi possível aplicar o filtro com esses parâmetros.")

                    # Eixo de tempo
                    t_x = np.arange(len(x)) / fs
                    t_y = np.arange(len(y)) / fs

                    # FFT
                    janela_x = np.hanning(len(x))
                    janela_y = np.hanning(len(y))
                    X_fft = np.abs(np.fft.rfft(x * janela_x))
                    Y_fft = np.abs(np.fft.rfft(y * janela_y))
                    freq_x = np.fft.rfftfreq(len(x), d=1/fs)
                    freq_y = np.fft.rfftfreq(len(y), d=1/fs)

                    # Pico de frequência
                    pico_x = freq_x[np.argmax(X_fft[1:]) + 1]
                    pico_y = freq_y[np.argmax(Y_fft[1:]) + 1]

                    # RMS e Pico a Pico
                    rms_x = np.sqrt(np.mean(x**2))
                    rms_y = np.sqrt(np.mean(y**2))
                    pp_x = np.max(x) - np.min(x)
                    pp_y = np.max(y) - np.min(y)

                    # Métricas no topo
                    st.subheader(f"📊 {experimento_sel} — Par {escolha_par}")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("🔵 RMS Canal 1", f"{rms_x:.4f}")
                    col2.metric("🟢 RMS Canal 2", f"{rms_y:.4f}")
                    col3.metric("🔵 Pico a Pico C1", f"{pp_x:.4f}")
                    col4.metric("🟢 Pico a Pico C2", f"{pp_y:.4f}")

                    st.markdown("---")

                    col_a, col_b = st.columns(2)
                    col_c, col_d = st.columns(2)

                    # Gráfico 1 - Tempo Canal 1
                    with col_a:
                        fig1, ax1 = plt.subplots(figsize=(6, 3))
                        ax1.plot(t_x, x, color="blue", linewidth=0.8)
                        ax1.set_title("Canal 1 — Tempo")
                        ax1.set_xlabel("Tempo (s)")
                        ax1.set_ylabel("Amplitude")
                        ax1.grid(True)
                        st.pyplot(fig1)
                        plt.close(fig1)

                    # Gráfico 2 - Tempo Canal 2
                    with col_b:
                        fig2, ax2 = plt.subplots(figsize=(6, 3))
                        ax2.plot(t_y, y, color="green", linewidth=0.8)
                        ax2.set_title("Canal 2 — Tempo")
                        ax2.set_xlabel("Tempo (s)")
                        ax2.set_ylabel("Amplitude")
                        ax2.grid(True)
                        st.pyplot(fig2)
                        plt.close(fig2)

                    # Gráfico 3 - FFT
                    with col_c:
                        fig3, ax3 = plt.subplots(figsize=(6, 3))
                        ax3.plot(freq_x, X_fft, color="blue", linewidth=0.8, label=f"Pico: {pico_x:.1f} Hz")
                        ax3.plot(freq_y, Y_fft, color="green", linewidth=0.8, label=f"Pico: {pico_y:.1f} Hz")
                        ax3.set_title("FFT — Frequência")
                        ax3.set_xlabel("Frequência (Hz)")
                        ax3.set_ylabel("Amplitude")
                        ax3.set_xlim(0, fs/2)
                        ax3.legend()
                        ax3.grid(True)
                        st.pyplot(fig3)
                        plt.close(fig3)

                    # Gráfico 4 - X vs Y
                    with col_d:
                        min_len = min(len(x), len(y))
                        fig4, ax4 = plt.subplots(figsize=(6, 3))
                        ax4.plot(x[:min_len], y[:min_len], color="purple", linewidth=0.5, alpha=0.7)
                        ax4.set_title("Canal 1 vs Canal 2 (Lissajous)")
                        ax4.set_xlabel("Canal 1")
                        ax4.set_ylabel("Canal 2")
                        ax4.grid(True)
                        st.pyplot(fig4)
                        plt.close(fig4)
