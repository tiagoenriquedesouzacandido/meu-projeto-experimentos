import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="FFT dos sinais", layout="wide")
st.title("Análise FFT de arquivos .txt")

def ler_txt_vallen_arquivo(arquivo):
    sample_rate = None
    dados = []
    lendo = False

    conteudo = arquivo.read().decode("utf-8", errors="ignore").splitlines()

    for linha in conteudo:
        linha = linha.strip()

        if linha.startswith("SampleRate[Hz]:"):
            sample_rate = float(linha.split(":")[1].strip())

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

arquivo_x = st.file_uploader("Envie o arquivo do Canal 1 (ex: 11.txt)", type=["txt"])
arquivo_y = st.file_uploader("Envie o arquivo do Canal 2 (ex: 12.txt)", type=["txt"])

if arquivo_x is not None and arquivo_y is not None:
    fs1, x = ler_txt_vallen_arquivo(arquivo_x)
    fs2, y = ler_txt_vallen_arquivo(arquivo_y)

    if fs1 is None or fs2 is None:
        st.error("Não foi possível encontrar o SampleRate nos arquivos.")
    else:
        fs = fs1

        x = x - np.mean(x)
        y = y - np.mean(y)

        janela_x = np.hanning(len(x))
        janela_y = np.hanning(len(y))

        xw = x * janela_x
        yw = y * janela_y

        X = np.fft.rfft(xw)
        Y = np.fft.rfft(yw)

        freq_x = np.fft.rfftfreq(len(xw), d=1/fs)
        freq_y = np.fft.rfftfreq(len(yw), d=1/fs)

        amp_x = np.abs(X)
        amp_y = np.abs(Y)

        pico_x = freq_x[np.argmax(amp_x[1:]) + 1]
        pico_y = freq_y[np.argmax(amp_y[1:]) + 1]

        st.subheader("Frequências dominantes")
        col1, col2 = st.columns(2)
        col1.metric("Canal 1", f"{pico_x:.2f} Hz")
        col2.metric("Canal 2", f"{pico_y:.2f} Hz")

        fig, axs = plt.subplots(2, 2, figsize=(14, 8))

        axs[0, 0].plot(x, color="blue")
        axs[0, 0].set_title("Canal 1 no tempo")
        axs[0, 0].set_xlabel("Amostra")
        axs[0, 0].set_ylabel("Amplitude")
        axs[0, 0].grid(True)

        axs[0, 1].plot(y, color="green")
        axs[0, 1].set_title("Canal 2 no tempo")
        axs[0, 1].set_xlabel("Amostra")
        axs[0, 1].set_ylabel("Amplitude")
        axs[0, 1].grid(True)

        axs[1, 0].plot(freq_x, amp_x, color="blue")
        axs[1, 0].set_title("FFT - Canal 1")
        axs[1, 0].set_xlabel("Frequência (Hz)")
        axs[1, 0].set_ylabel("Amplitude")
        axs[1, 0].grid(True)
        axs[1, 0].set_xlim(0, fs/2)

        axs[1, 1].plot(freq_y, amp_y, color="green")
        axs[1, 1].set_title("FFT - Canal 2")
        axs[1, 1].set_xlabel("Frequência (Hz)")
        axs[1, 1].set_ylabel("Amplitude")
        axs[1, 1].grid(True)
        axs[1, 1].set_xlim(0, fs/2)

        plt.tight_layout()
        st.pyplot(fig)
