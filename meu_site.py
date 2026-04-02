import numpy as np
import matplotlib.pyplot as plt

def ler_txt_vallen(nome_arquivo):
    sample_rate = None
    dados = []
    lendo = False

    with open(nome_arquivo, "r", encoding="utf-8") as f:
        for linha in f:
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

# Ler os dois canais
fs1, x = ler_txt_vallen("11.txt")
fs2, y = ler_txt_vallen("12.txt")

# usar a mesma taxa
fs = fs1

# remover média
x = x - np.mean(x)
y = y - np.mean(y)

# janela de Hanning
janela_x = np.hanning(len(x))
janela_y = np.hanning(len(y))

xw = x * janela_x
yw = y * janela_y

# FFT
X = np.fft.rfft(xw)
Y = np.fft.rfft(yw)

freq_x = np.fft.rfftfreq(len(xw), d=1/fs)
freq_y = np.fft.rfftfreq(len(yw), d=1/fs)

amp_x = np.abs(X)
amp_y = np.abs(Y)

# gráfico no tempo
fig, axs = plt.subplots(2, 2, figsize=(14, 8))

axs[0, 0].plot(x, color="blue")
axs[0, 0].set_title("Canal 1 no tempo")
axs[0, 0].set_xlabel("Amostra")
axs[0, 0].set_ylabel("Amplitude (mV)")
axs[0, 0].grid(True)

axs[0, 1].plot(y, color="green")
axs[0, 1].set_title("Canal 2 no tempo")
axs[0, 1].set_xlabel("Amostra")
axs[0, 1].set_ylabel("Amplitude (mV)")
axs[0, 1].grid(True)

# gráfico FFT
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
plt.show()

# frequência dominante
pico_x = freq_x[np.argmax(amp_x[1:]) + 1]
pico_y = freq_y[np.argmax(amp_y[1:]) + 1]

print(f"Frequência dominante Canal 1: {pico_x:.2f} Hz")
print(f"Frequência dominante Canal 2: {pico_y:.2f} Hz")
