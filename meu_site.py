import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

st.set_page_config(page_title="Ensaio de Emissão Acústica", layout="wide")
st.title("📡 Teste de Ensaio de Emissão Acústica")

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

    # ==========================================
    # MODO VISÃO DETALHADA
    # ==========================================
    if modo == "🔬 Visão Detalhada":
        experimento_sel = st.sidebar.selectbox("Escolha o Experimento:", subpastas)
        caminho_exp = os.path.join(PASTA_RAIZ, experimento_sel)
        arquivos = os.listdir(caminho_exp)
        prefixos = sorted(list(set([
            f[0] for f in arquivos
            if f.endswith(".txt") and len(f) >= 2 and f[0].isdigit()
        ])))

        if not prefixos:
            st.warning("Nenhum par de arquivos encontrado nessa pasta.")
        else:
            escolha_par = st.sidebar.radio(
                "Escolha o par:",
                prefixos,
                format_func=lambda x: f"Par {x}1 e {x}2"
            )

            if st.sidebar.button("📉 Gerar Análise"):
                fs, x = ler_txt_vallen(os.path.join(caminho_exp, f"{escolha_par}1.txt"))
                _, y = ler_txt_vallen(os.path.join(caminho_exp, f"{escolha_par}2.txt"))

                if x is None or y is None or len(x) == 0 or len(y) == 0:
                    st.error("Arquivos não encontrados ou vazios.")
                else:
                    x, y = x - np.mean(x), y - np.mean(y)

                    if usar_filtro and fs:
                        try:
                            sos = signal.butter(4, freq_corte, fs=fs, btype='low', output='sos')
                            x = signal.sosfilt(sos, x)
                            y = signal.sosfilt(sos, y)
                        except:
                            st.warning("Não foi possível aplicar o filtro. Tente reduzir a frequência de corte.")

                    # Eixo de tempo real
                    t_x = np.arange(len(x)) / (fs if fs else 1.0)
                    t_y = np.arange(len(y)) / (fs if fs else 1.0)

                    # Métricas
                    rms_x = np.sqrt(np.mean(x**2))
                    rms_y = np.sqrt(np.mean(y**2))
                    pp_x = np.max(x) - np.min(x)
                    pp_y = np.max(y) - np.min(y)
                    zeta_x = calcular_amortecimento(x, fs)
                    zeta_y = calcular_amortecimento(y, fs)

                    st.subheader(f"📊 {experimento_sel} — Par {escolha_par}")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("🔵 RMS Canal 1", f"{rms_x:.5f} V")
                    c2.metric("🟢 RMS Canal 2", f"{rms_y:.5f} V")
                    c3.metric("🔵 Pico a Pico C1", f"{pp_x:.5f} V")
                    c4.metric("🟢 Pico a Pico C2", f"{pp_y:.5f} V")

                    c5, c6 = st.columns(2)
                    c5.metric("🔵 Amortecimento C1 (ζ)", f"{zeta_x:.5f}")
                    c6.metric("🟢 Amortecimento C2 (ζ)", f"{zeta_y:.5f}")

                    st.markdown("---")

                    fig, axs = plt.subplots(2, 2, figsize=(14, 8))

                    # 1. Canal 1 - Tempo
                    axs[0,0].plot(t_x, x, color="blue", linewidth=0.8)
                    axs[0,0].set_title(f"Canal 1 — Tempo (fs = {fs:.0f} Hz)" if fs else "Canal 1 — Tempo")
                    axs[0,0].set_xlabel("Tempo (s)")
                    axs[0,0].set_ylabel("Amplitude (V)")
                    axs[0,0].grid(True)

                    # 2. Canal 2 - Tempo
                    axs[0,1].plot(t_y, y, color="green", linewidth=0.8)
                    axs[0,1].set_title(f"Canal 2 — Tempo (fs = {fs:.0f} Hz)" if fs else "Canal 2 — Tempo")
                    axs[0,1].set_xlabel("Tempo (s)")
                    axs[0,1].set_ylabel("Amplitude (V)")
                    axs[0,1].grid(True)

                    # 3. FFT Sobreposta
                    X_f = np.abs(np.fft.rfft(x))
                    Y_f = np.abs(np.fft.rfft(y))
                    freq = np.fft.rfftfreq(len(x), d=1/(fs if fs else 1.0))
                    pico_x = freq[np.argmax(X_f[1:]) + 1]
                    pico_y = freq[np.argmax(Y_f[1:]) + 1]
                    axs[1,0].plot(freq, X_f, color="blue", linewidth=0.8, label=f"C1 — Pico: {pico_x:.1f} Hz")
                    axs[1,0].plot(freq, Y_f, color="green", linewidth=0.8, label=f"C2 — Pico: {pico_y:.1f} Hz")
                    axs[1,0].set_title("FFT — Espectro de Frequência")
                    axs[1,0].set_xlabel("Frequência (Hz)")
                    axs[1,0].set_ylabel("Magnitude")
                    axs[1,0].set_xlim(0, (fs if fs else 1000) / 4)
                    axs[1,0].legend()
                    axs[1,0].grid(True)

                    # 4. Lissajous (C1 vs C2)
                    min_len = min(len(x), len(y))
                    axs[1,1].plot(x[:min_len], y[:min_len], color="purple", linewidth=0.5, alpha=0.7)
                    axs[1,1].set_title("Lissajous — Canal 1 vs Canal 2")
                    axs[1,1].set_xlabel("Canal 1 (V)")
                    axs[1,1].set_ylabel("Canal 2 (V)")
                    axs[1,1].grid(True)

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

    # ==========================================
    # MODO COMPARAÇÃO
    # ==========================================
    else:
        exp_a = st.sidebar.selectbox("Material A:", subpastas, index=0)
        exp_b = st.sidebar.selectbox("Material B:", subpastas, index=min(1, len(subpastas)-1))

        if st.sidebar.button("⚡ Comparar"):
            def pegar_dados(subpasta):
                caminho = os.path.join(PASTA_RAIZ, subpasta)
                arq_x = os.path.join(caminho, "11.txt")
                arq_y = os.path.join(caminho, "12.txt")
                if not os.path.exists(arq_x) or not os.path.exists(arq_y):
                    return None, None, None
                fs, x = ler_txt_vallen(arq_x)
                _, y = ler_txt_vallen(arq_y)
                return fs, x, y

            fs_a, x_a, y_a = pegar_dados(exp_a)
            fs_b, x_b, y_b = pegar_dados(exp_b)

            if x_a is None or x_b is None:
                st.error("Não foi possível carregar os dados. Verifique se os arquivos 11.txt e 12.txt existem nas pastas selecionadas.")
            else:
                x_a, x_b = x_a - np.mean(x_a), x_b - np.mean(x_b)

                if usar_filtro:
                    try:
                        sos_a = signal.butter(4, freq_corte, fs=fs_a, btype='low', output='sos')
                        x_a = signal.sosfilt(sos_a, x_a)
                        sos_b = signal.butter(4, freq_corte, fs=fs_b, btype='low', output='sos')
                        x_b = signal.sosfilt(sos_b, x_b)
                    except:
                        st.warning("Filtro não aplicado. Tente reduzir a frequência de corte.")

                # Métricas comparativas
                st.subheader(f"⚔️ Comparação: {exp_a} vs {exp_b}")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric(f"RMS {exp_a}", f"{np.sqrt(np.mean(x_a**2)):.5f} V")
                c2.metric(f"RMS {exp_b}", f"{np.sqrt(np.mean(x_b**2)):.5f} V")
                c3.metric(f"Pico a Pico {exp_a}", f"{np.max(x_a)-np.min(x_a):.5f} V")
                c4.metric(f"Pico a Pico {exp_b}", f"{np.max(x_b)-np.min(x_b):.5f} V")

                st.markdown("---")

                fig, axs = plt.subplots(2, 2, figsize=(14, 10))

                # 1. Sinais no Tempo Sobrepostos
                t_a = np.arange(len(x_a)) / (fs_a if fs_a else 1.0)
                t_b = np.arange(len(x_b)) / (fs_b if fs_b else 1.0)
                axs[0,0].plot(t_a, x_a, label=exp_a, alpha=0.8)
                axs[0,0].plot(t_b, x_b, label=exp_b, alpha=0.8)
                axs[0,0].set_title("Sinais no Tempo (Sobrepostos)")
                axs[0,0].set_xlabel("Tempo (s)")
                axs[0,0].set_ylabel("Amplitude (V)")
                axs[0,0].legend(); axs[0,0].grid(True)

                # 2. Lissajous A vs B
                ml = min(len(x_a), len(x_b))
                axs[0,1].plot(x_a[:ml], x_b[:ml], color="purple", alpha=0.5, linewidth=0.7)
                axs[0,1].set_title("Sincronia A vs B (Lissajous)")
                axs[0,1].set_xlabel(f"{exp_a} (V)")
                axs[0,1].set_ylabel(f"{exp_b} (V)")
                axs[0,1].grid(True)

                # 3. FFT Comparativa
                max_f = max(len(x_a), len(x_b))
                Xa_f = np.abs(np.fft.rfft(x_a, n=max_f))
                Xb_f = np.abs(np.fft.rfft(x_b, n=max_f))
                freq = np.fft.rfftfreq(max_f, d=1/(fs_a if fs_a else 1.0))
                axs[1,0].plot(freq, Xa_f, label=exp_a)
                axs[1,0].plot(freq, Xb_f, label=exp_b)
                axs[1,0].set_title("FFT Comparativa")
                axs[1,0].set_xlabel("Frequência (Hz)")
                axs[1,0].set_ylabel("Magnitude")
                axs[1,0].set_xlim(0, (fs_a if fs_a else 1000) / 4)
                axs[1,0].legend(); axs[1,0].grid(True)

                # 4. Correlação Cruzada
                na = (x_a - np.mean(x_a)) / (np.std(x_a) * len(x_a))
                nb = (x_b - np.mean(x_b)) / np.std(x_b)
                corr = signal.correlate(na, nb, mode='same')
                lags = np.arange(-len(corr)//2, len(corr)//2)
                axs[1,1].plot(lags, corr, color="red", linewidth=0.8)
                axs[1,1].set_title("Atraso entre Materiais (Correlação Cruzada)")
                axs[1,1].set_xlabel("Atraso (samples)")
                axs[1,1].set_ylabel("Correlação")
                axs[1,1].grid(True)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)