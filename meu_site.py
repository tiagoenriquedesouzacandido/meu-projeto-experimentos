import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================
st.set_page_config(page_title="Ensaio de Emissão Acústica", layout="wide")
st.title("📡 Ensaio de Emissão Acústica")

PASTA_RAIZ = "experimentos"

# ============================================================
# FUNÇÃO: Lê arquivo .txt no formato Vallen
# Extrai o SampleRate e os dados entre [DATA] e [ENDDATA]
# ============================================================
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

# ============================================================
# FUNÇÃO: Calcula amortecimento pelo decremento logarítmico
# Compara os dois primeiros picos do sinal
# ============================================================
def calcular_amortecimento(dados, fs):
    picos, _ = signal.find_peaks(np.abs(dados), height=np.max(np.abs(dados)) * 0.1)
    if len(picos) < 2:
        return 0.0
    val_picos = np.abs(dados[picos])[:2]
    if val_picos[0] <= 0 or val_picos[1] <= 0:
        return 0.0
    log_dec = np.log(val_picos[0] / val_picos[1])
    zeta = log_dec / np.sqrt((2 * np.pi)**2 + log_dec**2)
    return zeta

# ============================================================
# FUNÇÃO: Gera a Tabela 1 com os parâmetros do sinal
# RMS, Pico a Pico, Amplitude Máx, Energia, Freq. Dominante, Amortecimento
# ============================================================
def calcular_tabela1(sinal, fs, label="Canal"):
    rms = np.sqrt(np.mean(sinal**2))
    pp = np.max(sinal) - np.min(sinal)
    amp_max = np.max(np.abs(sinal))
    energia = np.sum(sinal**2)
    zeta = calcular_amortecimento(sinal, fs)
    X_f = np.abs(np.fft.rfft(sinal))
    freq = np.fft.rfftfreq(len(sinal), d=1 / (fs if fs else 1.0))
    freq_dom = freq[np.argmax(X_f[1:]) + 1] if len(X_f) > 1 else 0.0
    return {
        "Canal": label,
        "RMS (V)": round(rms, 6),
        "Pico a Pico (V)": round(pp, 6),
        "Amplitude Máx (V)": round(amp_max, 6),
        "Energia": f"{energia:.4e}",
        "Freq. Dominante (kHz)": round(freq_dom / 1000, 2),
        "Amortecimento (ζ)": round(zeta, 6),
    }

# ============================================================
# FUNÇÃO: Aplica filtro passa-banda de ordem baixa (ordem 2)
# Ordem 2 evita a "deriva" (rampa) que ordem 4 causava
# ============================================================
def aplicar_filtro(sinal, fs, freq_min, freq_max):
    if fs is None:
        return sinal
    nyquist = fs / 2
    if freq_min <= 0 or freq_max >= nyquist or freq_min >= freq_max:
        st.warning(f"⚠️ Frequências inválidas para fs={fs:.0f} Hz (Nyquist={nyquist/1000:.0f} kHz). Filtro não aplicado.")
        return sinal
    try:
        # Ordem 2 para evitar distorção no início do sinal
        sos = signal.butter(2, [freq_min, freq_max], fs=fs, btype='bandpass', output='sos')
        return signal.sosfilt(sos, sinal)
    except Exception as e:
        st.warning(f"Erro no filtro: {e}")
        return sinal

# ============================================================
# INÍCIO DA INTERFACE
# ============================================================
if not os.path.exists(PASTA_RAIZ):
    st.error(f"Pasta '{PASTA_RAIZ}' não encontrada.")
else:
    subpastas = sorted([
        f for f in os.listdir(PASTA_RAIZ)
        if os.path.isdir(os.path.join(PASTA_RAIZ, f))
    ])

    modo = st.sidebar.radio("Selecione o Modo:", ["🔬 Visão Detalhada", "🆚 Comparação"])

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Filtro Passa-Banda")
    usar_filtro = st.sidebar.checkbox("Ativar filtro passa-banda", value=False)

    # Filtro ajustado para faixa real de EA: 300–600 kHz
    if usar_filtro:
        freq_min = st.sidebar.number_input("Freq. Mínima (Hz)", value=300000, step=10000)
        freq_max = st.sidebar.number_input("Freq. Máxima (Hz)", value=600000, step=10000)
    else:
        freq_min = None
        freq_max = None

    # ============================================================
    # MODO: VISÃO DETALHADA
    # Analisa um par de arquivos por vez
    # ============================================================
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
                caminho_x = os.path.join(caminho_exp, f"{escolha_par}1.txt")
                caminho_y = os.path.join(caminho_exp, f"{escolha_par}2.txt")

                if not os.path.exists(caminho_x):
                    todos = os.listdir(caminho_exp)
                    arq_x = next((f for f in todos if f.startswith(f"{escolha_par}1") and f.endswith(".txt")), None)
                    arq_y = next((f for f in todos if f.startswith(f"{escolha_par}2") and f.endswith(".txt")), None)
                    caminho_x = os.path.join(caminho_exp, arq_x) if arq_x else caminho_x
                    caminho_y = os.path.join(caminho_exp, arq_y) if arq_y else caminho_y

                fs, x = ler_txt_vallen(caminho_x)
                _, y = ler_txt_vallen(caminho_y)

                if x is None or y is None or len(x) == 0 or len(y) == 0:
                    st.error("Arquivos não encontrados ou vazios.")
                else:
                    x = x - np.mean(x)
                    y = y - np.mean(y)

                    # Aplica filtro se ativado
                    if usar_filtro:
                        x = aplicar_filtro(x, fs, freq_min, freq_max)
                        y = aplicar_filtro(y, fs, freq_min, freq_max)

                    t_x = np.arange(len(x)) / (fs if fs else 1.0)
                    t_y = np.arange(len(y)) / (fs if fs else 1.0)

                    st.subheader(f"📊 {experimento_sel} — Par {escolha_par}")

                    # ----------------------------------------
                    # TABELA 1 — aparece PRIMEIRO, antes dos gráficos
                    # ----------------------------------------
                    st.subheader("📋 Tabela 1 — Parâmetros do Sinal")
                    linha_c1 = calcular_tabela1(x, fs, label="Canal 1")
                    linha_c2 = calcular_tabela1(y, fs, label="Canal 2")
                    df_tabela = pd.DataFrame([linha_c1, linha_c2])
                    st.dataframe(df_tabela, use_container_width=True)

                    st.markdown("---")

                    fig, axs = plt.subplots(2, 2, figsize=(14, 8))

                    # Gráfico 1: Canal 1 no tempo
                    axs[0, 0].plot(t_x * 1e6, x, color="blue", linewidth=0.8)
                    axs[0, 0].set_title(f"Canal 1 — Tempo (fs = {fs/1e6:.1f} MHz)" if fs else "Canal 1 — Tempo")
                    axs[0, 0].set_xlabel("Tempo (µs)")
                    axs[0, 0].set_ylabel("Amplitude (V)")
                    axs[0, 0].grid(True)

                    # Gráfico 2: Canal 2 no tempo
                    axs[0, 1].plot(t_y * 1e6, y, color="green", linewidth=0.8)
                    axs[0, 1].set_title(f"Canal 2 — Tempo (fs = {fs/1e6:.1f} MHz)" if fs else "Canal 2 — Tempo")
                    axs[0, 1].set_xlabel("Tempo (µs)")
                    axs[0, 1].set_ylabel("Amplitude (V)")
                    axs[0, 1].grid(True)

                    # Gráfico 3: FFT em kHz — mostra frequências dominantes
                    X_f = np.abs(np.fft.rfft(x))
                    Y_f = np.abs(np.fft.rfft(y))
                    freq = np.fft.rfftfreq(len(x), d=1 / (fs if fs else 1.0))
                    freq_khz = freq / 1000
                    pico_x = freq[np.argmax(X_f[1:]) + 1] / 1000
                    pico_y = freq[np.argmax(Y_f[1:]) + 1] / 1000
                    axs[1, 0].plot(freq_khz, X_f, color="blue", linewidth=0.8, label=f"C1 — Pico: {pico_x:.1f} kHz")
                    axs[1, 0].plot(freq_khz, Y_f, color="green", linewidth=0.8, label=f"C2 — Pico: {pico_y:.1f} kHz")
                    axs[1, 0].set_title("FFT — Espectro de Frequência")
                    axs[1, 0].set_xlabel("Frequência (kHz)")
                    axs[1, 0].set_ylabel("Magnitude")
                    axs[1, 0].set_xlim(0, (fs if fs else 1e6) / 2000)
                    axs[1, 0].legend()
                    axs[1, 0].grid(True)

                    # Gráfico 4: Lissajous — relação entre Canal 1 e Canal 2
                    min_len = min(len(x), len(y))
                    axs[1, 1].plot(x[:min_len], y[:min_len], color="purple", linewidth=0.5, alpha=0.7)
                    axs[1, 1].set_title("Lissajous — Canal 1 vs Canal 2")
                    axs[1, 1].set_xlabel("Canal 1 (V)")
                    axs[1, 1].set_ylabel("Canal 2 (V)")
                    axs[1, 1].grid(True)

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

    # ============================================================
    # MODO: COMPARAÇÃO
    # Compara dois experimentos lado a lado
    # ============================================================
    else:
        exp_a = st.sidebar.selectbox("Material A:", subpastas, index=0)
        exp_b = st.sidebar.selectbox("Material B:", subpastas, index=min(1, len(subpastas) - 1))

        if st.sidebar.button("⚡ Comparar"):
            def pegar_dados(subpasta):
                caminho = os.path.join(PASTA_RAIZ, subpasta)
                todos = os.listdir(caminho)
                arq_x = next((f for f in todos if f.startswith("11") and f.endswith(".txt")), None)
                arq_y = next((f for f in todos if f.startswith("12") and f.endswith(".txt")), None)
                if not arq_x or not arq_y:
                    return None, None, None
                fs, x = ler_txt_vallen(os.path.join(caminho, arq_x))
                _, y = ler_txt_vallen(os.path.join(caminho, arq_y))
                return fs, x, y

            fs_a, x_a, y_a = pegar_dados(exp_a)
            fs_b, x_b, y_b = pegar_dados(exp_b)

            if x_a is None or x_b is None:
                st.error("Não foi possível carregar os dados. Verifique se existem arquivos começando com 11 e 12 nas pastas selecionadas.")
            else:
                x_a = x_a - np.mean(x_a)
                x_b = x_b - np.mean(x_b)

                # Aplica filtro nos dois experimentos se ativado
                if usar_filtro:
                    x_a = aplicar_filtro(x_a, fs_a, freq_min, freq_max)
                    x_b = aplicar_filtro(x_b, fs_b, freq_min, freq_max)

                st.subheader(f"⚔️ Comparação: {exp_a} vs {exp_b}")

                # Tabela comparativa de parâmetros
                st.subheader("📋 Tabela 1 — Parâmetros Comparativos")
                linha_a = calcular_tabela1(x_a, fs_a, label=exp_a)
                linha_b = calcular_tabela1(x_b, fs_b, label=exp_b)
                df_comp = pd.DataFrame([linha_a, linha_b])
                st.dataframe(df_comp, use_container_width=True)

                st.markdown("---")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric(f"RMS {exp_a}", f"{np.sqrt(np.mean(x_a**2)):.5f} V")
                c2.metric(f"RMS {exp_b}", f"{np.sqrt(np.mean(x_b**2)):.5f} V")
                c3.metric(f"Pico a Pico {exp_a}", f"{np.max(x_a) - np.min(x_a):.5f} V")
                c4.metric(f"Pico a Pico {exp_b}", f"{np.max(x_b) - np.min(x_b):.5f} V")

                st.markdown("---")

                fig, axs = plt.subplots(2, 2, figsize=(14, 10))

                # Gráfico 1: Sinais sobrepostos no tempo (µs)
                t_a = np.arange(len(x_a)) / (fs_a if fs_a else 1.0) * 1e6
                t_b = np.arange(len(x_b)) / (fs_b if fs_b else 1.0) * 1e6
                axs[0, 0].plot(t_a, x_a, label=exp_a, alpha=0.8)
                axs[0, 0].plot(t_b, x_b, label=exp_b, alpha=0.8)
                axs[0, 0].set_title("Sinais no Tempo (Sobrepostos)")
                axs[0, 0].set_xlabel("Tempo (µs)")
                axs[0, 0].set_ylabel("Amplitude (V)")
                axs[0, 0].legend()
                axs[0, 0].grid(True)

                # Gráfico 2: Lissajous comparativo
                ml = min(len(x_a), len(x_b))
                axs[0, 1].plot(x_a[:ml], x_b[:ml], color="purple", alpha=0.5, linewidth=0.7)
                axs[0, 1].set_title("Sincronia A vs B (Lissajous)")
                axs[0, 1].set_xlabel(f"{exp_a} (V)")
                axs[0, 1].set_ylabel(f"{exp_b} (V)")
                axs[0, 1].grid(True)

                # Gráfico 3: FFT comparativa em kHz
                max_f = max(len(x_a), len(x_b))
                Xa_f = np.abs(np.fft.rfft(x_a, n=max_f))
                Xb_f = np.abs(np.fft.rfft(x_b, n=max_f))
                freq = np.fft.rfftfreq(max_f, d=1 / (fs_a if fs_a else 1.0)) / 1000
                axs[1, 0].plot(freq, Xa_f, label=exp_a)
                axs[1, 0].plot(freq, Xb_f, label=exp_b)
                axs[1, 0].set_title("FFT Comparativa")
                axs[1, 0].set_xlabel("Frequência (kHz)")
                axs[1, 0].set_ylabel("Magnitude")
                axs[1, 0].set_xlim(0, (fs_a if fs_a else 1e6) / 2000)
                axs[1, 0].legend()
                axs[1, 0].grid(True)

                # Gráfico 4: Correlação cruzada — mede atraso entre os dois sinais
                na = (x_a - np.mean(x_a)) / (np.std(x_a) * len(x_a))
                nb = (x_b - np.mean(x_b)) / np.std(x_b)
                corr = signal.correlate(na, nb, mode='same')
                lags = np.arange(-len(corr) // 2, len(corr) // 2)
                axs[1, 1].plot(lags, corr, color="red", linewidth=0.8)
                axs[1, 1].set_title("Atraso entre Materiais (Correlação Cruzada)")
                axs[1, 1].set_xlabel("Atraso (samples)")
                axs[1, 1].set_ylabel("Correlação")
                axs[1, 1].grid(True)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)