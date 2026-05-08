import streamlit as st
import os
import re
import io
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd

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
# FUNÇÕES AUXILIARES
# =========================
def buscar_arquivo(diretorio, prefixo):
    arquivos = os.listdir(diretorio)
    for f in arquivos:
        if f.startswith(prefixo) and f.endswith(".txt"):
            return os.path.join(diretorio, f)
    return None

def listar_pares_validos(diretorio):
    arquivos = os.listdir(diretorio)
    ids = set()

    for f in arquivos:
        match = re.match(r"(\d+)([12])\.txt$", f)
        if match:
            ids.add(match.group(1))

    pares_validos = []
    for pid in ids:
        arq1 = buscar_arquivo(diretorio, f"{pid}1")
        arq2 = buscar_arquivo(diretorio, f"{pid}2")
        if arq1 and arq2:
            pares_validos.append(pid)

    return sorted(pares_validos, key=int)

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

def aplicar_filtro(x, fs, fmin, fmax):
    if fs is None:
        return x

    nyq = fs / 2
    if fmax >= nyq or fmin <= 0 or fmin >= fmax:
        return x

    sos = signal.butter(2, [fmin, fmax], fs=fs, btype="bandpass", output="sos")
    return signal.sosfilt(sos, x)

def calcular_amortecimento(dados):
    if len(dados) == 0:
        return 0.0

    pico_ref = np.max(np.abs(dados))
    if pico_ref == 0:
        return 0.0

    picos, _ = signal.find_peaks(np.abs(dados), height=pico_ref * 0.2)

    if len(picos) < 2:
        return 0.0

    p1 = np.abs(dados[picos[0]])
    p2 = np.abs(dados[picos[1]])

    if p1 <= 0 or p2 <= 0 or p1 <= p2:
        return 0.0

    log_dec = np.log(p1 / p2)
    return log_dec / np.sqrt((2 * np.pi) ** 2 + log_dec ** 2)

def calcular_counts_duration_risetime(sinal, fs, threshold_frac=0.1):
    sinal_abs = np.abs(sinal)

    if len(sinal_abs) == 0:
        return 0, 0.0, 0.0, 0.0

    pico = np.max(sinal_abs)
    if pico == 0:
        return 0, 0.0, 0.0, 0.0

    threshold = pico * threshold_frac

    # Counts = número de cruzamentos ascendentes do limiar
    counts = np.sum((sinal_abs[:-1] < threshold) & (sinal_abs[1:] >= threshold))

    # Índices acima do limiar
    acima = np.where(sinal_abs >= threshold)[0]

    if len(acima) > 0 and fs:
        duracao_us = (acima[-1] - acima[0]) / fs * 1e6
        inicio_idx = acima[0]
        pico_idx = np.argmax(sinal_abs)
        if pico_idx >= inicio_idx:
            rise_time_us = (pico_idx - inicio_idx) / fs * 1e6
        else:
            rise_time_us = 0.0
    else:
        duracao_us = 0.0
        rise_time_us = 0.0

    return int(counts), float(duracao_us), float(rise_time_us), float(threshold)

def calcular_tabela(sinal, fs, nome, threshold_frac=0.1):
    rms = np.sqrt(np.mean(sinal ** 2)) if len(sinal) else 0.0
    pp = (np.max(sinal) - np.min(sinal)) if len(sinal) else 0.0
    energia = np.sum(sinal ** 2) if len(sinal) else 0.0
    pico = np.max(np.abs(sinal)) if len(sinal) else 0.0

    counts, duration_us, rise_time_us, threshold = calcular_counts_duration_risetime(
        sinal, fs, threshold_frac
    )

    if fs and len(sinal) > 1:
        X_f = np.abs(np.fft.rfft(sinal))
        freq = np.fft.rfftfreq(len(sinal), d=1 / fs)
        if len(X_f) > 1:
            freq_dom = freq[np.argmax(X_f[1:]) + 1] / 1000
        else:
            freq_dom = 0.0
    else:
        freq_dom = 0.0

    return {
        "Canal": nome,
        "RMS (V)": round(rms, 5),
        "Pico (V)": round(pico, 5),
        "Pico a Pico (V)": round(pp, 5),
        "Energia (V²)": f"{energia:.2e}",
        "Freq. Dom. (kHz)": round(freq_dom, 2),
        "Counts": counts,
        "Duration (µs)": round(duration_us, 2),
        "Rise Time (µs)": round(rise_time_us, 2),
        "Threshold (V)": round(threshold, 5),
        "ζ": round(calcular_amortecimento(sinal), 5)
    }

def verificar_par(x, y, fs):
    min_len = min(len(x), len(y))
    x = x[:min_len]
    y = y[:min_len]

    if len(x) < 2:
        return 0.0, 0.0

    corr = np.corrcoef(x, y)[0, 1]

    cross = signal.correlate(x, y, mode="full")
    lag = np.argmax(cross) - (len(x) - 1)
    atraso_us = (lag / fs) * 1e6 if fs else 0.0

    return corr, atraso_us

def gerar_excel_par(df_tabela, experimento, par_id):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_tabela.to_excel(writer, index=False, sheet_name="Par")
    output.seek(0)
    return output

def gerar_excel_experimento(diretorio, experimento, usar_filtro, fmin, fmax, threshold_frac):
    resultados = []

    pares = listar_pares_validos(diretorio)

    for par_id in pares:
        path_x = buscar_arquivo(diretorio, f"{par_id}1")
        path_y = buscar_arquivo(diretorio, f"{par_id}2")

        fs, x = ler_txt_vallen(path_x)
        _, y = ler_txt_vallen(path_y)

        if x is None or y is None or fs is None:
            continue

        x = x - np.mean(x)
        y = y - np.mean(y)

        if usar_filtro:
            x = aplicar_filtro(x, fs, fmin, fmax)
            y = aplicar_filtro(y, fs, fmin, fmax)

        corr, atraso = verificar_par(x, y, fs)

        tab_x = calcular_tabela(x, fs, "Canal 1", threshold_frac)
        tab_y = calcular_tabela(y, fs, "Canal 2", threshold_frac)

        linha = {
            "Experimento": experimento,
            "Par": par_id,
            "Correlação": round(corr, 5),
            "Atraso (µs)": round(atraso, 2),

            "C1 RMS (V)": tab_x["RMS (V)"],
            "C1 Pico (V)": tab_x["Pico (V)"],
            "C1 Pico a Pico (V)": tab_x["Pico a Pico (V)"],
            "C1 Energia (V²)": tab_x["Energia (V²)"],
            "C1 Freq. Dom. (kHz)": tab_x["Freq. Dom. (kHz)"],
            "C1 Counts": tab_x["Counts"],
            "C1 Duration (µs)": tab_x["Duration (µs)"],
            "C1 Rise Time (µs)": tab_x["Rise Time (µs)"],
            "C1 Threshold (V)": tab_x["Threshold (V)"],
            "C1 ζ": tab_x["ζ"],

            "C2 RMS (V)": tab_y["RMS (V)"],
            "C2 Pico (V)": tab_y["Pico (V)"],
            "C2 Pico a Pico (V)": tab_y["Pico a Pico (V)"],
            "C2 Energia (V²)": tab_y["Energia (V²)"],
            "C2 Freq. Dom. (kHz)": tab_y["Freq. Dom. (kHz)"],
            "C2 Counts": tab_y["Counts"],
            "C2 Duration (µs)": tab_y["Duration (µs)"],
            "C2 Rise Time (µs)": tab_y["Rise Time (µs)"],
            "C2 Threshold (V)": tab_y["Threshold (V)"],
            "C2 ζ": tab_y["ζ"],
        }

        resultados.append(linha)

    df_exp = pd.DataFrame(resultados)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_exp.to_excel(writer, index=False, sheet_name="Experimento")
    output.seek(0)

    return output, df_exp

# =========================
# INTERFACE
# =========================
if not os.path.exists(PASTA_RAIZ):
    st.error(f"Pasta '{PASTA_RAIZ}' não encontrada.")
else:
    subpastas = sorted(
        [f for f in os.listdir(PASTA_RAIZ) if os.path.isdir(os.path.join(PASTA_RAIZ, f))]
    )

    st.sidebar.subheader("🔧 Filtro")
    usar_filtro = st.sidebar.checkbox("Ativar filtro", value=False)
    fmin = st.sidebar.number_input("Freq Min (Hz)", value=300000)
    fmax = st.sidebar.number_input("Freq Max (Hz)", value=600000)

    st.sidebar.subheader("📏 Limiar")
    threshold_percent = st.sidebar.slider("Threshold (% do pico)", 1, 50, 10)
    threshold_frac = threshold_percent / 100

    exp = st.sidebar.selectbox("📁 Experimento", subpastas)
    caminho_dir = os.path.join(PASTA_RAIZ, exp)

    prefixos = listar_pares_validos(caminho_dir)

    if len(prefixos) == 0:
        st.warning("Nenhum par válido encontrado nessa pasta.")
    else:
        par_id = st.sidebar.selectbox(
            "🔢 Selecione o Par",
            prefixos,
            format_func=lambda x: f"Par {x}1 e {x}2"
        )

        if st.sidebar.button("📉 Gerar Análise"):
            path_x = buscar_arquivo(caminho_dir, f"{par_id}1")
            path_y = buscar_arquivo(caminho_dir, f"{par_id}2")

            fs, x = ler_txt_vallen(path_x)
            _, y = ler_txt_vallen(path_y)

            if x is None or y is None or fs is None:
                st.error(f"Erro ao ler arquivos do par {par_id}.")
            else:
                x = x - np.mean(x)
                y = y - np.mean(y)

                if usar_filtro:
                    x = aplicar_filtro(x, fs, fmin, fmax)
                    y = aplicar_filtro(y, fs, fmin, fmax)

                corr, atraso = verificar_par(x, y, fs)

                st.subheader("✅ Verificação do Par")
                col1, col2 = st.columns(2)
                col1.metric("Correlação", f"{corr:.3f}")
                col2.metric("Atraso (µs)", f"{atraso:.2f}")

                if corr < 0.3:
                    st.warning("⚠️ Baixa correlação detectada.")
                else:
                    st.success("Par consistente ✅")

                st.subheader("📋 Tabela 1 — Parâmetros")
                df = pd.DataFrame([
                    calcular_tabela(x, fs, "Canal 1", threshold_frac),
                    calcular_tabela(y, fs, "Canal 2", threshold_frac)
                ])
                st.dataframe(df, use_container_width=True)

                # Download Excel do par atual
                excel_par = gerar_excel_par(df, exp, par_id)
                st.download_button(
                    label="📥 Baixar Excel do Par Atual",
                    data=excel_par,
                    file_name=f"{exp}_par_{par_id}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

                # Download Excel do experimento inteiro
                excel_exp, df_exp = gerar_excel_experimento(
                    caminho_dir, exp, usar_filtro, fmin, fmax, threshold_frac
                )

                st.download_button(
                    label="📥 Baixar Excel do Experimento Inteiro",
                    data=excel_exp,
                    file_name=f"{exp}_completo.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

                # Gráficos
                t = np.arange(len(x)) / fs * 1e6
                fig, axs = plt.subplots(2, 2, figsize=(14, 8))

                axs[0, 0].plot(t, x, color="blue", linewidth=0.8)
                axs[0, 0].set_title("Canal 1 — Sinal no Tempo")
                axs[0, 0].set_xlabel("Tempo (µs)")
                axs[0, 0].set_ylabel("Amplitude (V)")
                axs[0, 0].grid(True)

                axs[0, 1].plot(t, y, color="green", linewidth=0.8)
                axs[0, 1].set_title("Canal 2 — Sinal no Tempo")
                axs[0, 1].set_xlabel("Tempo (µs)")
                axs[0, 1].set_ylabel("Amplitude (V)")
                axs[0, 1].grid(True)

                Xf = np.abs(np.fft.rfft(x))
                Yf = np.abs(np.fft.rfft(y))
                freqs = np.fft.rfftfreq(len(x), 1 / fs) / 1000

                pico_x = freqs[np.argmax(Xf[1:]) + 1] if len(Xf) > 1 else 0
                pico_y = freqs[np.argmax(Yf[1:]) + 1] if len(Yf) > 1 else 0

                axs[1, 0].plot(freqs, Xf, color="blue", linewidth=0.8, label=f"C1 — Pico: {pico_x:.1f} kHz")
                axs[1, 0].plot(freqs, Yf, color="green", linewidth=0.8, label=f"C2 — Pico: {pico_y:.1f} kHz")
                axs[1, 0].set_title("FFT — Espectro de Frequência")
                axs[1, 0].set_xlabel("Frequência (kHz)")
                axs[1, 0].set_ylabel("Magnitude")
                axs[1, 0].legend()
                axs[1, 0].grid(True)

                min_len = min(len(x), len(y))
                axs[1, 1].plot(x[:min_len], y[:min_len], color="purple", linewidth=0.5, alpha=0.7)
                axs[1, 1].set_title("Lissajous — Canal 1 vs Canal 2")
                axs[1, 1].set_xlabel("Canal 1 (V)")
                axs[1, 1].set_ylabel("Canal 2 (V)")
                axs[1, 1].grid(True)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                st.subheader("📊 Prévia dos dados do experimento")
                st.dataframe(df_exp, use_container_width=True)