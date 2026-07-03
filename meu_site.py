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
def buscar_arquivo(diretorio, nome_base):
    alvo = f"{nome_base}.txt"
    caminho = os.path.join(diretorio, alvo)
    return caminho if os.path.exists(caminho) else None

def listar_pares_validos(diretorio):
    arquivos = os.listdir(diretorio)
    ids = set()
    for f in arquivos:
        match = re.match(r"^(\d+)([12])\.txt$", f)
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
    return sample_rate, np.array(dados, dtype=float)

def aplicar_filtro(x, fs, fmin, fmax):
    if fs is None or len(x) < 3:
        return x
    nyq = fs / 2
    if fmin <= 0 or fmax <= 0 or fmin >= fmax or fmax >= nyq:
        return x
    sos = signal.butter(2, [fmin, fmax], fs=fs, btype="bandpass", output="sos")
    return signal.sosfilt(sos, x)

def calcular_amortecimento(dados):
    if len(dados) < 3:
        return 0.0
    ref = np.max(np.abs(dados))
    if ref == 0:
        return 0.0
    picos, _ = signal.find_peaks(np.abs(dados), height=ref * 0.2)
    if len(picos) < 2:
        return 0.0
    p1 = np.abs(dados[picos[0]])
    p2 = np.abs(dados[picos[1]])
    if p1 <= 0 or p2 <= 0 or p1 <= p2:
        return 0.0
    log_dec = np.log(p1 / p2)
    return log_dec / np.sqrt((2 * np.pi) ** 2 + log_dec ** 2)

def calcular_counts_duration_risetime(sinal, fs, threshold_v):
    sinal_abs = np.abs(sinal)
    if len(sinal_abs) == 0 or fs is None or fs <= 0:
        return 0, 0.0, 0.0
    if threshold_v <= 0:
        threshold_v = 0.0
    acima = sinal_abs >= threshold_v
    if len(acima) < 2:
        return 0, 0.0, 0.0
    counts = np.sum((acima[:-1] == False) & (acima[1:] == True))
    idx_acima = np.where(acima)[0]
    if len(idx_acima) == 0:
        return int(counts), 0.0, 0.0
    inicio = idx_acima[0]
    fim = idx_acima[-1]
    duration_us = (fim - inicio) / fs * 1e6
    janela = sinal_abs[inicio:fim + 1]
    if len(janela) > 0:
        pico_local = np.argmax(janela) + inicio
        rise_time_us = (pico_local - inicio) / fs * 1e6
    else:
        rise_time_us = 0.0
    return int(counts), float(duration_us), float(rise_time_us)

def calcular_temporal_centroid(sinal, fs):
    if fs is None or len(sinal) == 0:
        return 0.0
    energia = sinal ** 2
    soma_energia = np.sum(energia)
    if soma_energia <= 0:
        return 0.0
    t_us = np.arange(len(sinal)) / fs * 1e6
    return float(np.sum(t_us * energia) / soma_energia)

def calcular_parametros_espectrais(sinal, fs, roll_on_frac=0.05, roll_off_frac=0.95):
    if fs is None or len(sinal) < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    fft_vals = np.fft.rfft(sinal)
    mag = np.abs(fft_vals)
    power = mag ** 2
    freqs = np.fft.rfftfreq(len(sinal), d=1 / fs)
    if len(freqs) <= 1:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    freqs = freqs[1:]
    power = power[1:]
    soma_power = np.sum(power)
    if soma_power <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    freq_dom_hz = freqs[np.argmax(power)]
    freq_centroid_hz = np.sum(freqs * power) / soma_power
    spectral_spread_hz = np.sqrt(np.sum(((freqs - freq_centroid_hz) ** 2) * power) / soma_power)
    cumulativa = np.cumsum(power) / soma_power
    idx_on = min(np.searchsorted(cumulativa, roll_on_frac), len(freqs) - 1)
    idx_off = min(np.searchsorted(cumulativa, roll_off_frac), len(freqs) - 1)
    return (
        float(freq_dom_hz / 1000),
        float(freq_centroid_hz / 1000),
        float(freqs[idx_on] / 1000),
        float(freqs[idx_off] / 1000),
        float(spectral_spread_hz / 1000),
    )

def calcular_tabela(sinal, fs, nome, threshold_v):
    rms = np.sqrt(np.mean(sinal ** 2)) if len(sinal) else 0.0
    pico = np.max(np.abs(sinal)) if len(sinal) else 0.0
    pico_a_pico = (np.max(sinal) - np.min(sinal)) if len(sinal) else 0.0
    energia = np.sum(sinal ** 2) if len(sinal) else 0.0
    counts, duration_us, rise_time_us = calcular_counts_duration_risetime(sinal, fs, threshold_v)
    temporal_centroid_us = calcular_temporal_centroid(sinal, fs)
    freq_dom_khz, freq_centroid_khz, roll_on_khz, roll_off_khz, spectral_spread_khz = calcular_parametros_espectrais(sinal, fs)
    average_frequency_khz = (counts / duration_us) * 1000 if duration_us > 0 else 0.0
    return {
        "Canal": nome,
        "RMS (V)": round(rms, 5),
        "Pico (V)": round(pico, 5),
        "Pico a Pico (V)": round(pico_a_pico, 5),
        "Energia (V²)": f"{energia:.2e}",
        "Counts": int(counts),
        "Duration (µs)": round(duration_us, 2),
        "Rise Time (µs)": round(rise_time_us, 2),
        "Threshold (V)": round(float(threshold_v), 5),
        "Temporal Centroid (µs)": round(temporal_centroid_us, 2),
        "Average Frequency (kHz)": round(average_frequency_khz, 2),
        "Freq. Dom. (kHz)": round(freq_dom_khz, 2),
        "Frequency Centroid (kHz)": round(freq_centroid_khz, 2),
        "Roll-on Frequency (kHz)": round(roll_on_khz, 2),
        "Roll-off Frequency (kHz)": round(roll_off_khz, 2),
        "Spectral Spread (kHz)": round(spectral_spread_khz, 2),
        "ζ": round(calcular_amortecimento(sinal), 5)
    }

def verificar_par(x, y, fs):
    min_len = min(len(x), len(y))
    x = x[:min_len]
    y = y[:min_len]
    if len(x) < 2 or len(y) < 2:
        return 0.0, 0.0
    corr = np.corrcoef(x, y)[0, 1]
    cross = signal.correlate(x, y, mode="full")
    lag = np.argmax(cross) - (len(x) - 1)
    atraso_us = (lag / fs) * 1e6 if fs else 0.0
    return float(corr), float(atraso_us)

def calcular_localizacao_toa(atraso_us, velocidade_ms, distancia_mm):
    if velocidade_ms <= 0 or distancia_mm <= 0:
        return None, None, None
    atraso_s = atraso_us * 1e-6
    delta_d = velocidade_ms * atraso_s * 1000
    d1 = (distancia_mm + delta_d) / 2
    d2 = distancia_mm - d1
    d1 = max(0.0, min(d1, distancia_mm))
    d2 = max(0.0, min(d2, distancia_mm))
    return float(d1), float(d2), float(delta_d)

def gerar_excel_bytes(planilhas):
    output = io.BytesIO()
    try:
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            for nome, df in planilhas.items():
                df.to_excel(writer, index=False, sheet_name=nome[:31])
        output.seek(0)
        return output
    except:
        return None

def processar_experimento_completo(diretorio, experimento, usar_filtro, fmin, fmax, threshold_v, velocidade_ms, distancia_mm):
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
        tab_x = calcular_tabela(x, fs, "Canal 1", threshold_v)
        tab_y = calcular_tabela(y, fs, "Canal 2", threshold_v)
        d1, d2, delta_d = calcular_localizacao_toa(atraso, velocidade_ms, distancia_mm)

        # Energia como float para estatísticas
        energia_x = np.sum(x ** 2)
        energia_y = np.sum(y ** 2)

        linha = {
            "Material": experimento,
            "Par": par_id,
            "Correlação": round(corr, 5),
            "Atraso (µs)": round(atraso, 2),
            "d1 (mm)": round(d1, 3) if d1 is not None else None,
            "d2 (mm)": round(d2, 3) if d2 is not None else None,
            "C1 RMS (V)": tab_x["RMS (V)"],
            "C1 Pico (V)": tab_x["Pico (V)"],
            "C1 Energia (V²)": round(energia_x, 6),
            "C1 Counts": tab_x["Counts"],
            "C1 Duration (µs)": tab_x["Duration (µs)"],
            "C1 Rise Time (µs)": tab_x["Rise Time (µs)"],
            "C1 Temporal Centroid (µs)": tab_x["Temporal Centroid (µs)"],
            "C1 Average Frequency (kHz)": tab_x["Average Frequency (kHz)"],
            "C1 Freq. Dom. (kHz)": tab_x["Freq. Dom. (kHz)"],
            "C1 Frequency Centroid (kHz)": tab_x["Frequency Centroid (kHz)"],
            "C1 Roll-on (kHz)": tab_x["Roll-on Frequency (kHz)"],
            "C1 Roll-off (kHz)": tab_x["Roll-off Frequency (kHz)"],
            "C1 Spectral Spread (kHz)": tab_x["Spectral Spread (kHz)"],
            "C1 ζ": tab_x["ζ"],
            "C2 RMS (V)": tab_y["RMS (V)"],
            "C2 Pico (V)": tab_y["Pico (V)"],
            "C2 Energia (V²)": round(energia_y, 6),
            "C2 Counts": tab_y["Counts"],
            "C2 Duration (µs)": tab_y["Duration (µs)"],
            "C2 Rise Time (µs)": tab_y["Rise Time (µs)"],
            "C2 Temporal Centroid (µs)": tab_y["Temporal Centroid (µs)"],
            "C2 Average Frequency (kHz)": tab_y["Average Frequency (kHz)"],
            "C2 Freq. Dom. (kHz)": tab_y["Freq. Dom. (kHz)"],
            "C2 Frequency Centroid (kHz)": tab_y["Frequency Centroid (kHz)"],
            "C2 Roll-on (kHz)": tab_y["Roll-on Frequency (kHz)"],
            "C2 Roll-off (kHz)": tab_y["Roll-off Frequency (kHz)"],
            "C2 Spectral Spread (kHz)": tab_y["Spectral Spread (kHz)"],
            "C2 ζ": tab_y["ζ"],
        }
        resultados.append(linha)
    return pd.DataFrame(resultados)

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

    st.sidebar.subheader("📏 Threshold")
    threshold_v = st.sidebar.number_input(
        "Threshold absoluto (V)",
        min_value=0.0,
        value=0.10,
        step=0.01,
        format="%.4f"
    )

    st.sidebar.subheader("📍 Localização ToA")
    velocidade_ms = st.sidebar.number_input("Velocidade de propagação (m/s)", min_value=1.0, value=3000.0, step=100.0)
    distancia_mm = st.sidebar.number_input("Distância entre sensores (mm)", min_value=0.1, value=29.75, step=0.01, format="%.2f")

    # =========================
    # ABAS PRINCIPAIS
    # =========================
    aba1, aba2, aba3 = st.tabs([
        "📊 Análise Individual",
        "📚 Biblioteca de Sinais",
        "📈 Comparação entre Materiais"
    ])

    # =========================
    # ABA 1 — ANÁLISE INDIVIDUAL
    # =========================
    with aba1:
        exp = st.selectbox("📁 Experimento", subpastas)
        caminho_dir = os.path.join(PASTA_RAIZ, exp)
        prefixos = listar_pares_validos(caminho_dir)

        if len(prefixos) == 0:
            st.warning("Nenhum par válido encontrado nessa pasta.")
        else:
            par_id = st.selectbox("🔢 Selecione o Par", prefixos, format_func=lambda x: f"Par {x}1 e {x}2")

            if st.button("📉 Gerar Análise"):
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

                    d1, d2, delta_d = calcular_localizacao_toa(atraso, velocidade_ms, distancia_mm)

                    st.subheader("📍 Localização da Fonte (Time of Arrival)")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("d1 (mm)", f"{d1:.2f}" if d1 is not None else "N/A")
                    col2.metric("d2 (mm)", f"{d2:.2f}" if d2 is not None else "N/A")
                    col3.metric("Δd (mm)", f"{delta_d:.2f}" if delta_d is not None else "N/A")

                    fig_loc, ax_loc = plt.subplots(figsize=(10, 2))
                    ax_loc.set_xlim(0, distancia_mm)
                    ax_loc.set_ylim(-0.5, 0.5)
                    ax_loc.axhline(0, color="gray", linewidth=2)
                    ax_loc.plot(0, 0, "bs", markersize=14, label="Sensor 1")
                    ax_loc.plot(distancia_mm, 0, "gs", markersize=14, label="Sensor 2")
                    if d1 is not None:
                        ax_loc.plot(d1, 0, "r*", markersize=18, label=f"Fonte ({d1:.2f} mm)")
                    ax_loc.set_xlabel("Posição (mm)")
                    ax_loc.set_title("Localização estimada da fonte acústica")
                    ax_loc.legend(loc="upper center", ncol=3)
                    ax_loc.set_yticks([])
                    ax_loc.grid(True, axis="x")
                    st.pyplot(fig_loc)
                    plt.close(fig_loc)

                    st.caption("Roll-on = 5% | Roll-off = 95% da energia espectral acumulada.")

                    st.subheader("📋 Parâmetros")
                    df = pd.DataFrame([
                        calcular_tabela(x, fs, "Canal 1", threshold_v),
                        calcular_tabela(y, fs, "Canal 2", threshold_v)
                    ])
                    st.dataframe(df, use_container_width=True)

                    excel_par = gerar_excel_bytes({"Par_Atual": df})
                    if excel_par:
                        st.download_button(
                            "📥 Baixar Excel do Par",
                            data=excel_par,
                            file_name=f"{exp}_par_{par_id}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

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

                    axs[1, 0].plot(freqs, Xf, color="blue", linewidth=0.8, label=f"C1 — {pico_x:.1f} kHz")
                    axs[1, 0].plot(freqs, Yf, color="green", linewidth=0.8, label=f"C2 — {pico_y:.1f} kHz")
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

    # =========================
    # ABA 2 — BIBLIOTECA DE SINAIS
    # =========================
    with aba2:
        st.subheader("📚 Biblioteca de Sinais — Dataset Completo")
        st.markdown("Gera um único arquivo Excel com **todos os experimentos** rotulados por material.")

        materiais_selecionados = st.multiselect(
            "Selecione os materiais para incluir na biblioteca:",
            options=subpastas,
            default=subpastas
        )

        if st.button("🔄 Gerar Biblioteca"):
            if not materiais_selecionados:
                st.warning("Selecione ao menos um material.")
            else:
                todos = []
                barra = st.progress(0)
                for i, mat in enumerate(materiais_selecionados):
                    caminho_mat = os.path.join(PASTA_RAIZ, mat)
                    df_mat = processar_experimento_completo(
                        caminho_mat, mat, usar_filtro, fmin, fmax,
                        threshold_v, velocidade_ms, distancia_mm
                    )
                    todos.append(df_mat)
                    barra.progress((i + 1) / len(materiais_selecionados))

                if todos:
                    df_biblioteca = pd.concat(todos, ignore_index=True)

                    st.success(f"✅ Biblioteca gerada com {len(df_biblioteca)} eventos de {len(materiais_selecionados)} materiais.")
                    st.dataframe(df_biblioteca, use_container_width=True)

                    excel_bib = gerar_excel_bytes({"Biblioteca": df_biblioteca})
                    if excel_bib:
                        st.download_button(
                            "📥 Baixar Biblioteca Completa (Excel)",
                            data=excel_bib,
                            file_name="biblioteca_sinais_AE.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                else:
                    st.error("Nenhum dado encontrado nos materiais selecionados.")

    # =========================
    # ABA 3 — COMPARAÇÃO ENTRE MATERIAIS
    # =========================
    with aba3:
        st.subheader("📈 Comparação Estatística entre Materiais")

        materiais_comp = st.multiselect(
            "Selecione os materiais para comparar:",
            options=subpastas,
            default=subpastas,
            key="comp_mat"
        )

        FEATURES_DISPONIVEIS = {
            "RMS (V)": "C1 RMS (V)",
            "Pico (V)": "C1 Pico (V)",
            "Energia (V²)": "C1 Energia (V²)",
            "Counts": "C1 Counts",
            "Duration (µs)": "C1 Duration (µs)",
            "Rise Time (µs)": "C1 Rise Time (µs)",
            "Temporal Centroid (µs)": "C1 Temporal Centroid (µs)",
            "Average Frequency (kHz)": "C1 Average Frequency (kHz)",
            "Frequency Centroid (kHz)": "C1 Frequency Centroid (kHz)",
            "Spectral Spread (kHz)": "C1 Spectral Spread (kHz)",
            "Roll-on (kHz)": "C1 Roll-on (kHz)",
            "Roll-off (kHz)": "C1 Roll-off (kHz)",
            "ζ (Amortecimento)": "C1 ζ",
        }

        features_escolhidas = st.multiselect(
            "Selecione os parâmetros para comparar:",
            options=list(FEATURES_DISPONIVEIS.keys()),
            default=["RMS (V)", "Frequency Centroid (kHz)", "Energia (V²)", "Counts"]
        )

        if st.button("📊 Gerar Comparação"):
            if not materiais_comp:
                st.warning("Selecione ao menos um material.")
            elif not features_escolhidas:
                st.warning("Selecione ao menos um parâmetro.")
            else:
                todos_comp = []
                barra2 = st.progress(0)
                for i, mat in enumerate(materiais_comp):
                    caminho_mat = os.path.join(PASTA_RAIZ, mat)
                    df_mat = processar_experimento_completo(
                        caminho_mat, mat, usar_filtro, fmin, fmax,
                        threshold_v, velocidade_ms, distancia_mm
                    )
                    todos_comp.append(df_mat)
                    barra2.progress((i + 1) / len(materiais_comp))

                if todos_comp:
                    df_comp = pd.concat(todos_comp, ignore_index=True)

                    # Tabela estatística resumo
                    st.subheader("📋 Tabela Resumo Estatístico")
                    colunas_stat = [FEATURES_DISPONIVEIS[f] for f in features_escolhidas if FEATURES_DISPONIVEIS[f] in df_comp.columns]
                    df_stat = df_comp.groupby("Material")[colunas_stat].agg(["mean", "std", "min", "max"]).round(4)
                    st.dataframe(df_stat, use_container_width=True)

                    excel_stat = gerar_excel_bytes({"Comparacao": df_comp, "Resumo": df_stat.reset_index()})
                    if excel_stat:
                        st.download_button(
                            "📥 Baixar Comparação (Excel)",
                            data=excel_stat,
                            file_name="comparacao_materiais_AE.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

                    # Gráficos de barras com erro
                    st.subheader("📊 Gráficos Comparativos")
                    n_features = len(features_escolhidas)
                    cols_graf = 2
                    rows_graf = (n_features + 1) // cols_graf

                    fig_comp, axs_comp = plt.subplots(rows_graf, cols_graf, figsize=(14, 4 * rows_graf))
                    axs_comp = np.array(axs_comp).flatten()

                    cores = plt.cm.tab10(np.linspace(0, 1, len(materiais_comp)))

                    for idx, feat_nome in enumerate(features_escolhidas):
                        col_key = FEATURES_DISPONIVEIS[feat_nome]
                        if col_key not in df_comp.columns:
                            continue

                        ax = axs_comp[idx]
                        medias = []
                        erros = []
                        nomes = []

                        for mat in materiais_comp:
                            subset = df_comp[df_comp["Material"] == mat][col_key].dropna()
                            if len(subset) > 0:
                                medias.append(subset.mean())
                                erros.append(subset.std())
                                nomes.append(mat)

                        bars = ax.bar(nomes, medias, yerr=erros, capsize=5,
                                      color=cores[:len(nomes)], alpha=0.85, edgecolor="black")
                        ax.set_title(feat_nome, fontsize=11, fontweight="bold")
                        ax.set_ylabel(feat_nome)
                        ax.set_xlabel("Material")
                        ax.tick_params(axis="x", rotation=20)
                        ax.grid(True, axis="y", alpha=0.4)

                        for bar, val in zip(bars, medias):
                            ax.text(bar.get_x() + bar.get_width() / 2,
                                    bar.get_height() * 1.02,
                                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

                    # Esconder eixos extras
                    for j in range(idx + 1, len(axs_comp)):
                        axs_comp[j].set_visible(False)

                    plt.tight_layout()
                    st.pyplot(fig_comp)
                    plt.close(fig_comp)

                    st.caption("Barras de erro representam o desvio padrão entre os eventos do material.")