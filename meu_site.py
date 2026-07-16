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
USUARIOS = {"Cermat1": "acustica2026"}

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

# ✅ ATUALIZADO: Counts to Peak e Zero Crossings incluídos aqui
def calcular_features_temporais(sinal, fs, threshold_v):
    if len(sinal) == 0 or fs is None or fs <= 0:
        return 0, 0.0, 0.0, 0, 0

    sinal_abs = np.abs(sinal)
    threshold_v = max(threshold_v, 0.0)

    # Counts
    picos, _ = signal.find_peaks(sinal_abs, height=threshold_v)
    counts = len(picos)

    # Zero Crossings (Sinal centralizado e cruzando o eixo zero)
    zero_crossings = np.count_nonzero(np.diff(np.sign(sinal)))

    # Duration
    acima = np.where(sinal_abs >= threshold_v)[0]
    if len(acima) == 0:
        return int(counts), 0.0, 0.0, 0, int(zero_crossings)

    inicio = acima[0]
    fim = acima[-1]
    duration_us = (fim - inicio) / fs * 1e6

    # Rise Time e Counts to Peak
    janela = sinal_abs[inicio:fim + 1]
    if len(janela) > 0:
        idx_pico_relativo = np.argmax(janela)
        idx_pico_global = idx_pico_relativo + inicio
        rise_time_us = (idx_pico_global - inicio) / fs * 1e6
        
        # Counts to Peak: picos acima do threshold dentro do intervalo [inicio, idx_pico_global]
        counts_to_peak = np.sum((picos >= inicio) & (picos <= idx_pico_global))
    else:
        rise_time_us = 0.0
        counts_to_peak = 0

    return int(counts), float(duration_us), float(rise_time_us), int(counts_to_peak), int(zero_crossings)

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

def calcular_tabela(sinal, fs, nome, threshold_v, d1=None, d2=None):
    rms = np.sqrt(np.mean(sinal ** 2)) if len(sinal) else 0.0
    pico = np.max(np.abs(sinal)) if len(sinal) else 0.0
    pico_a_pico = (np.max(sinal) - np.min(sinal)) if len(sinal) else 0.0
    energia = np.sum(sinal ** 2) if len(sinal) else 0.0
    
    # Novas Features Temporais
    counts, duration_us, rise_time_us, c_to_peak, z_cross = calcular_features_temporais(sinal, fs, threshold_v)
    
    # RA Value: Rise Time (µs) / Amplitude (V)
    ra_value = (rise_time_us / pico) if pico > 0 else 0.0
    
    temporal_centroid_us = calcular_temporal_centroid(sinal, fs)
    freq_dom_khz, freq_centroid_khz, roll_on_khz, roll_off_khz, spectral_spread_khz = calcular_parametros_espectrais(sinal, fs)
    average_frequency_khz = (counts / duration_us) * 1000 if duration_us > 0 else 0.0

    resultado = {
        "Canal": nome,
        "RMS (V)": round(rms, 5),
        "Pico (V)": round(pico, 5),
        "Pico a Pico (V)": round(pico_a_pico, 5),
        "Energia (V²)": f"{energia:.2e}",
        "Counts": int(counts),
        "Duration (µs)": round(duration_us, 2),
        "Rise Time (µs)": round(rise_time_us, 2),
        # ✅ NOVAS FEATURES
        "RA Value (µs/V)": round(ra_value, 4),
        "Zero Crossings": int(z_cross),
        "Counts to Peak": int(c_to_peak),
        
        "Threshold (V)": round(float(threshold_v), 5),
        "Temporal Centroid (µs)": round(temporal_centroid_us, 2),
        "Average Frequency (kHz)": round(average_frequency_khz, 2),
        "Freq. Dom. (kHz)": round(freq_dom_khz, 2),
        "Frequency Centroid (kHz)": round(freq_centroid_khz, 2),
        "Roll-on Frequency (kHz)": round(roll_on_khz, 2),
        "Roll-off Frequency (kHz)": round(roll_off_khz, 2),
        "Spectral Spread (kHz)": round(spectral_spread_khz, 2),
        "ζ": round(calcular_amortecimento(sinal), 5),
    }

    if d1 is not None: resultado["d1 (mm)"] = round(d1, 3)
    if d2 is not None: resultado["d2 (mm)"] = round(d2, 3)

    return resultado

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
        x = x - np.mean(x); y = y - np.mean(y)
        if usar_filtro:
            x = aplicar_filtro(x, fs, fmin, fmax)
            y = aplicar_filtro(y, fs, fmin, fmax)
        corr, atraso = verificar_par(x, y, fs)
        d1, d2, delta_d = calcular_localizacao_toa(atraso, velocidade_ms, distancia_mm)
        tab_x = calcular_tabela(x, fs, "Canal 1", threshold_v, d1=d1, d2=d2)
        tab_y = calcular_tabela(y, fs, "Canal 2", threshold_v)
        
        linha = {
            "Material": experimento, "Par": par_id, "Correlação": round(corr, 5), "Atraso (µs)": round(atraso, 2),
            "d1 (mm)": round(d1, 3) if d1 is not None else None, "d2 (mm)": round(d2, 3) if d2 is not None else None,
            # Canal 1 features
            "C1 RMS (V)": tab_x["RMS (V)"], "C1 Pico (V)": tab_x["Pico (V)"], "C1 Energia (V²)": tab_x["Energia (V²)"],
            "C1 Counts": tab_x["Counts"], "C1 Zero Crossings": tab_x["Zero Crossings"], "C1 RA Value": tab_x["RA Value (µs/V)"],
            "C1 Counts to Peak": tab_x["Counts to Peak"], "C1 Duration (µs)": tab_x["Duration (µs)"], "C1 Rise Time (µs)": tab_x["Rise Time (µs)"],
            "C1 Frequency Centroid (kHz)": tab_x["Frequency Centroid (kHz)"], "C1 ζ": tab_x["ζ"],
            # Canal 2 features
            "C2 RMS (V)": tab_y["RMS (V)"], "C2 Pico (V)": tab_y["Pico (V)"], "C2 Energia (V²)": tab_y["Energia (V²)"],
            "C2 RA Value": tab_y["RA Value (µs/V)"], "C2 Frequency Centroid (kHz)": tab_y["Frequency Centroid (kHz)"],
        }
        resultados.append(linha)
    return pd.DataFrame(resultados)

# =========================
# INTERFACE
# =========================
if not os.path.exists(PASTA_RAIZ):
    st.error(f"Pasta '{PASTA_RAIZ}' não encontrada.")
else:
    subpastas = sorted([f for f in os.listdir(PASTA_RAIZ) if os.path.isdir(os.path.join(PASTA_RAIZ, f))])
    
    st.sidebar.subheader("🔧 Filtro e Threshold")
    usar_filtro = st.sidebar.checkbox("Ativar filtro", value=False)
    fmin = st.sidebar.number_input("Freq Min (Hz)", value=300000)
    fmax = st.sidebar.number_input("Freq Max (Hz)", value=600000)
    threshold_v = st.sidebar.number_input("Threshold absoluto (V)", min_value=0.0, value=0.10, step=0.01, format="%.4f")
    distancia_mm = st.sidebar.number_input("Distância (mm)", min_value=0.1, value=29.75, step=0.01, format="%.2f")

    st.sidebar.subheader("⚡ Velocidade p/ Mat. (m/s)")
    velocidades_por_material = {}
    for mat in subpastas:
        velocidades_por_material[mat] = st.sidebar.number_input(f"{mat}", min_value=1.0, value=3000.0, key=f"v_{mat}")

    aba1, aba2, aba3 = st.tabs(["📊 Análise Individual", "📚 Biblioteca", "📈 Comparação"])

    with aba1:
        exp = st.selectbox("📁 Experimento", subpastas)
        caminho_dir = os.path.join(PASTA_RAIZ, exp)
        prefixos = listar_pares_validos(caminho_dir)
        v_ms = velocidades_por_material.get(exp, 3000.0)

        if len(prefixos) > 0:
            par_id = st.selectbox("🔢 Selecione o Par", prefixos)
            if st.button("📉 Gerar Análise"):
                path_x, path_y = buscar_arquivo(caminho_dir, f"{par_id}1"), buscar_arquivo(caminho_dir, f"{par_id}2")
                fs, x = ler_txt_vallen(path_x); _, y = ler_txt_vallen(path_y)

                if x is not None and y is not None:
                    x = x - np.mean(x); y = y - np.mean(y)
                    if usar_filtro: x = aplicar_filtro(x, fs, fmin, fmax); y = aplicar_filtro(y, fs, fmin, fmax)
                    
                    corr, atraso = verificar_par(x, y, fs)
                    d1, d2, _ = calcular_localizacao_toa(atraso, v_ms, distancia_mm)
                    
                    st.subheader("📍 Localização ToA")
                    col1, col2 = st.columns(2)
                    col1.metric("d1 (mm)", f"{d1:.2f}")
                    col2.metric("d2 (mm)", f"{d2:.2f}")

                    st.subheader("📋 Parâmetros (Novas Features Incluídas)")
                    df = pd.DataFrame([
                        calcular_tabela(x, fs, "Canal 1", threshold_v, d1=d1, d2=d2),
                        calcular_tabela(y, fs, "Canal 2", threshold_v)
                    ])
                    st.dataframe(df, use_container_width=True)

                    excel_par = gerar_excel_bytes({"Par_Atual": df})
                    st.download_button("📥 Baixar Excel", data=excel_par, file_name=f"analise_{par_id}.xlsx")

                    fig, axs = plt.subplots(1, 2, figsize=(14, 4))
                    t = np.arange(len(x)) / fs * 1e6
                    axs[0].plot(t, x, label="C1", alpha=0.8); axs[0].plot(t, y, label="C2", alpha=0.8)
                    axs[0].set_title("Tempo (µs)"); axs[0].legend()
                    
                    xf = np.abs(np.fft.rfft(x)); freqs = np.fft.rfftfreq(len(x), 1/fs)/1000
                    axs[1].plot(freqs, xf); axs[1].set_title("FFT (kHz)")
                    st.pyplot(fig)

    with aba2:
        if st.button("🔄 Gerar Biblioteca Completa"):
            mats = []
            for m in subpastas:
                mats.append(processar_experimento_completo(os.path.join(PASTA_RAIZ, m), m, usar_filtro, fmin, fmax, threshold_v, velocidades_por_material[m], distancia_mm))
            df_bib = pd.concat(mats, ignore_index=True)
            st.dataframe(df_bib)
            st.download_button("📥 Baixar Biblioteca", data=gerar_excel_bytes({"Dataset": df_bib}), file_name="biblioteca_AE.xlsx")

    with aba3:
        st.info("Aqui você pode comparar as novas métricas (RA Value, Zero Crossings e Counts to Peak) entre os materiais selecionados para ver qual diferencia melhor cada falha.")