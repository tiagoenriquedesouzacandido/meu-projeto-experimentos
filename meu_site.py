import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import pandas as pd

# ============================================================
# SISTEMA DE LOGIN — PROTEÇÃO DOS DADOS NÃO PUBLICADOS
# ============================================================
def verificar_login():
    if "autenticado" not in st.session_state:
        st.session_state["autenticado"] = False

    if not st.session_state["autenticado"]:
        st.set_page_config(page_title="Acesso Restrito", layout="centered")
        st.title("🔐 Acesso Restrito — CERMAT/UFSC")
        st.markdown("Este sistema contém dados experimentais não publicados.")
        usuario = st.text_input("Usuário")
        senha = st.text_input("Senha", type="password")
        if st.button("Entrar"):
            if usuario == "Cermat1" and senha == "acustica2026":
                st.session_state["autenticado"] = True
                st.rerun()
            else:
                st.error("❌ Usuário ou senha incorretos.")
        st.stop()

verificar_login()

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================
st.set_page_config(page_title="Ensaio de Emissão Acústica", layout="wide")
st.title("📡 Ensaio de Emissão Acústica — CERMAT/UFSC")

PASTA_RAIZ = "experimentos"

# ============================================================
# FUNÇÃO: LER ARQUIVO .TXT DO SISTEMA VALLEN
# Retorna sample_rate (Hz), dados (mV) e metadata (dict)
# ============================================================
def ler_txt_vallen(caminho_arquivo):
    if not os.path.exists(caminho_arquivo):
        return None, None, {}

    sample_rate = None
    dados = []
    metadata = {}
    lendo = False

    with open(caminho_arquivo, "r", encoding="utf-8", errors="ignore") as f:
        for linha in f:
            linha = linha.strip()

            if linha.startswith("SampleRate[Hz]:"):
                try:
                    sample_rate = float(linha.split(":")[1].strip())
                except:
                    pass

            elif linha.startswith("Time:"):
                metadata["time_raw"] = linha.replace("Time:", "").strip()

            elif linha.startswith("Channel:"):
                try:
                    metadata["channel"] = int(linha.split(":")[1].strip())
                except:
                    pass

            elif linha.startswith("PreTriggerSamples:"):
                try:
                    metadata["pretrigger"] = int(linha.split(":")[1].strip())
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

    return sample_rate, np.array(dados), metadata


# ============================================================
# FUNÇÃO: CONVERTER TEMPO DO METADATA PARA SEGUNDOS
# Formato: "30 14:41:55 378.4898" → segundos desde meia-noite
# Usado para calcular o Δt entre Canal 1 e Canal 2 (ToA)
# ============================================================
def parse_tempo(time_raw):
    try:
        partes = time_raw.strip().split()
        hms = partes[1].split(":")
        h, m, s = int(hms[0]), int(hms[1]), int(hms[2])
        ms = float(partes[2]) / 1000.0
        return h * 3600 + m * 60 + s + ms
    except:
        return None


# ============================================================
# FUNÇÃO: CALCULAR FEATURES TEMPORAIS E ESPECTRAIS
# Todos os cálculos são feitos sobre o sinal em mV
# ============================================================
def calcular_features(dados, fs, threshold_mv=0.5):
    n = len(dados)
    if n == 0:
        return {}

    dt = 1.0 / fs
    t = np.arange(n) * dt

    # --- FEATURES TEMPORAIS BÁSICAS ---

    # RMS: energia média do sinal
    rms = np.sqrt(np.mean(dados**2))

    # Pico: valor máximo absoluto
    pico = np.max(np.abs(dados))

    # Energia: integral da potência no tempo
    energia = np.sum(dados**2) * dt

    # Duration: tempo total do sinal em µs
    duration_us = n * dt * 1e6

    # Rise Time: tempo do início até o pico máximo (em µs)
    idx_pico = np.argmax(np.abs(dados))
    rise_time_us = idx_pico * dt * 1e6

    # Counts: cruzamentos positivos do threshold (de baixo para cima)
    acima = dados > threshold_mv
    counts = int(np.sum(np.diff(acima.astype(int)) == 1))

    # Counts to Peak: cruzamentos até o pico máximo
    acima_ate_pico = dados[:idx_pico] > threshold_mv
    counts_to_peak = int(np.sum(np.diff(acima_ate_pico.astype(int)) == 1))

    # RA Value: Rise Time / Amplitude — indica tipo de fratura
    # Alto → cisalhamento; Baixo → tração
    ra_value = rise_time_us / pico if pico > 0 else 0

    # Zero Crossings: número de vezes que o sinal cruza o zero
    zc = int(np.sum(np.diff(np.sign(dados)) != 0))

    # --- FEATURES ESPECTRAIS (FFT) ---
    freqs = fftfreq(n, d=dt)
    espectro = np.abs(fft(dados))
    freqs_pos = freqs[:n // 2]
    espectro_pos = espectro[:n // 2]

    # Frequência Centroide: centro de massa do espectro
    if np.sum(espectro_pos) > 0:
        freq_centroide = np.sum(freqs_pos * espectro_pos) / np.sum(espectro_pos)
    else:
        freq_centroide = 0

    # Frequência de Pico: frequência com maior amplitude
    idx_max_freq = np.argmax(espectro_pos)
    freq_pico = freqs_pos[idx_max_freq]

    # --- FEATURES AVANÇADAS ---

    # ZCR: taxa de cruzamentos de zero por segundo
    zcr = zc / (n * dt)

    # Amplitude Não-Dimensional: pico / RMS — quão impulsivo é o sinal
    amp_nd = pico / rms if rms > 0 else 0

    # Partial Power: potência em banda 1–5 kHz / potência total
    banda_inf, banda_sup = 1000, 5000
    mask_banda = (freqs_pos >= banda_inf) & (freqs_pos <= banda_sup)
    pot_banda = np.sum(espectro_pos[mask_banda]**2)
    pot_total = np.sum(espectro_pos**2)
    partial_power = pot_banda / pot_total if pot_total > 0 else 0

    # Frequência de Iniciação: freq. dominante no 1º quarto do sinal
    quarto = max(n // 4, 4)
    espectro_inicio = np.abs(fft(dados[:quarto]))[:quarto // 2]
    freqs_inicio = fftfreq(quarto, d=dt)[:quarto // 2]
    freq_iniciacao = freqs_inicio[np.argmax(espectro_inicio)] if len(espectro_inicio) > 0 else 0

    # Frequência de Reverberação: freq. dominante no último quarto do sinal
    espectro_rev = np.abs(fft(dados[-quarto:]))[:quarto // 2]
    freq_reverbacao = freqs_inicio[np.argmax(espectro_rev)] if len(espectro_rev) > 0 else 0

    # Frequência de Pico Ponderada: média ponderada pelo espectro²
    freq_pico_pond = np.sum(freqs_pos * espectro_pos**2) / pot_total if pot_total > 0 else 0

    return {
        "RMS (mV)":                  round(rms, 4),
        "Pico (mV)":                 round(pico, 4),
        "Energia (mV²·s)":           round(energia, 6),
        "Duration (µs)":             round(duration_us, 2),
        "Rise Time (µs)":            round(rise_time_us, 2),
        "Counts":                    counts,
        "Counts to Peak":            counts_to_peak,
        "RA Value (µs/mV)":          round(ra_value, 4),
        "Zero Crossings":            zc,
        "Freq. Centroide (Hz)":      round(freq_centroide, 2),
        "Freq. Pico (Hz)":           round(freq_pico, 2),
        "ZCR (Hz)":                  round(zcr, 2),
        "Amp. Não-Dimensional":      round(amp_nd, 4),
        "Partial Power (banda)":     round(partial_power, 6),
        "Freq. Iniciação (Hz)":      round(freq_iniciacao, 2),
        "Freq. Reverberação (Hz)":   round(freq_reverbacao, 2),
        "Freq. Pico Ponderada (Hz)": round(freq_pico_pond, 2),
    }


# ============================================================
# FUNÇÃO: CALCULAR AMORTECIMENTO LOGARÍTMICO
# ζ próximo de 0 → pouco amortecimento; próximo de 1 → muito amortecido
# ============================================================
def calcular_amortecimento(dados, fs):
    picos, _ = signal.find_peaks(dados)
    if len(picos) < 2:
        return 0.0
    val_picos = dados[picos][:2]
    if val_picos[0] <= 0 or val_picos[1] <= 0:
        return 0.0
    log_dec = np.log(val_picos[0] / val_picos[1])
    zeta = log_dec / np.sqrt((2 * np.pi)**2 + log_dec**2)
    return round(zeta, 6)


# ============================================================
# INÍCIO DA INTERFACE
# ============================================================
if not os.path.exists(PASTA_RAIZ):
    st.error(f"Pasta '{PASTA_RAIZ}' não encontrada.")
    st.stop()

subpastas = sorted([
    f for f in os.listdir(PASTA_RAIZ)
    if os.path.isdir(os.path.join(PASTA_RAIZ, f))
])

if not subpastas:
    st.warning("Nenhum experimento encontrado.")
    st.stop()

# ============================================================
# SIDEBAR — ordem: Modo → Experimento → Par → Parâmetros
# ============================================================
modo = st.sidebar.radio("Selecione o Modo:", [
    "🔬 Visão Detalhada",
    "🆚 Comparação",
    "📚 Biblioteca de Sinais"
])

st.sidebar.markdown("---")

# Seletor de experimento sempre visível no topo
material = st.sidebar.selectbox("Selecione o Experimento:", subpastas)
pasta_material = os.path.join(PASTA_RAIZ, material)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Parâmetros de Análise")

# Threshold em mV
threshold_mv = st.sidebar.number_input(
    "Threshold (mV)", min_value=0.01, max_value=100.0, value=0.5, step=0.1
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔊 Velocidade de Propagação")

# Velocidade por material para cálculo de distância via ToA: d = v × Δt
vel_propagacao = {}
for pasta in subpastas:
    vel_propagacao[pasta] = st.sidebar.number_input(
        f"Vel. {pasta} (m/s)",
        min_value=100,
        max_value=10000,
        value=2000,
        step=100
    )

# Botão de logout
st.sidebar.markdown("---")
if st.sidebar.button("🚪 Sair"):
    st.session_state["autenticado"] = False
    st.rerun()


# ============================================================
# MODO 1: VISÃO DETALHADA
# Analisa um par (Canal 1 + Canal 2) por vez
# ============================================================
if modo == "🔬 Visão Detalhada":

    # Busca e organiza pares de arquivos
    # Lógica: último dígito do nome = canal (1 ou 2)
    #         tudo antes = chave do par (ex: "3" para 31.txt e 32.txt)
    arquivos = sorted([f for f in os.listdir(pasta_material) if f.endswith(".txt")])
    pares = {}
    for arq in arquivos:
        nome = os.path.splitext(arq)[0]
        if len(nome) >= 2:
            ultimo = nome[-1]
            chave = nome[:-1]
            pares.setdefault(chave, {})
            if ultimo == "1":
                pares[chave]["c1"] = arq
            elif ultimo == "2":
                pares[chave]["c2"] = arq

    if not pares:
        st.warning("Nenhum par de arquivos encontrado.")
        st.stop()

    # Ordenação numérica correta: 1, 2, 3 ... 10, 11 (não 1, 10, 11, 2)
    pares_ordenados = sorted(pares.keys(), key=lambda x: int(x) if x.isdigit() else x)
    par_sel = st.sidebar.selectbox("Selecione o Par:", pares_ordenados)
    par = pares[par_sel]

    # Leitura dos dois canais
    arq_c1 = os.path.join(pasta_material, par.get("c1", ""))
    arq_c2 = os.path.join(pasta_material, par.get("c2", ""))
    fs1, dados1, meta1 = ler_txt_vallen(arq_c1)
    fs2, dados2, meta2 = ler_txt_vallen(arq_c2)

    st.subheader(f"📂 {material} — Par {par_sel}")

    # ---- CÁLCULO DO ToA (Time of Arrival) ----
    # Δt entre os dois canais → estima localização da fonte de emissão
    if "time_raw" in meta1 and "time_raw" in meta2:
        t1 = parse_tempo(meta1["time_raw"])
        t2 = parse_tempo(meta2["time_raw"])
        if t1 is not None and t2 is not None:
            delta_t = abs(t2 - t1)
            vel = vel_propagacao.get(material, 2000)
            distancia = delta_t * vel
            with st.expander("⏱️ Time of Arrival (ToA) — Localização da Fonte"):
                st.markdown(
                    f"**ToA Canal 1:** `{meta1['time_raw']}`  \n"
                    f"**ToA Canal 2:** `{meta2['time_raw']}`  \n"
                    f"**Δt:** `{delta_t * 1e6:.4f} µs`  \n"
                    f"**Distância estimada da fonte:** `{distancia * 100:.4f} cm`  \n"
                    f"*(velocidade configurada: {vel} m/s)*"
                )

    # ---- GRÁFICOS E TABELAS ----
    col1, col2 = st.columns(2)

    for col, dados, fs, meta, label in [
        (col1, dados1, fs1, meta1, "Canal 1"),
        (col2, dados2, fs2, meta2, "Canal 2"),
    ]:
        if dados is None or len(dados) == 0:
            col.warning(f"{label}: arquivo não encontrado ou vazio.")
            continue

        with col:
            st.markdown(f"**{label}** | fs = {fs:.0f} Hz | {len(dados)} amostras | Unidade: mV")

            t_eixo = np.arange(len(dados)) / fs * 1e6  # tempo em µs

            fig, axes = plt.subplots(2, 1, figsize=(6, 5))

            # Sinal no tempo
            axes[0].plot(t_eixo, dados, color="steelblue", linewidth=0.8)
            axes[0].axhline(threshold_mv, color="red", linestyle="--",
                            linewidth=0.8, label=f"Threshold ({threshold_mv} mV)")
            axes[0].axhline(-threshold_mv, color="red", linestyle="--", linewidth=0.8)
            axes[0].set_xlabel("Tempo (µs)")
            axes[0].set_ylabel("Amplitude (mV)")
            axes[0].set_title(f"Sinal no Tempo — {label}")
            axes[0].legend(fontsize=7)
            axes[0].grid(True, alpha=0.3)

            # Espectro de frequências (FFT)
            n = len(dados)
            freqs = fftfreq(n, d=1 / fs)
            espectro = np.abs(fft(dados))
            freqs_pos = freqs[:n // 2]
            espectro_pos = espectro[:n // 2]

            axes[1].plot(freqs_pos, espectro_pos, color="darkorange", linewidth=0.8)
            axes[1].set_xlabel("Frequência (Hz)")
            axes[1].set_ylabel("Amplitude (mV)")
            axes[1].set_title("Espectro de Frequências (FFT)")
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Tabela de features
            features = calcular_features(dados, fs, threshold_mv)
            amort = calcular_amortecimento(dados, fs)
            features["Amortecimento (ζ)"] = amort

            df_feat = pd.DataFrame(features.items(), columns=["Parâmetro", "Valor"])
            st.dataframe(df_feat, use_container_width=True)


# ============================================================
# MODO 2: COMPARAÇÃO ENTRE MATERIAIS
# Mostra médias das features de cada material lado a lado
# ============================================================
elif modo == "🆚 Comparação":
    st.subheader("🆚 Comparação Estatística entre Materiais")

    resumo = {}

    for pasta in subpastas:
        pasta_mat = os.path.join(PASTA_RAIZ, pasta)
        arquivos = sorted(os.listdir(pasta_mat))
        todas_features = []

        for arq in arquivos:
            if arq.endswith(".txt"):
                fs, dados, meta = ler_txt_vallen(os.path.join(pasta_mat, arq))
                if dados is not None and len(dados) > 0:
                    feat = calcular_features(dados, fs, threshold_mv)
                    feat["Amortecimento (ζ)"] = calcular_amortecimento(dados, fs)
                    todas_features.append(feat)

        if todas_features:
            df_mat = pd.DataFrame(todas_features)
            resumo[pasta] = df_mat.mean().round(4)

    if resumo:
        df_comp = pd.DataFrame(resumo).T
        st.markdown("### 📋 Tabela de Médias por Material")
        st.dataframe(df_comp, use_container_width=True)

        st.markdown("### 📊 Gráficos Comparativos")
        for col_feat in df_comp.columns:
            fig, ax = plt.subplots(figsize=(8, 3))
            cores = plt.cm.tab10(np.linspace(0, 1, len(df_comp)))
            ax.bar(df_comp.index, df_comp[col_feat], color=cores, edgecolor="black")
            ax.set_title(f"{col_feat} — Média por Material")
            ax.set_ylabel(col_feat)
            ax.set_xlabel("Material")
            ax.grid(True, axis="y", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    else:
        st.warning("Nenhum dado encontrado para comparação.")


# ============================================================
# MODO 3: BIBLIOTECA DE SINAIS
# Dataset completo com todas as features de todos os arquivos
# Exportável como CSV para uso em machine learning
# ============================================================
elif modo == "📚 Biblioteca de Sinais":
    st.subheader("📚 Biblioteca de Sinais — Dataset Completo")
    st.markdown(
        "Todos os sinais de todos os materiais são carregados aqui com suas features calculadas. "
        "Exporte o CSV para usar em modelos de aprendizado de máquina."
    )

    registros = []

    for pasta in subpastas:
        pasta_mat = os.path.join(PASTA_RAIZ, pasta)
        arquivos = sorted(os.listdir(pasta_mat))

        for arq in arquivos:
            if arq.endswith(".txt"):
                caminho = os.path.join(pasta_mat, arq)
                fs, dados, meta = ler_txt_vallen(caminho)
                if dados is not None and len(dados) > 0:
                    feat = calcular_features(dados, fs, threshold_mv)
                    feat["Amortecimento (ζ)"] = calcular_amortecimento(dados, fs)
                    feat["Material"] = pasta
                    feat["Arquivo"] = arq
                    feat["Canal"] = meta.get("channel", "?")
                    feat["Tempo (raw)"] = meta.get("time_raw", "?")
                    feat["fs (Hz)"] = fs
                    registros.append(feat)

    if registros:
        df_lib = pd.DataFrame(registros)

        # Reordena: identificação primeiro, depois features
        cols_info = ["Material", "Arquivo", "Canal", "Tempo (raw)", "fs (Hz)"]
        cols_feat = [c for c in df_lib.columns if c not in cols_info]
        df_lib = df_lib[cols_info + cols_feat]

        st.dataframe(df_lib, use_container_width=True)

        # Estatísticas por material
        st.markdown("### 📈 Estatísticas por Material")
        st.dataframe(
            df_lib.groupby("Material")[cols_feat].agg(["mean", "std"]).round(4),
            use_container_width=True
        )

        # Exportar CSV
        csv = df_lib.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Exportar Dataset (CSV)",
            data=csv,
            file_name="biblioteca_sinais_AE.csv",
            mime="text/csv"
        )

        st.success(f"✅ {len(registros)} sinais carregados na biblioteca.")
    else:
        st.warning("Nenhum sinal encontrado na pasta de experimentos.")