import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import pandas as pd

# --- Machine Learning (scikit-learn) ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

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
# ============================================================
def calcular_features(dados, fs, threshold_mv=0.5, pretrigger=0):
    n = len(dados)
    if n == 0:
        return {}

    dt = 1.0 / fs
    t = np.arange(n) * dt

    # --- FEATURES TEMPORAIS ---
    rms = np.sqrt(np.mean(dados**2))
    pico = np.max(np.abs(dados))
    energia = np.sum(dados**2) * dt
    duration_us = n * dt * 1e6

    # Rise Time: tempo do trigger (descontando pretrigger) até o pico
    idx_pico = np.argmax(np.abs(dados))
    idx_trigger = pretrigger if pretrigger > 0 else 0
    rise_samples = max(idx_pico - idx_trigger, 0)
    rise_time_us = rise_samples * dt * 1e6

    # Temporal Centroid: centro de massa temporal ponderado pela energia
    energia_t = dados**2
    if np.sum(energia_t) > 0:
        temporal_centroid_us = np.sum(t * energia_t) / np.sum(energia_t) * 1e6
    else:
        temporal_centroid_us = 0.0

    # Counts: cruzamentos ascendentes do threshold
    acima = dados > threshold_mv
    counts = int(np.sum(np.diff(acima.astype(int)) == 1))

    # Counts to Peak
    acima_ate_pico = dados[:idx_pico] > threshold_mv
    counts_to_peak = int(np.sum(np.diff(acima_ate_pico.astype(int)) == 1))

    # RA Value
    ra_value = rise_time_us / pico if pico > 0 else 0

    # Zero Crossings
    zc = int(np.sum(np.diff(np.sign(dados)) != 0))

    # --- FEATURES ESPECTRAIS (FFT) ---
    freqs = fftfreq(n, d=dt)
    espectro = np.abs(fft(dados))
    freqs_pos = freqs[:n // 2]
    espectro_pos = espectro[:n // 2]
    pot_espectro = espectro_pos**2
    pot_total = np.sum(pot_espectro)

    # Frequência de Pico
    idx_max_freq = np.argmax(espectro_pos)
    freq_pico = freqs_pos[idx_max_freq]

    # Frequência Centroide
    if np.sum(espectro_pos) > 0:
        freq_centroide = np.sum(freqs_pos * espectro_pos) / np.sum(espectro_pos)
    else:
        freq_centroide = 0

    # Average Frequency (ponderada pela potência)
    if pot_total > 0:
        avg_frequency = np.sum(freqs_pos * pot_espectro) / pot_total
    else:
        avg_frequency = 0

    # Spectral Spread
    if np.sum(espectro_pos) > 0:
        spectral_spread = np.sqrt(
            np.sum(((freqs_pos - freq_centroide)**2) * espectro_pos) / np.sum(espectro_pos)
        )
    else:
        spectral_spread = 0

    # Roll-on (15%) e Roll-off (85%) da energia espectral acumulada
    energia_acum = np.cumsum(pot_espectro)
    energia_total_acum = energia_acum[-1] if energia_acum[-1] > 0 else 1
    idx_rollon = np.searchsorted(energia_acum, 0.15 * energia_total_acum)
    freq_rollon = freqs_pos[idx_rollon] if idx_rollon < len(freqs_pos) else 0
    idx_rolloff = np.searchsorted(energia_acum, 0.85 * energia_total_acum)
    freq_rolloff = freqs_pos[idx_rolloff] if idx_rolloff < len(freqs_pos) else 0

    # ZCR
    zcr = zc / (n * dt)

    # Amplitude Não-Dimensional
    amp_nd = pico / rms if rms > 0 else 0

    # Partial Power (banda 1–5 kHz)
    banda_inf, banda_sup = 1000, 5000
    mask_banda = (freqs_pos >= banda_inf) & (freqs_pos <= banda_sup)
    pot_banda = np.sum(pot_espectro[mask_banda])
    partial_power = pot_banda / pot_total if pot_total > 0 else 0

    # Freq. Pico Ponderada
    freq_pico_pond = avg_frequency

    # Frequência de Iniciação (1º quarto) com interpolação parabólica
    quarto = max(n // 4, 16)
    trecho_inicio = dados[:quarto]
    esp_inicio = np.abs(fft(trecho_inicio))
    freq_inicio_arr = fftfreq(len(trecho_inicio), d=dt)
    mask_pos = freq_inicio_arr > 0
    esp_i_pos = esp_inicio[mask_pos]
    freq_i_pos = freq_inicio_arr[mask_pos]
    if len(esp_i_pos) > 2:
        idx_pk = np.argmax(esp_i_pos)
        if 0 < idx_pk < len(esp_i_pos) - 1:
            alpha = esp_i_pos[idx_pk - 1]
            beta = esp_i_pos[idx_pk]
            gamma = esp_i_pos[idx_pk + 1]
            denom = alpha - 2 * beta + gamma
            offset = 0.5 * (alpha - gamma) / denom if denom != 0 else 0
            freq_iniciacao = freq_i_pos[idx_pk] + offset * (freq_i_pos[1] - freq_i_pos[0])
        else:
            freq_iniciacao = freq_i_pos[idx_pk]
    else:
        freq_iniciacao = 0

    # Frequência de Reverberação (último quarto) com interpolação parabólica
    trecho_rev = dados[-quarto:]
    esp_rev = np.abs(fft(trecho_rev))
    freq_rev_arr = fftfreq(len(trecho_rev), d=dt)
    mask_pos_r = freq_rev_arr > 0
    esp_r_pos = esp_rev[mask_pos_r]
    freq_r_pos = freq_rev_arr[mask_pos_r]
    if len(esp_r_pos) > 2:
        idx_pk_r = np.argmax(esp_r_pos)
        if 0 < idx_pk_r < len(esp_r_pos) - 1:
            alpha = esp_r_pos[idx_pk_r - 1]
            beta = esp_r_pos[idx_pk_r]
            gamma = esp_r_pos[idx_pk_r + 1]
            denom = alpha - 2 * beta + gamma
            offset = 0.5 * (alpha - gamma) / denom if denom != 0 else 0
            freq_reverbacao = freq_r_pos[idx_pk_r] + offset * (freq_r_pos[1] - freq_r_pos[0])
        else:
            freq_reverbacao = freq_r_pos[idx_pk_r]
    else:
        freq_reverbacao = 0

    return {
        "RMS (mV)":                    round(rms, 4),
        "Pico (mV)":                   round(pico, 4),
        "Energia (mV²·s)":             round(energia, 6),
        "Duration (µs)":               round(duration_us, 2),
        "Rise Time (µs)":              round(rise_time_us, 2),
        "Temporal Centroid (µs)":      round(temporal_centroid_us, 2),
        "Counts":                      counts,
        "Counts to Peak":              counts_to_peak,
        "RA Value (µs/mV)":            round(ra_value, 4),
        "Zero Crossings":              zc,
        "Freq. Centroide (Hz)":        round(freq_centroide, 2),
        "Freq. Pico (Hz)":             round(freq_pico, 2),
        "Average Frequency (Hz)":      round(avg_frequency, 2),
        "Spectral Spread (Hz)":        round(spectral_spread, 2),
        "Roll-on Frequency (Hz)":      round(freq_rollon, 2),
        "Roll-off Frequency (Hz)":     round(freq_rolloff, 2),
        "ZCR (Hz)":                    round(zcr, 2),
        "Amp. Não-Dimensional":        round(amp_nd, 4),
        "Partial Power (banda)":       round(partial_power, 6),
        "Freq. Iniciação (Hz)":        round(freq_iniciacao, 2),
        "Freq. Reverberação (Hz)":     round(freq_reverbacao, 2),
        "Freq. Pico Ponderada (Hz)":   round(freq_pico_pond, 2),
    }


# ============================================================
# FUNÇÃO: CALCULAR AMORTECIMENTO LOGARÍTMICO
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
# FUNÇÃO: MONTA A TABELA DE FEATURES DE TODOS OS SINAIS (para o ML)
# Mesma lógica da Biblioteca de Sinais; fica em cache para não recalcular
# ============================================================
@st.cache_data(show_spinner="Calculando features de todos os sinais...")
def montar_dataset_ml(pasta_raiz, lista_materiais, threshold):
    registros = []
    for pasta in lista_materiais:
        pasta_mat = os.path.join(pasta_raiz, pasta)
        for arq in sorted(os.listdir(pasta_mat)):
            if not arq.endswith(".txt"):
                continue
            fs, dados, meta = ler_txt_vallen(os.path.join(pasta_mat, arq))
            if dados is None or len(dados) == 0:
                continue
            feat = calcular_features(dados, fs, threshold, meta.get("pretrigger", 0))
            feat["Amortecimento (ζ)"] = calcular_amortecimento(dados, fs)
            feat["Material"] = pasta
            feat["Arquivo"] = arq
            registros.append(feat)
    return pd.DataFrame(registros)


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
# SIDEBAR
# ============================================================
modo = st.sidebar.radio("Selecione o Modo:", [
    "🔬 Visão Detalhada",
    "🆚 Comparação",
    "📚 Biblioteca de Sinais",
    "🤖 Machine Learning"
])

st.sidebar.markdown("---")

material = st.sidebar.selectbox("Selecione o Experimento:", subpastas)
pasta_material = os.path.join(PASTA_RAIZ, material)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Parâmetros de Análise")

threshold_mv = st.sidebar.number_input(
    "Threshold (mV)", min_value=0.01, max_value=100.0, value=0.5, step=0.1
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔊 Velocidade de Propagação")

vel_propagacao = {}
for pasta in subpastas:
    vel_propagacao[pasta] = st.sidebar.number_input(
        f"Vel. {pasta} (m/s)",
        min_value=100,
        max_value=10000,
        value=2000,
        step=100
    )

st.sidebar.markdown("---")
if st.sidebar.button("🚪 Sair"):
    st.session_state["autenticado"] = False
    st.rerun()


# ============================================================
# MODO 1: VISÃO DETALHADA
# ============================================================
if modo == "🔬 Visão Detalhada":

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

    pares_ordenados = sorted(pares.keys(), key=lambda x: int(x) if x.isdigit() else x)
    par_sel = st.sidebar.selectbox("Selecione o Par:", pares_ordenados)
    par = pares[par_sel]

    arq_c1 = os.path.join(pasta_material, par.get("c1", ""))
    arq_c2 = os.path.join(pasta_material, par.get("c2", ""))
    fs1, dados1, meta1 = ler_txt_vallen(arq_c1)
    fs2, dados2, meta2 = ler_txt_vallen(arq_c2)

    st.subheader(f"📂 {material} — Par {par_sel}")

    # ToA
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

            t_eixo = np.arange(len(dados)) / fs * 1e6

            fig, axes = plt.subplots(2, 1, figsize=(6, 5))
            fig.patch.set_facecolor("white")

            # --- Sinal no tempo: mantém a unidade física (mV) ---
            axes[0].plot(t_eixo, dados, color="steelblue", linewidth=0.8)
            axes[0].axhline(threshold_mv, color="red", linestyle="--",
                            linewidth=0.8, label=f"Threshold ({threshold_mv} mV)")
            axes[0].axhline(-threshold_mv, color="red", linestyle="--", linewidth=0.8)
            axes[0].set_xlabel("Tempo (µs)")
            axes[0].set_ylabel("Amplitude (mV)")
            axes[0].set_title(f"Sinal no Tempo — {label}")
            axes[0].legend(fontsize=7)
            axes[0].grid(True, alpha=0.3)
            axes[0].set_facecolor("white")
            axes[0].tick_params(labelsize=9)

            # --- Espectro de frequências (FFT): magnitude não normalizada ---
            n = len(dados)
            freqs = fftfreq(n, d=1 / fs)
            espectro = np.abs(fft(dados))
            freqs_pos = freqs[:n // 2]
            espectro_pos = espectro[:n // 2]

            axes[1].plot(freqs_pos, espectro_pos, color="darkorange", linewidth=0.8)
            axes[1].set_xlabel("Frequência (Hz)")
            axes[1].set_ylabel("Magnitude da FFT (u.a.)")
            axes[1].set_title("Espectro de Frequências (FFT)")
            axes[1].grid(True, alpha=0.3)
            axes[1].set_facecolor("white")
            axes[1].tick_params(labelsize=9)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            pretrigger = meta.get("pretrigger", 0)
            features = calcular_features(dados, fs, threshold_mv, pretrigger)
            amort = calcular_amortecimento(dados, fs)
            features["Amortecimento (ζ)"] = amort

            df_feat = pd.DataFrame(features.items(), columns=["Parâmetro", "Valor"])
            st.dataframe(df_feat, use_container_width=True)


# ============================================================
# MODO 2: COMPARAÇÃO ENTRE MATERIAIS
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
                    pretrigger = meta.get("pretrigger", 0)
                    feat = calcular_features(dados, fs, threshold_mv, pretrigger)
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
            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")

            cores = plt.cm.tab10(np.linspace(0, 1, len(df_comp)))
            ax.bar(df_comp.index, df_comp[col_feat], color=cores, edgecolor="black")

            ax.set_title(f"{col_feat} — Média por Material")
            # O nome da coluna já traz a unidade: mV (RMS e Pico) e mV²·s (Energia)
            ax.set_ylabel(col_feat)
            ax.set_xlabel("Material")

            # Nomes dos materiais estavam sobrepostos: 30° alinhados à direita
            ax.set_xticks(range(len(df_comp.index)))
            ax.set_xticklabels(df_comp.index, rotation=30, ha="right", fontsize=9)

            ax.grid(True, axis="y", alpha=0.3)

            # Energia: notação científica no eixo Y, sem casas decimais fixas
            if col_feat == "Energia (mV²·s)":
                ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

            fig.tight_layout()
            st.pyplot(fig)
            plt.close()
    else:
        st.warning("Nenhum dado encontrado para comparação.")


# ============================================================
# MODO 3: BIBLIOTECA DE SINAIS
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
        arquivos_txt = sorted([f for f in os.listdir(pasta_mat) if f.endswith(".txt")])

        pares_bib = {}
        for arq in arquivos_txt:
            nome = os.path.splitext(arq)[0]
            if len(nome) >= 2:
                ultimo = nome[-1]
                chave = nome[:-1]
                pares_bib.setdefault(chave, {})
                if ultimo == "1":
                    pares_bib[chave]["c1"] = arq
                elif ultimo == "2":
                    pares_bib[chave]["c2"] = arq

        vel = vel_propagacao.get(pasta, 2000)

        for chave, par in pares_bib.items():
            for canal_key, label_canal in [("c1", 1), ("c2", 2)]:
                arq = par.get(canal_key)
                if not arq:
                    continue
                caminho = os.path.join(pasta_mat, arq)
                fs, dados, meta = ler_txt_vallen(caminho)
                if dados is None or len(dados) == 0:
                    continue

                pretrigger = meta.get("pretrigger", 0)
                feat = calcular_features(dados, fs, threshold_mv, pretrigger)
                feat["Amortecimento (ζ)"] = calcular_amortecimento(dados, fs)

                # Distância estimada do sensor via ToA do par: d = v × Δt / 2
                dist_sensor = None
                outro_key = "c2" if canal_key == "c1" else "c1"
                outro_arq = par.get(outro_key)
                if outro_arq:
                    _, _, meta_outro = ler_txt_vallen(os.path.join(pasta_mat, outro_arq))
                    t_self = parse_tempo(meta.get("time_raw", ""))
                    t_outro = parse_tempo(meta_outro.get("time_raw", ""))
                    if t_self is not None and t_outro is not None:
                        delta_t = abs(t_self - t_outro)
                        dist_sensor = round(delta_t * vel * 100 / 2, 4)  # cm

                feat["Material"] = pasta
                feat["Arquivo"] = arq
                feat["Canal"] = label_canal
                feat["Tempo (raw)"] = meta.get("time_raw", "?")
                feat["fs (Hz)"] = fs
                feat["Dist. Estimada (cm)"] = dist_sensor if dist_sensor is not None else "N/A"
                registros.append(feat)

    if registros:
        df_lib = pd.DataFrame(registros)

        cols_info = ["Material", "Arquivo", "Canal", "Tempo (raw)", "fs (Hz)", "Dist. Estimada (cm)"]
        cols_feat = [c for c in df_lib.columns if c not in cols_info]
        df_lib = df_lib[cols_info + cols_feat]

        st.dataframe(df_lib, use_container_width=True)

        st.markdown("### 📈 Estatísticas por Material")
        cols_num = [c for c in cols_feat if df_lib[c].dtype != object]
        st.dataframe(
            df_lib.groupby("Material")[cols_num].agg(["mean", "std"]).round(4),
            use_container_width=True
        )

        csv = df_lib.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Exportar Dataset (CSV)",
            data=csv,
            file_name="biblioteca_sinais_AE.csv",
            mime="text/csv"
        )

        st.success(f"✅ {len(registros)} sinais carregados na biblioteca.")


# ============================================================
# MODO 4: MACHINE LEARNING
# ============================================================
elif modo == "🤖 Machine Learning":
    st.subheader("🤖 Machine Learning — Classificação de Materiais (Prova de Conceito)")
    st.markdown(
        "Prova de conceito: os modelos tentam identificar o **material** de origem de cada sinal "
        "a partir dos parâmetros de EA extraídos. Validação cruzada **agrupada e estratificada**: "
        "os dois canais de uma mesma aquisição nunca ficam um no treino e outro no teste."
    )

    if st.button("Treinar e Avaliar Modelos de ML"):
        df_ml = montar_dataset_ml(PASTA_RAIZ, subpastas, threshold_mv)

        # --- Colunas removidas (não são parâmetros de EA) ---
        cols_remover = ["Material", "Arquivo", "Canal", "Tempo (raw)", "fs (Hz)",
                        "Duração (µs)", "Duration (µs)", "Dist. Estimada (cm)"]
        y = df_ml["Material"]
        X = df_ml.drop(columns=[c for c in cols_remover if c in df_ml.columns])
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan)   # infinitos viram nulos → mediana no Pipeline

        # --- Grupos: sinais da mesma aquisição (par canal 1 + canal 2) ficam juntos ---
        # Grupo = "Material|nome do arquivo sem o último dígito (1 ou 2)"
        # (mesma regra de pares que o app já usa nos outros modos)
        def chave_aquisicao(material_, arquivo_):
            nome = os.path.splitext(arquivo_)[0]
            if len(nome) >= 2 and nome[-1] in ("1", "2"):
                base = nome[:-1]
            else:
                base = nome
            return f"{material_}|{base}"

        grupos = np.array([chave_aquisicao(m, a)
                           for m, a in zip(df_ml["Material"], df_ml["Arquivo"])])

        contagem = y.value_counts()
        grupos_por_material = pd.Series(grupos).groupby(y.values).nunique()

        if y.nunique() < 2:
            st.error("É preciso ter pelo menos 2 materiais para classificar.")
            st.stop()

        # nº de divisões limitado pelo material com menos grupos (aquisições)
        n_splits = int(min(5, grupos_por_material.min()))
        if n_splits < 2:
            st.error("Algum material tem só 1 aquisição (grupo); a validação agrupada "
                     f"não é possível. Grupos por material:\n{grupos_por_material.to_string()}")
            st.stop()
        if n_splits < 5:
            st.warning(f"Há material com apenas {grupos_por_material.min()} aquisições; "
                       f"usando {n_splits} divisões em vez de 5.")

        st.info(f"📊 {len(X)} sinais | {len(set(grupos))} aquisições (grupos) | "
                f"{X.shape[1]} atributos | {y.nunique()} materiais | "
                f"{n_splits} divisões na validação")
        st.dataframe(pd.DataFrame({"Nº de sinais": contagem,
                                   "Nº de aquisições": grupos_por_material}),
                     use_container_width=True)

        # --- Modelos (imputação pela mediana + StandardScaler dentro do Pipeline,
        #     assim o pré-processamento é ajustado só com dados de treino em cada divisão) ---
        modelos = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "SVM (RBF)":     SVC(kernel="rbf", C=1.0),
            "k-NN (k=5)":    KNeighborsClassifier(n_neighbors=5),
        }

        def criar_pipe(modelo):
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("modelo", modelo),
            ])

        # Validação agrupada e estratificada (nenhum grupo aparece em treino e teste ao mesmo tempo)
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Confere se cada divisão é válida (todas as classes no treino; avisa se faltar no teste)
        todas = set(y.unique())
        for k, (tr, te) in enumerate(cv.split(X, y, groups=grupos), 1):
            if set(y.iloc[tr].unique()) != todas:
                st.error(f"Divisão {k}: treino sem algum material. "
                         "Não há grupos suficientes para essa validação.")
                st.stop()
            ausentes = todas - set(y.iloc[te].unique())
            if ausentes:
                st.warning(f"Divisão {k}: materiais ausentes no teste: {sorted(ausentes)}. "
                           "O F1 macro dessa divisão fica menos confiável.")

        # --- Validação única: cada modelo é validado UMA vez (predições out-of-fold).
        #     A tabela de métricas e a matriz de confusão do Random Forest saem daqui. ---
        resultados = []
        predicoes_cv = {}
        for nome, modelo in modelos.items():
            y_pred_cv = cross_val_predict(criar_pipe(modelo), X, y, cv=cv, groups=grupos)
            predicoes_cv[nome] = y_pred_cv

            accs, f1s = [], []
            for tr, te in cv.split(X, y, groups=grupos):
                accs.append(accuracy_score(y.iloc[te], y_pred_cv[te]))
                f1s.append(f1_score(y.iloc[te], y_pred_cv[te], average="macro"))

            accs = np.array(accs)
            resultados.append({
                "Modelo": nome,
                "Acurácia Média": round(accs.mean(), 4),
                "Desvio Padrão (±)": round(accs.std(), 4),
                "F1-Score Macro": round(np.mean(f1s), 4),
            })

        st.markdown("### 📋 Desempenho dos Modelos (validação cruzada agrupada)")
        df_res = pd.DataFrame(resultados)
        st.dataframe(df_res, use_container_width=True)
        for r in resultados:
            st.write(f"**{r['Modelo']}** — Acurácia: {r['Acurácia Média']*100:.1f}% "
                     f"(± {r['Desvio Padrão (±)']*100:.1f}%) | F1 macro: {r['F1-Score Macro']:.3f}")

        # --- Gráfico A: Top 10 importâncias (Random Forest treinado com todos os dados;
        #     gráfico descritivo, não entra em nenhuma métrica) ---
        pipe_rf = criar_pipe(RandomForestClassifier(n_estimators=100, random_state=42))
        pipe_rf.fit(X, y)
        importancias = pd.Series(
            pipe_rf.named_steps["modelo"].feature_importances_, index=X.columns
        ).sort_values(ascending=False).head(10)[::-1]

        st.markdown("### 🌲 Top 10 Atributos Mais Importantes (Random Forest)")
        fig1, ax1 = plt.subplots(figsize=(8, 4.5))
        ax1.barh(importancias.index, importancias.values, color="steelblue", edgecolor="black")
        ax1.set_xlabel("Importância (Gini)")
        ax1.set_title("Top 10 Atributos — Random Forest")
        ax1.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close()

        # --- Gráfico B: Matriz de confusão (predições out-of-fold já calculadas acima) ---
        y_pred = predicoes_cv["Random Forest"]
        classes = sorted(y.unique())
        cm = confusion_matrix(y, y_pred, labels=classes)

        st.markdown("### 🔥 Matriz de Confusão — Random Forest")
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        im = ax2.imshow(cm, cmap="Blues")
        ax2.set_xticks(range(len(classes)))
        ax2.set_yticks(range(len(classes)))
        ax2.set_xticklabels(classes, rotation=45, ha="right")
        ax2.set_yticklabels(classes)
        ax2.set_xlabel("Material previsto")
        ax2.set_ylabel("Material real")
        ax2.set_title("Matriz de Confusão (validação cruzada agrupada)")
        limite = cm.max() / 2 if cm.max() > 0 else 0
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax2.text(j, i, cm[i, j], ha="center", va="center",
                         color="white" if cm[i, j] > limite else "black")
        fig2.colorbar(im, ax=ax2)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        df_cm = pd.DataFrame(cm, index=classes, columns=classes)
        df_cm.index.name = "Real \\ Previsto"
        st.download_button(
            label="⬇️ Exportar matriz de confusão (CSV)",
            data=df_cm.to_csv().encode("utf-8"),
            file_name="matriz_confusao_random_forest.csv",
            mime="text/csv",
        )
    else:
        st.caption("Clique no botão acima para treinar e avaliar os modelos.")