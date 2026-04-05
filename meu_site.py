<<<<<<< HEAD
import os
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Analisador de Experimentos", layout="wide")
st.title("🔬 Analisador de Experimentos Online")

PASTA_RAIZ = "experimentos"

def ler_dados(file_or_path, is_path=False):
    valores = []
    lendo = False
    if is_path:
        linhas = open(file_or_path, "r", encoding="utf-8").readlines()
    else:
        linhas = file_or_path.getvalue().decode("utf-8").splitlines()
    for linha in linhas:
        linha = linha.strip()
        if linha == "[DATA]": lendo = True
        elif linha == "[ENDDATA]": break
        elif lendo and linha:
            try: valores.append(float(linha))
            except: pass
    return valores

def gerar_grafico(xs, ys, titulo=""):
    n = min(len(xs), len(ys))
    pontos = list(dict.fromkeys(zip(xs[:n], ys[:n])))
    if not pontos:
        st.warning("Nenhum dado válido encontrado.")
        return
    xs_f, ys_f = zip(*pontos)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(xs_f, ys_f, color='#1f77b4', linewidth=1)
    ax.set_xlabel("Canal X (mV)")
    ax.set_ylabel("Canal Y (mV)")
    ax.grid(True, alpha=0.3)
    if titulo: ax.set_title(titulo)
    st.pyplot(fig)
    st.success(f"{len(xs_f)} pontos exibidos.")

# MODO AUTOMÁTICO (Rodando no seu PC)
if os.path.exists(PASTA_RAIZ):
    st.info("✅ Modo Local: Lendo pastas do seu computador.")
    
    pastas = [d for d in os.listdir(PASTA_RAIZ) if os.path.isdir(os.path.join(PASTA_RAIZ, d))]
    col1, col2 = st.columns(2)
    
    with col1:
        exp = st.selectbox("📁 Experimento:", ["Selecione..."] + pastas)
    
    if exp != "Selecione...":
        caminho_exp = os.path.join(PASTA_RAIZ, exp)
        arquivos = sorted([f for f in os.listdir(caminho_exp) if f.endswith(".txt")])
        pares = []
        for arq in arquivos:
            nome = arq.replace(".txt", "")
            if nome.isdigit() and int(nome) % 2 != 0:
                par = f"{int(nome)+1}.txt"
                if par in arquivos:
                    pares.append((arq, par))
        
        with col2:
            par = st.selectbox("📊 Par de arquivos:", [f"{x} x {y}" for x,y in pares])
        
        if par:
            arq_x, arq_y = par.split(" x ")
            xs = ler_dados(os.path.join(caminho_exp, arq_x), is_path=True)
            ys = ler_dados(os.path.join(caminho_exp, arq_y), is_path=True)
            gerar_grafico(xs, ys, titulo=f"{exp} | {par}")

# MODO UPLOAD (Rodando na nuvem / link público)'
else:
    st.info("☁️ Modo Online: Faça upload dos seus arquivos.")
    col1, col2 = st.columns(2)
    with col1:
        file_x = st.file_uploader("📤 Arquivo X (ex: 11.txt)", type="txt")
    with col2:
        file_y = st.file_uploader("📤 Arquivo Y (ex: 12.txt)", type="txt")
    
    if file_x and file_y:
        xs = ler_dados(file_x)
        ys = ler_dados(file_y)
        gerar_grafico(xs, ys, titulo=f"{file_x.name} vs {file_y.name}")
    else:
        st.info("Aguardando upload dos dois arquivos...")
=======
import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Análise de Experimentos", layout="wide")
st.title("📊 Visualizador de Experimentos")

# --- CONFIGURAÇÃO DE CAMINHO ---
# Ajuste para o nome da pasta que você subiu no GitHub
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

# 1. Verificar se a pasta existe
if not os.path.exists(PASTA_RAIZ):
    st.error(f"A pasta '{PASTA_RAIZ}' não foi encontrada no repositório. Verifique o nome no GitHub.")
else:
    # 2. Selecionar a Subpasta (Experimento)
    subpastas = [f for f in os.listdir(PASTA_RAIZ) if os.path.isdir(os.path.join(PASTA_RAIZ, f))]
    experimento_sel = st.sidebar.selectbox("Escolha o Experimento:", subpastas)

    if experimento_sel:
        caminho_exp = os.path.join(PASTA_RAIZ, experimento_sel)
        
        # 3. Identificar pares de arquivos (ex: 11 e 12, 21 e 22)
        arquivos = os.listdir(caminho_exp)
        prefixos = sorted(list(set([f[0] for f in arquivos if f.endswith(".txt") and len(f) >= 2])))
        
        escolha_par = st.sidebar.radio("Escolha o par de arquivos:", prefixos, format_func=lambda x: f"Arquivos {x}1.txt e {x}2.txt")

        arq_x = os.path.join(caminho_exp, f"{escolha_par}1.txt")
        arq_y = os.path.join(caminho_exp, f"{escolha_par}2.txt")

        if st.sidebar.button("Gerar Gráfico"):
            fs1, x = ler_txt_vallen(arq_x)
            fs2, y = ler_txt_vallen(arq_y)

            if x is not None and y is not None:
                st.success(f"Exibindo: {experimento_sel} - Par {escolha_par}")
                
                # --- CÁLCULOS ---
                fs = fs1 if fs1 else 1.0
                x_detrend = x - np.mean(x)
                y_detrend = y - np.mean(y)

                # --- PLOT ---
                fig, ax = plt.subplots(2, 1, figsize=(10, 8))
                
                ax[0].plot(x_detrend, label="Canal 1", color="blue")
                ax[0].set_title("Sinal no Tempo - Canal 1")
                ax[0].grid(True)

                ax[1].plot(y_detrend, label="Canal 2", color="green")
                ax[1].set_title("Sinal no Tempo - Canal 2")
                ax[1].grid(True)

                plt.tight_layout()
                st.pyplot(fig)
                
                # Botão para download dos dados processados (opcional)
                st.download_button("Baixar dados Canal 1 (CSV)", str(x_detrend), "canal1.csv")
            else:
                st.error("Arquivos não encontrados ou vazios.")
>>>>>>> da0033443619a32de3fa6756139b59a770eb2f06
