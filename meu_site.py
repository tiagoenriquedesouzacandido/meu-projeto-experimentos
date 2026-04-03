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
