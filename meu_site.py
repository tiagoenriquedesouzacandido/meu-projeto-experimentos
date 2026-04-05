import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Analisador de Experimentos", layout="wide")
st.title("📊 Analisador de Experimentos Online")

# --- CONFIGURAÇÃO DE CAMINHO ---
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

if not os.path.exists(PASTA_RAIZ):
    st.error(f"A pasta '{PASTA_RAIZ}' não foi encontrada. Verifique no GitHub.")
else:
    subpastas = [f for f in os.listdir(PASTA_RAIZ) if os.path.isdir(os.path.join(PASTA_RAIZ, f))]
    experimento_sel = st.sidebar.selectbox("Escolha o Experimento:", subpastas)

    if experimento_sel:
        caminho_exp = os.path.join(PASTA_RAIZ, experimento_sel)
        arquivos = os.listdir(caminho_exp)
        prefixos = sorted(list(set([f[0] for f in arquivos if f.endswith(".txt") and len(f) >= 2])))
        escolha_par = st.sidebar.radio("Escolha o par:", prefixos, format_func=lambda x: f"Par {x}1 e {x}2")

        if st.sidebar.button("Gerar Gráfico"):
            fs1, x = ler_txt_vallen(os.path.join(caminho_exp, f"{escolha_par}1.txt"))
            fs2, y = ler_txt_vallen(os.path.join(caminho_exp, f"{escolha_par}2.txt"))

            if x is not None and y is not None:
                fig, ax = plt.subplots(2, 1, figsize=(10, 8))
                ax[0].plot(x - np.mean(x), color="blue")
                ax[0].set_title("Canal 1")
                ax[1].plot(y - np.mean(y), color="green")
                ax[1].set_title("Canal 2")
                st.pyplot(fig)
            else:
                st.error("Arquivos não encontrados.")