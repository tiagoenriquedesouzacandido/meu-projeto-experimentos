import os
import streamlit as st
import matplotlib.pyplot as plt

# PASTA ONDE ESTÃO SEUS EXPERIMENTOS
PASTA_RAIZ = r"C:\Users\soufi\Desktop\experimentos"

st.set_page_config(page_title="Análise de Experimentos", layout="wide")

st.title("🔬 Painel de Controle de Experimentos")
st.markdown("---")

if not os.path.exists(PASTA_RAIZ):
    st.error(f"⚠️ Erro: A pasta {PASTA_RAIZ} não foi encontrada!")
else:
    pastas = [d for d in os.listdir(PASTA_RAIZ) if os.path.isdir(os.path.join(PASTA_RAIZ, d))]
    
    col1, col2 = st.columns(2)
    
    with col1:
        exp_selecionado = st.selectbox("📁 Escolha o Experimento:", ["Selecione..."] + pastas)

    if exp_selecionado != "Selecione...":
        caminho_exp = os.path.join(PASTA_RAIZ, exp_selecionado)
        
        arquivos = sorted([f for f in os.listdir(caminho_exp) if f.endswith(".txt")])
        pares = []
        for arq in arquivos:
            nome = arq.replace(".txt", "")
            if nome.isdigit() and int(nome) % 2 != 0:
                par = f"{int(nome) + 1}.txt"
                if par in arquivos:
                    pares.append((arq, par))
        
        with col2:
            par_selecionado = st.selectbox("📊 Escolha o Par de Dados:", [f"{p[0]} x {p[1]}" for p in pares])

        if par_selecionado:
            arq_x, arq_y = par_selecionado.split(" x ")
            
            def ler_dados(caminho):
                vals = []
                lendo = False
                with open(caminho, "r", encoding="utf-8") as f:
                    for linha in f:
                        linha = linha.strip()
                        if linha == "[DATA]": lendo = True
                        elif linha == "[ENDDATA]": break
                        elif lendo and linha:
                            try: vals.append(float(linha))
                            except: pass
                return vals

            xs = ler_dados(os.path.join(caminho_exp, arq_x))
            ys = ler_dados(os.path.join(caminho_exp, arq_y))
            
            n = min(len(xs), len(ys))
            pontos = list(dict.fromkeys(zip(xs[:n], ys[:n]))) 
            
            if pontos:
                xs_f, ys_f = zip(*pontos)
                st.subheader(f"Visualizando: {exp_selecionado} ({par_selecionado})")
                
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(xs_f, ys_f, color='#1f77b4', linewidth=1)
                ax.set_xlabel("Canal X (mV)")
                ax.set_ylabel("Canal Y (mV)")
                ax.grid(True, linestyle='--', alpha=0.6)
                
                st.pyplot(fig)
                st.info(f"Total de pontos após filtro: {len(xs_f)}")