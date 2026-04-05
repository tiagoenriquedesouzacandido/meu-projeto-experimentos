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