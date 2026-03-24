import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Analisador de Experimentos", layout="wide")

st.title("🔬 Analisador de Experimentos Online")
st.markdown("Faça upload dos arquivos X e Y para gerar o gráfico.")

col1, col2 = st.columns(2)

with col1:
    file_x = st.file_uploader("📤 Arquivo X (ex: 11.txt)", type="txt")

with col2:
    file_y = st.file_uploader("📤 Arquivo Y (ex: 12.txt)", type="txt")

def ler_dados(file):
    valores = []
    lendo = False
    
    conteudo = file.getvalue().decode("utf-8")
    
    for linha in conteudo.splitlines():
        linha = linha.strip()
        if linha == "[DATA]":
            lendo = True
        elif linha == "[ENDDATA]":
            break
        elif lendo and linha:
            try:
                valores.append(float(linha))
            except:
                pass
                
    return valores

if file_x and file_y:
    xs = ler_dados(file_x)
    ys = ler_dados(file_y)

    n = min(len(xs), len(ys))

    if n > 0:
        pontos = list(dict.fromkeys(zip(xs[:n], ys[:n])))

        if pontos:
            xs_f, ys_f = zip(*pontos)

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(xs_f, ys_f)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.grid(True)

            st.pyplot(fig)
            st.success(f"{len(xs_f)} pontos exibidos")

    else:
        st.warning("Arquivos sem dados válidos")
else:
    st.info("Aguardando upload dos dois arquivos...")
