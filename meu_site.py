import streamlit as st
import matplotlib.pyplot as plt
import re
from collections import defaultdict

st.set_page_config(page_title="Analisador de Experimentos", layout="wide")

st.title("🔬 Analisador de Experimentos")
st.markdown("---")

st.subheader("📂 Upload dos arquivos")
uploaded_files = st.file_uploader(
    "Selecione todos os arquivos .txt",
    type="txt",
    accept_multiple_files=True
)

def ler_dados_txt(file):
    valores = []
    lendo = False
    conteudo = file.getvalue().decode("utf-8", errors="ignore")
    for linha in conteudo.splitlines():
        linha = linha.strip()
        if linha == "[DATA]":
            lendo = True
            continue
        if linha == "[ENDDATA]":
            break
        if lendo and linha:
            try:
                valores.append(float(linha))
            except:
                pass
    return valores

def extrair_numero_base(nome):
    """
    Pega o número do arquivo.
    Ex:
    11.txt -> 11
    12.txt -> 12
    """
    m = re.fullmatch(r"(\d+)\.txt", nome.strip(), re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None

if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} arquivos carregados")

    # Evita conflito de nomes repetidos
    arquivos_por_nome = {}
    nomes_repetidos = []

    for f in uploaded_files:
        if f.name in arquivos_por_nome:
            nomes_repetidos.append(f.name)
        arquivos_por_nome[f.name] = f

    if nomes_repetidos:
        st.warning("⚠️ Existem arquivos com nomes repetidos. Alguns podem ter sido sobrescritos:")
        st.write(sorted(set(nomes_repetidos)))

    # Separar arquivos válidos
    numeros_disponiveis = {}
    arquivos_invalidos = []

    for nome, f in arquivos_por_nome.items():
        numero = extrair_numero_base(nome)
        if numero is None:
            arquivos_invalidos.append(nome)
        else:
            numeros_disponiveis[numero] = f

    # Encontrar pares: ímpar -> próximo par
    pares = []
    arquivos_sem_par = []

    for numero in sorted(numeros_disponiveis.keys()):
        if numero % 2 != 0:  # ímpar
            if (numero + 1) in numeros_disponiveis:
                pares.append((numero, numero + 1))
            else:
                arquivos_sem_par.append(f"{numero}.txt")

    st.info(f"📌 Pares encontrados: {len(pares)}")
    st.info(f"📌 Arquivos válidos identificados: {len(numeros_disponiveis)}")

    with st.expander("Ver diagnóstico dos arquivos"):
        st.write("**Arquivos válidos identificados:**")
        st.write(sorted([f"{n}.txt" for n in numeros_disponiveis.keys()]))

        if arquivos_invalidos:
            st.write("**Arquivos ignorados por nome inválido:**")
            st.write(sorted(arquivos_invalidos))

        if arquivos_sem_par:
            st.write("**Arquivos sem par correspondente:**")
            st.write(sorted(arquivos_sem_par))

    if pares:
        st.markdown("---")
        st.subheader("📊 Escolha o gráfico")

        opcoes = [f"{x}.txt x {y}.txt" for x, y in pares]
        escolha = st.selectbox("Selecione um par:", opcoes)

        arq_x_num = int(escolha.split(".txt x ")[0])
        arq_y_num = int(escolha.split(" x ")[1].replace(".txt", ""))

        file_x = numeros_disponiveis[arq_x_num]
        file_y = numeros_disponiveis[arq_y_num]

        xs = ler_dados_txt(file_x)
        ys = ler_dados_txt(file_y)

        n = min(len(xs), len(ys))

        if n > 0:
            pontos = list(dict.fromkeys(zip(xs[:n], ys[:n])))

            if pontos:
                xs_f, ys_f = zip(*pontos)

                st.subheader(f"📈 Gráfico: {arq_x_num}.txt x {arq_y_num}.txt")
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(xs_f, ys_f, color="#1f77b4", linewidth=1.5)
                ax.set_xlabel("Canal X (mV)")
                ax.set_ylabel("Canal Y (mV)")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

                st.success(f"Exibindo {len(xs_f)} pontos únicos.")
            else:
                st.warning("Não há pontos válidos após remover duplicatas.")
        else:
            st.error("Os arquivos selecionados não possuem dados válidos.")
    else:
        st.warning("Nenhum par ímpar/par foi encontrado.")
else:
    st.info("Envie os arquivos .txt para começar.")
