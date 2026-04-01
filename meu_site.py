import streamlit as st
import matplotlib.pyplot as plt
import re
import zipfile
import io

st.set_page_config(page_title="Analisador de Experimentos", layout="wide")
st.title("🔬 Analisador de Experimentos")
st.markdown("---")

st.subheader("📂 Upload da Pasta (ZIP)")
st.info("Compacte sua pasta de experimentos em ZIP e suba aqui. Sem limite de arquivos!")

uploaded_zip = st.file_uploader("📤 Suba o arquivo .zip da pasta", type="zip")

def ler_dados_bytes(conteudo_bytes):
    valores = []
    lendo = False
    for linha in conteudo_bytes.decode("utf-8", errors="ignore").splitlines():
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

if uploaded_zip:
    with zipfile.ZipFile(io.BytesIO(uploaded_zip.read())) as z:
        # Listar todos os .txt dentro do ZIP
        todos_arquivos = [f for f in z.namelist() if f.endswith(".txt")]
        
        st.success(f"✅ {len(todos_arquivos)} arquivos .txt encontrados no ZIP!")

        # Organizar por número
        numeros_disponiveis = {}
        arquivos_invalidos = []

        for caminho in todos_arquivos:
            nome = caminho.split("/")[-1]  # pega só o nome do arquivo
            m = re.fullmatch(r"(\d+)\.txt", nome.strip(), re.IGNORECASE)
            if m:
                numero = int(m.group(1))
                numeros_disponiveis[numero] = caminho
            else:
                arquivos_invalidos.append(nome)

        # Encontrar pares
        pares = []
        sem_par = []

        for numero in sorted(numeros_disponiveis.keys()):
            if numero % 2 != 0:
                if (numero + 1) in numeros_disponiveis:
                    pares.append((numero, numero + 1))
                else:
                    sem_par.append(f"{numero}.txt")

        st.info(f"📌 {len(pares)} pares encontrados")

        with st.expander("🔍 Ver diagnóstico"):
            st.write("**Arquivos válidos:**", sorted([f"{n}.txt" for n in numeros_disponiveis.keys()]))
            if arquivos_invalidos:
                st.write("**Ignorados:**", sorted(arquivos_invalidos))
            if sem_par:
                st.write("**Sem par:**", sorted(sem_par))

        if pares:
            st.markdown("---")
            st.subheader("📊 Escolha o gráfico")

            opcoes = [f"{x}.txt x {y}.txt" for x, y in pares]
            escolha = st.selectbox("Selecione um par:", opcoes)

            arq_x_num = int(escolha.split(".txt")[0])
            arq_y_num = arq_x_num + 1

            with z.open(numeros_disponiveis[arq_x_num]) as fx:
                xs = ler_dados_bytes(fx.read())
            with z.open(numeros_disponiveis[arq_y_num]) as fy:
                ys = ler_dados_bytes(fy.read())

            n = min(len(xs), len(ys))

            if n > 0:
                pontos = list(dict.fromkeys(zip(xs[:n], ys[:n])))
                if pontos:
                    xs_f, ys_f = zip(*pontos)

                    st.subheader(f"📈 {escolha}")
                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.plot(xs_f, ys_f, color="#1f77b4", linewidth=1.5)
                    ax.set_xlabel("Canal X (mV)")
                    ax.set_ylabel("Canal Y (mV)")
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)

                    st.success(f"✅ {len(xs_f)} pontos únicos exibidos.")

                    plt.savefig("resultado.png", dpi=300, bbox_inches="tight")
                    with open("resultado.png", "rb") as img:
                        st.download_button(
                            label="💾 Baixar Gráfico (PNG)",
                            data=img,
                            file_name=f"grafico_{escolha}.png",
                            mime="image/png"
                        )
        else:
            st.warning("Nenhum par encontrado no ZIP.")
else:
    st.info("👆 Suba o arquivo ZIP para começar.")

st.markdown("---")
st.caption("Plataforma de análise de experimentos de laboratório.")
