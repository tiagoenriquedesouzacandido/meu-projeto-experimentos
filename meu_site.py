import streamlit as st
import matplotlib.pyplot as plt
import re

st.set_page_config(page_title="Analisador de Pastas", layout="wide")

st.title("🔬 Analisador de Experimentos (Pasta Completa)")
st.markdown("Selecione todos os arquivos .txt da sua pasta de uma vez.")

# Botão de upload que aceita múltiplos arquivos
uploaded_files = st.file_uploader("📤 Arraste todos os arquivos .txt aqui", type="txt", accept_multiple_files=True)

def ler_dados_txt(file):
    valores = []
    lendo = False
    conteudo = file.getvalue().decode("utf-8")
    for linha in conteudo.splitlines():
        linha = linha.strip()
        if linha == "[DATA]": lendo = True
        elif linha == "[ENDDATA]": break
        elif lendo and linha:
            try: valores.append(float(linha))
            except: pass
    return valores

if uploaded_files:
    # Organizar os arquivos em um dicionário pelo nome
    arquivos_dict = {f.name: f for f in uploaded_files}
    nomes = sorted(arquivos_dict.keys())
    
    # Encontrar os pares (Ímpar x Par)
    pares = []
    for nome in nomes:
        numero_match = re.search(r'(\d+)', nome)
        if numero_match:
            num = int(numero_match.group(1))
            if num % 2 != 0: # Se for ímpar (X)
                par_nome = nome.replace(str(num), str(num + 1))
                if par_nome in arquivos_dict:
                    pares.append((nome, par_nome))

    if pares:
        st.sidebar.header("📊 Seleção de Gráfico")
        escolha = st.sidebar.selectbox("Escolha o par para visualizar:", [f"{p[0]} x {p[1]}" for p in pares])
        
        if escolha:
            arq_x_nome, arq_y_nome = escolha.split(" x ")
            
            with st.spinner(f'Gerando gráfico de {escolha}...'):
                xs = ler_dados_txt(arquivos_dict[arq_x_nome])
                ys = ler_dados_txt(arquivos_dict[arq_y_nome])
                
                n = min(len(xs), len(ys))
                if n > 0:
                    # Filtro de duplicatas
                    pontos = list(dict.fromkeys(zip(xs[:n], ys[:n])))
                    xs_f, ys_f = zip(*pontos)

                    st.subheader(f"📈 Gráfico: {escolha}")
                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.plot(xs_f, ys_f, color='#1f77b4', linewidth=1.5)
                    ax.set_xlabel("Canal X (mV)")
                    ax.set_ylabel("Canal Y (mV)")
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                    st.success(f"Exibindo {len(xs_f)} pontos únicos.")
                else:
                    st.error("Dados inválidos nos arquivos selecionados.")
    else:
        st.warning("Nenhum par de arquivos (Ímpar/Par) foi identificado. Verifique os nomes dos arquivos.")
else:
    st.info("💡 Dica: Entre na sua pasta de experimentos, aperte 'Ctrl + A' para selecionar todos os arquivos e arraste para cá.")

st.markdown("---")
st.caption("Análise em lote via navegador.")
