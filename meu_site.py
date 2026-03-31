import streamlit as st
import matplotlib.pyplot as plt

# Configuração visual do site
st.set_page_config(page_title="Analisador de Experimentos", layout="wide")

st.title("🔬 Painel de Análise de Experimentos")
st.markdown("---")

st.sidebar.header("Instruções")
st.sidebar.info("1. Selecione o arquivo X (ímpar)\n2. Selecione o arquivo Y (par)\n3. O gráfico aparecerá automaticamente.")

# Área de Upload
col1, col2 = st.columns(2)

with col1:
    file_x = st.file_uploader("📤 Arquivo Canal X (ex: 11.txt)", type="txt")
with col2:
    file_y = st.file_uploader("📤 Arquivo Canal Y (ex: 12.txt)", type="txt")

def ler_dados_txt(file):
    if file is None: return []
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

if file_x and file_y:
    with st.spinner('Processando dados e removendo duplicatas...'):
        xs = ler_dados_txt(file_x)
        ys = ler_dados_txt(file_y)
        
        n = min(len(xs), len(ys))
        if n > 0:
            # Filtro de pontos repetidos (Removendo ruído)
            pontos = list(dict.fromkeys(zip(xs[:n], ys[:n])))
            xs_f, ys_f = zip(*pontos)

            st.subheader(f"📊 Resultado: {file_x.name} vs {file_y.name}")
            
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(xs_f, ys_f, color='#1f77b4', linewidth=1.5, label="Curva de Experimento")
            ax.set_xlabel("Canal X (mV)")
            ax.set_ylabel("Canal Y (mV)")
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend()
            
            st.pyplot(fig)
            
            # Estatísticas rápidas
            st.success(f"Análise concluída! {len(xs_f)} pontos únicos processados.")
            
            # Botão de Download do Gráfico
            plt.savefig("resultado.png", dpi=300)
            with open("resultado.png", "rb") as img:
                st.download_button(
                    label="💾 Baixar Gráfico (PNG)",
                    data=img,
                    file_name=f"grafico_{file_x.name}.png",
                    mime="image/png"
                )
        else:
            st.error("Erro: Os arquivos não contêm dados válidos entre [DATA] e [ENDDATA].")
else:
    st.info("Aguardando o upload dos arquivos para iniciar a análise...")

st.markdown("---")
st.caption("Plataforma de análise rápida desenvolvida para laboratórios.")
