import streamlit as st
import matplotlib.pyplot as plt
import re

st.set_page_config(page_title="Analisador de Experimentos", layout="wide")

st.title("🔬 Analisador de Experimentos")
st.markdown("---")

# Upload de MÚLTIPLAS PASTAS (sem limite)
st.subheader("📂 Passo 1: Suba os arquivos das suas pastas")
st.info("Você pode selecionar arquivos de várias pastas ao mesmo tempo! Segure Ctrl e clique em cada pasta para selecionar todas.")

uploaded_files = st.file_uploader(
    "Arraste ou selecione TODOS os arquivos .txt aqui (de todas as pastas)",
    type="txt",
    accept_multiple_files=True,
    help="Dica: Abra cada pasta, aperte Ctrl+A e arraste para cá. Repita para cada pasta."
)

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
    st.success(f"✅ {len(uploaded_files)} arquivos carregados com sucesso!")
    
    # Organizar arquivos
    arquivos_dict = {f.name: f for f in uploaded_files}
    nomes = sorted(arquivos_dict.keys())
    
    # Encontrar pares automaticamente
    pares = []
    for nome in nomes:
        numero_match = re.search(r'(\d+)', nome)
        if numero_match:
            num = int(numero_match.group(1))
            if num % 2 != 0:
                par_nome = nome.replace(str(num), str(num + 1))
                if par_nome in arquivos_dict:
                    pares.append((nome, par_nome))
    
    if pares:
        st.markdown("---")
        st.subheader("📊 Passo 2: Escolha o gráfico que quer ver")
        
        # BOTÕES para cada par (resolve problema 1)
        cols = st.columns(min(len(pares), 4))
        
        if "par_selecionado" not in st.session_state:
            st.session_state.par_selecionado = f"{pares[0][0]} x {pares[0][1]}"
        
        for i, (px, py) in enumerate(pares):
            with cols[i % 4]:
                label = f"{px}\n x\n{py}"
                if st.button(f"📈 {px} x {py}", key=f"btn_{i}", use_container_width=True):
                    st.session_state.par_selecionado = f"{px} x {py}"
        
        st.markdown("---")
        
        # Gerar gráfico do par selecionado
        escolha = st.session_state.par_selecionado
        arq_x_nome, arq_y_nome = escolha.split(" x ")
        
        with st.spinner(f'Gerando gráfico: {escolha}...'):
            xs = ler_dados_txt(arquivos_dict[arq_x_nome])
            ys = ler_dados_txt(arquivos_dict[arq_y_nome])
            
            n = min(len(xs), len(ys))
            if n > 0:
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
                
                # Botão de download
                plt.savefig("resultado.png", dpi=300, bbox_inches='tight')
                with open("resultado.png", "rb") as img:
                    st.download_button(
                        label="💾 Baixar Gráfico (PNG)",
                        data=img,
                        file_name=f"grafico_{escolha}.png",
                        mime="image/png"
                    )
    else:
        st.warning("⚠️ Nenhum par encontrado. Verifique se os arquivos seguem o padrão (ex: 11.txt e 12.txt).")
else:
    st.info("👆 Suba os arquivos acima para começar.")

st.markdown("---")
st.caption("Plataforma de análise de experimentos de laboratório.")
