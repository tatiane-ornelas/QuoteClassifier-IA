import gradio as gr
import pandas as pd
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

from main_controller import MainController
from assistente_classificador import chain_assistente

# Inicializa controlador principal
controller = MainController()

# === Interface principal ===
with gr.Blocks() as app:
    df_state = gr.State()

    with gr.Row():
        # === Coluna lateral: Chat de Ajuda ===
        with gr.Group(visible=False) as chat_coluna_lateral:
            with gr.Column(scale=1):
                gr.Markdown("## ℹ️ Ajuda e Suporte")
                chatbox_ajuda = gr.Chatbot(label="Assistente de Ajuda", height=300, type="messages")
                chat_input_usuario = gr.Textbox(
                    placeholder="Digite sua dúvida e pressione Enter", 
                    show_label=False
                )
                botao_chat_fechar = gr.Button("❌ Fechar Chat")

        # === Coluna principal ===
        with gr.Column(scale=4):
            gr.Markdown("## 🧠 QuoteClassifier-AI: Assistente Inteligente para Suporte à Análise Qualitativa")

            botao_chat_abrir = gr.Button("❓ Precisa de ajuda?", visible=True)
            botao_ir_etapa5_avaliacao = gr.Button("📥 Já tenho classificações. Quero avaliar!", visible=True)

            # === Etapa 1 – Escopo da Pesquisa ===
            with gr.Group() as et1_grupo:
                gr.Markdown("### Etapa 1 – Escopo da Pesquisa")
                et1_input_escopo = gr.Textbox(label="Digite o escopo da sua pesquisa")
                et1_output_escopo = gr.Markdown()
                et1_botao_salvar = gr.Button("Salvar escopo")


            # === Etapa 2 – Cadastro de Constructos ===
            with gr.Group(visible=False) as et2_grupo:
                gr.Markdown("### Etapa 2 – Cadastro de Constructos")

                et2_radio_metodo = gr.Radio(
                    ["Manual", "Upload de planilha"], 
                    label="Como deseja cadastrar os constructos?"
                )

                et2_input_qtd = gr.Number(
                    label="Quantos constructos deseja cadastrar?", 
                    precision=0, 
                    visible=False
                )

                et2_botao_confirmar_qtd = gr.Button("Confirmar quantidade", visible=False)

                MAX_CONSTRUCTOS = 10
                et2_campos_constructos = [
                    (
                        gr.Textbox(label=f"Nome do Constructo {i+1}", visible=False),
                        gr.Textbox(label=f"Definição do Constructo {i+1}", visible=False)
                    )
                    for i in range(MAX_CONSTRUCTOS)
                ]

                et2_botao_salvar = gr.Button("Salvar constructos", visible=False)

                et2_upload_planilha = gr.File(
                    label="Upload de planilha (.xlsx)", 
                    file_types=[".xlsx"], 
                    visible=False
                )

                et2_output_resumo = gr.Markdown()


            # === Etapa 3 – Upload da Planilha de Quotes ===
            with gr.Group(visible=False) as et3_grupo:
                gr.Markdown("### Etapa 3 – Upload da Planilha e Seleção de Colunas")

                et3_upload_planilha = gr.File(
                    label="Faça upload da planilha (.xlsx)", 
                    interactive=True
                )

                et3_dropdown_coluna_quote = gr.Dropdown(
                    label="Coluna com os QUOTES", 
                    visible=False
                )

                et3_dropdown_coluna_classificacao = gr.Dropdown(
                    label="Coluna para CLASSIFICAÇÃO", 
                    visible=False
                )

                et3_texto_status = gr.Markdown()


            # === Etapa 4 – Classificação com IA ===
            with gr.Group(visible=False) as et4_grupo:
                gr.Markdown("### Etapa 4 – Classificação")

                et4_dropdown_modelo = gr.Dropdown(
                    label="Modelo de classificação",
                    choices=[
                        "EmbeddingQuoteClassifier", "openai-3.5", "openai-4",
                        "HybridQuoteClassifier", "ConstructSimilarityClassifier"
                    ]
                )

                # Informações sobre few-shot learning
                et4_mensagem_exemplos = gr.Markdown(
                    "**ℹ️ Se desejar incluir exemplos anotados (few-shot learning)** "
                    "para ajudar os modelos baseados em linguagem natural, envie uma planilha com as colunas: "
                    "**quote**, **constructo** e **justificativa**.",
                    visible=False
                )

                # Upload de exemplos anotados (few-shot)
                et4_upload_exemplos = gr.File(
                    label="📄 Upload de exemplos anotados (opcional)", 
                    file_types=[".xlsx"],
                    visible=False
                )

                # Grupo para seleção das colunas dos exemplos
                with gr.Row(visible=False) as et4_grupo_exemplos:
                    et4_status_exemplos = gr.Markdown()

                et4_dropdown_exemplo_quote = gr.Dropdown(
                    label="Coluna do Quote", visible=False, interactive=True
                )
                et4_dropdown_exemplo_constructo = gr.Dropdown(
                    label="Coluna do Constructo", visible=False, interactive=True
                )
                et4_dropdown_exemplo_justificativa = gr.Dropdown(
                    label="Coluna da Justificativa", visible=False, interactive=True
                )

                et4_botao_confirmar_exemplos = gr.Button("✅ Confirmar exemplos", visible=False)

                et4_status_exemplos_msg = gr.Textbox(label="Status dos Exemplos", visible=False)

                et4_exemplos_formatados = gr.Textbox(
                    label="Exemplos Confirmados",
                    visible=False,
                    interactive=False,
                    lines=15
                )

                # Classificação
                et4_botao_classificar = gr.Button("Iniciar Classificação com IA")
                et4_texto_confirmacao = gr.Markdown(visible=False)

                with gr.Row(visible=False) as et4_botoes_confirmacao:
                    et4_botao_sim = gr.Button("✅ Sim")
                    et4_botao_nao = gr.Button("❌ Não")

                et4_status_classificacao = gr.Markdown()
                et4_arquivo_resultado = gr.File(label="Download da Planilha Classificada")
                et4_botao_interromper = gr.Button("❌ Interromper Classificação", visible=False)

            # === Etapa 5 – Avaliação dos Resultados ===
            with gr.Group(visible=False) as et5_grupo:
                gr.Markdown("### Etapa 5 – Avaliação dos Resultados")

                et5_upload_planilha = gr.File(
                    label="Upload da planilha classificada (.xlsx)", 
                    file_types=[".xlsx"]
                )

                et5_status_carregamento = gr.Markdown(visible=False)

                et5_dropdown_col_manual = gr.Dropdown(
                    label="Coluna com Classificação Manual", 
                    visible=False
                )

                et5_dropdown_col_automatica = gr.Dropdown(
                    label="Coluna com Classificação Automática", 
                    visible=False
                )

                et5_dropdown_col_resultado = gr.Dropdown(
                    label="Coluna para salvar Resultado (Certa/Errada)", 
                    visible=False
                )

                et5_input_nova_coluna = gr.Textbox(
                    label="Nome da nova coluna de resultado",
                    placeholder="Ex: Resultado",
                    visible=False
                )

                et5_botao_avaliar = gr.Button("📊 Avaliar Classificação", visible=False)

                et5_output_resultado = gr.Markdown()
                et5_download_planilha = gr.File(label="Download da Planilha Avaliada", visible=False)

                et5_botao_reiniciar = gr.Button("🔄 Iniciar nova análise")
                et5_download_pdf = gr.File(label="📄 Download do Relatório em PDF", visible=False)

#Funções

    # === Função auxiliar ===
    # Relacionada à Etapa 5 – Avaliação
    def mostrar_input_nova_coluna(p_valor_coluna_selecionada):
        """
        Exibe o campo de input para criação de nova coluna quando o usuário seleciona
        "Criar nova coluna..." no dropdown de colunas de resultado.
    
        Parâmetros:
            p_valor_coluna_selecionada (str): Valor selecionado no dropdown.
    
        Retorna:
            gr.update: Visibilidade do campo de input.
        """
        if p_valor_coluna_selecionada == "Criar nova coluna...":
            return gr.update(visible=True)
        return gr.update(visible=False)
    
    
    # === Função de navegação ===
    # Transita da tela inicial para a Etapa 5
    def ir_para_etapa5():
        """
        Oculta todas as etapas anteriores e exibe a Etapa 5 – Avaliação dos Resultados.
    
        Retorna:
            Tuple[gr.update]: Visibilidade de cada grupo de etapas.
        """
        return (
            gr.update(visible=False),  # et1_grupo
            gr.update(visible=False),  # et2_grupo
            gr.update(visible=False),  # et3_grupo
            gr.update(visible=False),  # et4_grupo
            gr.update(visible=True),   # et5_grupo
        )
    
    
    # === Etapa 5 – Avaliação dos Resultados ===
    def carregar_planilha_avaliacao(p_arquivo):
        """
        Lê o arquivo Excel carregado na Etapa 5 e extrai os nomes das colunas para
        configurar os dropdowns de seleção de classificação manual e automática.
    
        Parâmetros:
            p_arquivo (tempfile): Arquivo Excel (.xlsx) carregado pelo usuário.
    
        Retorna:
            Tuple[gr.update]: Atualizações de estado dos componentes da Etapa 5.
        """
        try:
            colunas = controller.carregar_planilha_quotes(p_arquivo.name)
            return (
                gr.update(choices=colunas, visible=True),  # et5_dropdown_col_manual
                gr.update(choices=colunas, visible=True),  # et5_dropdown_col_automatica
                gr.update(choices=colunas + ["Criar nova coluna..."], visible=True),  # et5_dropdown_col_resultado
                gr.update(visible=True),  # et5_botao_avaliar
                gr.update(visible=True),  # et5_grupo
                gr.update(value="✅ Planilha carregada! Agora selecione as colunas abaixo.", visible=True)
            )
        except Exception:
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(value="❌ Erro ao carregar a planilha. Verifique o formato.", visible=True)
            )
    
    
    # === Etapa 5 – Avaliação dos Resultados ===
    def avaliar_classificacao_fn(p_col_manual, p_col_automatica, p_col_resultado, p_nome_nova_coluna):
        """
        Avalia a classificação automática comparando com a classificação manual.
        Gera relatório com métricas de acerto e permite download do arquivo e do PDF.
    
        Parâmetros:
            p_col_manual (str): Nome da coluna com a classificação feita manualmente.
            p_col_automatica (str): Nome da coluna com a classificação feita pela IA.
            p_col_resultado (str): Nome da coluna onde será salvo o resultado (Certa/Errada).
            p_nome_nova_coluna (str): Nome personalizado para a nova coluna, se aplicável.
    
        Retorna:
            Tuple: Markdown com resultado, caminho do .xlsx avaliado, caminho do PDF.
        """
        try:
            if p_col_resultado == "Criar nova coluna...":
                p_col_resultado = p_nome_nova_coluna.strip() or "Resultado"
    
            markdown, caminho_arquivo, _, acertos, erros, caminho_pdf, relatorio_md = controller.avaliar_resultados(
                p_col_manual, p_col_automatica, p_col_resultado
            )
    
            markdown_completo = f"{markdown.strip()}\n\n{relatorio_md.strip()}"
    
            return (
                markdown_completo,
                gr.update(value=caminho_arquivo, visible=True),
                gr.update(value=caminho_pdf, visible=True)
            )
        except Exception as e:
            return (
                f"❌ Erro na avaliação: {e}",
                gr.update(visible=False),
                gr.update(visible=False)
            )


# === Chat de Ajuda (Coluna lateral) ===
    def abrir_chat():
        """
        Torna visível a coluna lateral de ajuda (chat com o assistente).
    
        Retorna:
            gr.update: Atualiza a visibilidade da coluna lateral para True.
        """
        return gr.update(visible=True)
    
    
    def fechar_chat():
        """
        Oculta a coluna lateral de ajuda e limpa o campo de entrada do chat.
    
        Retorna:
            Tuple[gr.update, list]: Atualiza a visibilidade para False e zera o histórico.
        """
        return gr.update(visible=False), []
    
    
    def responder_chat(p_msg_usuario, p_historico):
        """
        Gera uma resposta do assistente com base na pergunta do usuário.
    
        Parâmetros:
            p_msg_usuario (str): Mensagem digitada pelo usuário no chat.
            p_historico (list): Histórico de mensagens anteriores (como lista de tuplas ou dicts).
    
        Retorna:
            Tuple[list, str]: Histórico atualizado com a nova interação, e string vazia para limpar input.
        """
        resposta = chain_assistente.invoke({"pergunta": p_msg_usuario})
    
        # Converte histórico antigo (tuplas) para formato compatível
        historico_formatado = []
        for item in p_historico:
            if isinstance(item, tuple) and len(item) == 2:
                historico_formatado.append({"role": "user", "content": item[0]})
                historico_formatado.append({"role": "assistant", "content": item[1]})
            elif isinstance(item, dict) and "role" in item and "content" in item:
                historico_formatado.append(item)
    
        # Adiciona nova interação
        historico_formatado.append({"role": "user", "content": p_msg_usuario})
        historico_formatado.append({"role": "assistant", "content": resposta})
    
        return historico_formatado, ""

# === Etapa 1 – Escopo da Pesquisa ===
    def salvar_escopo_fn(p_texto_escopo):
        """
        Salva o escopo da pesquisa fornecido pelo usuário.
    
        Parâmetros:
            p_texto_escopo (str): Texto com o escopo da pesquisa.
    
        Retorna:
            Tuple[gr.update, gr.update]: Confirmação da operação e acionamento da Etapa 2.
        """
        controller.salvar_escopo(p_texto_escopo)
        return (
            gr.update(value=f"✅ Escopo salvo: {p_texto_escopo}"), 
            gr.update(visible=True)  # Exibe a Etapa 2
        )

# === Etapa 2 – Cadastro de Constructos ===
    def escolher_metodo(p_metodo):
        """
        Define a visibilidade dos campos com base na escolha entre cadastro manual ou upload de planilha.
    
        Parâmetros:
            p_metodo (str): Método escolhido pelo usuário ("Manual" ou "Upload de planilha").
    
        Retorna:
            List[gr.update]: Visibilidade dos campos de entrada e botões da Etapa 2.
        """
        vis_manual = p_metodo == "Manual"
        vis_upload = p_metodo == "Upload de planilha"
    
        updates = [gr.update(visible=vis_manual), gr.update(visible=vis_manual)]
        updates += [gr.update(visible=vis_manual) for _ in et2_campos_constructos for _ in range(2)]
        updates += [gr.update(visible=vis_manual), gr.update(visible=vis_upload)]
    
        return updates
    
    
    def mostrar_campos(p_qtd):
        """
        Exibe os campos de nome e definição dos constructos conforme a quantidade escolhida.
    
        Parâmetros:
            p_qtd (int): Quantidade de constructos que o usuário deseja cadastrar.
    
        Retorna:
            List[gr.update]: Visibilidade de campos até a quantidade informada.
        """
        updates = []
        for i, (campo_nome, campo_def) in enumerate(et2_campos_constructos):
            visivel = i < p_qtd
            updates.extend([gr.update(visible=visivel), gr.update(visible=visivel)])
        return updates + [gr.update(visible=True)]  # Exibe o botão salvar
    
    
    def processar_constructos_fn(*p_entradas):
        """
        Recebe os dados preenchidos manualmente nos campos de constructos e os processa.
    
        Parâmetros:
            *p_entradas: Lista intercalada com nomes e definições dos constructos.
    
        Retorna:
            Tuple: Resumo formatado, visibilidade da Etapa 3 e visibilidade do seletor de modelo.
        """
        resumo = controller.carregar_constructos_manualmente(p_entradas)
        return resumo, gr.update(visible=True), gr.update(visible=True)
    
    
    def carregar_constructos_de_planilha(p_arquivo):
        """
        Carrega os constructos a partir de uma planilha .xlsx enviada pelo usuário.
    
        Parâmetros:
            p_arquivo (tempfile): Arquivo Excel contendo as colunas de nome e definição.
    
        Retorna:
            Tuple: Resumo formatado, visibilidade da Etapa 3 e visibilidade do seletor de modelo.
        """
        try:
            resumo = controller.carregar_constructos_de_planilha(p_arquivo.name)
            return resumo, gr.update(visible=True), gr.update(visible=True)
        except Exception as e:
            return f"❌ Erro: {e}", gr.update(visible=False), gr.update(visible=False)

    # === Etapa 3 – Upload da Planilha de Quotes ===
    def carregar_planilha(p_arquivo):
        """
        Carrega a planilha de quotes fornecida pelo usuário, extrai as colunas disponíveis
        e armazena o DataFrame carregado no estado.
    
        Parâmetros:
            p_arquivo (tempfile): Arquivo Excel contendo os quotes.
    
        Retorna:
            Tuple: Atualização das colunas dropdown, mensagem de status, DataFrame carregado e visibilidade da Etapa 4.
        """
        if p_arquivo is None:
            return (
                gr.update(visible=False),  # et3_dropdown_coluna_quote
                gr.update(visible=False),  # et3_dropdown_coluna_classificacao
                gr.update(value="ℹ️ Nenhuma planilha carregada"),
                gr.update(value=None),
                gr.update(visible=False),  # et4_botao_classificar
                gr.update(visible=False),  # et4_grupo
            )
    
        try:
            colunas = controller.carregar_planilha_quotes(p_arquivo.name)
            df = controller.quote_loader.get_dataframe()
    
            return (
                gr.update(choices=colunas, visible=True),
                gr.update(choices=colunas, visible=True),
                gr.update(value="✅ Planilha carregada com sucesso"),
                gr.update(value=df),
                gr.update(visible=True),
                gr.update(visible=True)
            )
    
        except Exception as e:
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(value=f"❌ Erro ao carregar a planilha: {e}"),
                gr.update(value=None),
                gr.update(visible=False),
                gr.update(visible=False)
            )
    
    
    def habilitar_botao_classificar(p_col_quote, p_col_class, p_modelo):
        """
        Habilita o botão de classificação apenas se todos os campos obrigatórios estiverem preenchidos.
    
        Parâmetros:
            p_col_quote (str): Nome da coluna dos quotes.
            p_col_class (str): Nome da coluna para salvar classificação.
            p_modelo (str): Nome do modelo selecionado.
    
        Retorna:
            gr.update: Visibilidade do botão de classificação.
        """
        return gr.update(visible=bool(p_col_quote and p_col_class and p_modelo))

# === Etapa 4 – Classificação com IA ===
    def confirmar_modelo_fn(p_modelo_escolhido):
        """
        Mostra mensagem de confirmação da escolha do modelo e exibe os botões de confirmação.
    
        Parâmetros:
            p_modelo_escolhido (str): Nome do modelo selecionado.
    
        Retorna:
            Tuple[gr.update]: Mensagem de confirmação, visibilidade do botão de interrupção e botões de decisão.
        """
        return (
            gr.update(value=f"Você confirma a seleção do modelo: **{p_modelo_escolhido}**?", visible=True),
            gr.update(visible=True),  # et4_botao_interromper
            gr.update(visible=True)   # et4_botoes_confirmacao
        )
    
    
    def confirmar_e_classificar(p_confirmacao, p_df_state, p_col_quote, p_col_class, p_modelo):
        """
        Realiza a classificação dos quotes utilizando o modelo selecionado, considerando exemplos se houver.
    
        Parâmetros:
            p_confirmacao (str): "Sim" ou "Não", confirmação do usuário.
            p_df_state (DataFrame): Planilha carregada no estado.
            p_col_quote (str): Nome da coluna de quotes.
            p_col_class (str): Nome da coluna onde será salva a classificação.
            p_modelo (str): Nome do modelo selecionado.
    
        Retorna:
            Tuple: Mensagem de status, arquivo de saída, visibilidade das etapas e botões.
        """
        if p_confirmacao != "Sim":
            return (
                "❌ Cancelado. Você pode escolher outro modelo ou revisar os dados e clicar em Iniciar Classificação com IA novamente.",
                None,
                gr.update(visible=True),   # et4_grupo
                gr.update(visible=False),  # et5_grupo
                gr.update(visible=False),  # et4_botao_interromper
                gr.update(visible=False)   # et4_botoes_confirmacao
            )
    
        controller.resetar_interrupcao()
    
        exemplos = controller.exemplos_usuario
        suporta_exemplos = p_modelo in ["openai-3.5", "openai-4", "ConstructSimilarityClassifier"]
    
        status, arquivo_saida = controller.classificar(
            col_quote=p_col_quote,
            col_class=p_col_class,
            modelo=p_modelo,
            progress=None,
            deve_interromper=controller.verificar_interrupcao,
            exemplos=exemplos if suporta_exemplos else None
        )
    
        return (
            status,
            arquivo_saida,
            gr.update(visible=True),  # et4_grupo
            gr.update(visible=True),  # et5_grupo
            gr.update(visible=False),
            gr.update(visible=False)
        )
    
    
    def interromper_classificacao():
        """
        Solicita ao sistema a interrupção da classificação em andamento.
    
        Retorna:
            gr.update: Mensagem informando que a interrupção foi solicitada.
        """
        controller.solicitar_interrupcao()
        return gr.update(value="⚠️ Interrupção solicitada.")

# === Etapa 4 – Few-Shot Learning: Upload e Visibilidade ===
    def mostrar_upload_exemplos(p_modelo_escolhido):
        """
        Define a visibilidade dos campos de few-shot learning com base no modelo selecionado.
    
        Parâmetros:
            p_modelo_escolhido (str): Nome do modelo selecionado pelo usuário.
    
        Retorna:
            Tuple[gr.update]: Visibilidade dos componentes de upload e instrução de exemplos.
        """
        visivel = p_modelo_escolhido in ["openai-3.5", "openai-4", "HybridQuoteClassifier"]
        return (
            gr.update(visible=visivel),  # et4_upload_exemplos
            gr.update(visible=visivel),  # et4_grupo_exemplos
            gr.update(visible=visivel)   # et4_mensagem_exemplos
        )
    
    
    # === Etapa 4 – Few-Shot Learning: Carregamento da planilha de exemplos ===
    def carregar_colunas_planilha_exemplos(p_arquivo):
        """
        Carrega a planilha de exemplos e extrai os nomes das colunas para seleção pelo usuário.
    
        Parâmetros:
            p_arquivo (tempfile): Arquivo Excel com colunas quote, constructo e justificativa.
    
        Retorna:
            Tuple: Atualizações dos dropdowns e botão de confirmação.
        """
        try:
            df_exemplos = pd.read_excel(p_arquivo.name)
            colunas = df_exemplos.columns.tolist()
            controller._df_exemplos_temp = df_exemplos
    
            return (
                gr.update(choices=colunas, visible=True),  # quote
                gr.update(choices=colunas, visible=True),  # constructo
                gr.update(choices=colunas, visible=True),  # justificativa
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True)
            )
        except Exception:
            return (
                gr.update(choices=[], visible=False),
                gr.update(choices=[], visible=False),
                gr.update(choices=[], visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False)
            )
    
    
    # === Etapa 4 – Few-Shot Learning: Confirmação dos exemplos ===
    def confirmar_exemplos(p_col_quote, p_col_constructo, p_col_justificativa):
        """
        Valida e formata os exemplos anotados fornecidos pelo usuário.
    
        Parâmetros:
            p_col_quote (str): Nome da coluna com os quotes.
            p_col_constructo (str): Nome da coluna com os constructos.
            p_col_justificativa (str): Nome da coluna com a justificativa da escolha.
    
        Retorna:
            Tuple: Mensagem de confirmação, texto com os exemplos formatados e visibilidade.
        """
        df = controller._df_exemplos_temp.copy()
        exemplos_formatados = []
    
        exemplos = []
        for _, linha in df.iterrows():
            exemplo = {
                "quote": str(linha[p_col_quote]),
                "constructo": str(linha[p_col_constructo]),
                "justificativa": str(linha[p_col_justificativa])
            }
            exemplos.append(exemplo)
            exemplos_formatados.append(
                f"Exemplo {len(exemplos)}:\nQuote: {exemplo['quote']}\nConstructo: {exemplo['constructo']}\nJustificativa: {exemplo['justificativa']}\n"
            )
    
        controller.setar_exemplos_usuario(exemplos)
    
        return (
            f"✅ {len(exemplos)} exemplos confirmados com sucesso!",
            "\n".join(exemplos_formatados).strip(),
            gr.update(visible=True)
        )

# === Reinicialização Geral do Sistema ===
    def reiniciar_processo():
        """
        Limpa todos os estados e campos do sistema para permitir nova análise desde a Etapa 1.
    
        Retorna:
            Tuple[gr.update]: Reset de valores, visibilidade e campos das 5 etapas.
        """
        controller.resetar_interrupcao()
    
        updates_constructos = [gr.update(value="", visible=False) for pair in et2_campos_constructos for _ in pair]
    
        return (
            # Etapa 1
            gr.update(value=""),          # et1_input_escopo
            gr.update(value=""),          # et1_output_escopo
    
            # Etapa 2
            gr.update(value=None),        # et2_upload_planilha
    
            # Etapa 3
            gr.update(value=None),        # et3_upload_planilha
            gr.update(choices=[], value=None, visible=False),  # et3_dropdown_coluna_quote
            gr.update(choices=[], value=None, visible=False),  # et3_dropdown_coluna_classificacao
            gr.update(value=""),          # et3_texto_status
    
            # Etapa 4
            gr.update(value=None),        # et4_dropdown_modelo
            gr.update(value=""),          # et4_texto_confirmacao
            gr.update(value=""),          # et4_status_classificacao
            gr.update(value=None),        # et4_arquivo_resultado
            gr.update(visible=False),     # et4_botao_interromper
            gr.update(visible=False),     # et4_botoes_confirmacao
    
            # Etapa 5
            gr.update(value=None),        # et5_upload_planilha
            gr.update(visible=False),     # et5_dropdown_col_manual
            gr.update(visible=False),     # et5_dropdown_col_automatica
            gr.update(visible=False),     # et5_dropdown_col_resultado
            gr.update(visible=False),     # et5_input_nova_coluna
            gr.update(value="", visible=False),  # et5_status_carregamento
            gr.update(value=None, visible=False),  # et5_download_planilha
    
            # Visibilidade das etapas
            gr.update(visible=True),      # et1_grupo
            gr.update(visible=False),     # et2_grupo
            gr.update(visible=False),     # et3_grupo
            gr.update(visible=False),     # et4_grupo
            gr.update(visible=False),     # et5_grupo
    
            # Oculta ajuda
            gr.update(visible=False),     # chat_coluna_lateral
    
            # Estado de dados
            gr.update(value=None),        # df_state
    
            *updates_constructos
        )
#Conexões Entre Componentes 

# Etapa 1 – Salvar escopo e avançar para Etapa 2
    et1_botao_salvar.click(
        salvar_escopo_fn,
        inputs=[et1_input_escopo],
        outputs=[et1_output_escopo, et2_grupo]
    )

# Conexões – Chat de Ajuda (coluna lateral)
# Botão "Precisa de ajuda?" abre o chat
    botao_chat_abrir.click(
        abrir_chat, 
        inputs=[], 
        outputs=[chat_coluna_lateral]
    )
    
    # Botão " Fechar Chat" oculta o chat e limpa entrada
    botao_chat_fechar.click(
        fechar_chat,
        inputs=[], 
        outputs=[chat_coluna_lateral, chat_input_usuario]
    )

    # Envio de mensagem no campo de ajuda
    chat_input_usuario.submit(
        responder_chat, 
        inputs=[chat_input_usuario, chatbox_ajuda],
        outputs=[chatbox_ajuda, chat_input_usuario]
    )


#conexões etapa2
# Escolha do método de cadastro (manual ou planilha)
    et2_radio_metodo.change(
        escolher_metodo,
        inputs=[et2_radio_metodo],
        outputs=[
            et2_input_qtd, 
            et2_botao_confirmar_qtd,
            *[campo for par in et2_campos_constructos for campo in par],
            et2_botao_salvar,
            et2_upload_planilha
        ]
    )
    
    # Confirmação da quantidade de constructos a serem exibidos
    et2_botao_confirmar_qtd.click(
        mostrar_campos,
        inputs=[et2_input_qtd],
        outputs=[campo for par in et2_campos_constructos for campo in par] + [et2_botao_salvar]
    )
    
    # Processa os constructos digitados manualmente
    et2_botao_salvar.click(
        processar_constructos_fn,
        inputs=[campo for par in et2_campos_constructos for campo in par],
        outputs=[et2_output_resumo, et3_grupo, et4_dropdown_modelo]
    )
    
    # Carrega constructos a partir de planilha
    et2_upload_planilha.change(
        carregar_constructos_de_planilha,
        inputs=[et2_upload_planilha],
        outputs=[et2_output_resumo, et3_grupo, et4_dropdown_modelo]
    )

#conexões etapa 3: Upload de Quotes

    # Upload da planilha de quotes
    et3_upload_planilha.change(
        carregar_planilha,
        inputs=[et3_upload_planilha],
        outputs=[
            et3_dropdown_coluna_quote,
            et3_dropdown_coluna_classificacao,
            et3_texto_status,
            df_state,
            et4_botao_classificar,
            et4_grupo
        ]
    )
    
    # Habilita o botão de classificação quando todos os campos obrigatórios estão preenchidos
    et3_dropdown_coluna_quote.change(
        habilitar_botao_classificar,
        inputs=[et3_dropdown_coluna_quote, et3_dropdown_coluna_classificacao, et4_dropdown_modelo],
        outputs=[et4_botao_classificar]
    )
    
    et3_dropdown_coluna_classificacao.change(
        habilitar_botao_classificar,
        inputs=[et3_dropdown_coluna_quote, et3_dropdown_coluna_classificacao, et4_dropdown_modelo],
        outputs=[et4_botao_classificar]
    )
    
    et4_dropdown_modelo.change(
        habilitar_botao_classificar,
        inputs=[et3_dropdown_coluna_quote, et3_dropdown_coluna_classificacao, et4_dropdown_modelo],
        outputs=[et4_botao_classificar]
    )

#Conexões – Etapa 4: Classificação com IA
    # Clique no botão "Iniciar Classificação com IA" → confirmação do modelo
    et4_botao_classificar.click(
        confirmar_modelo_fn,
        inputs=[et4_dropdown_modelo],
        outputs=[
            et4_texto_confirmacao,
            et4_botao_interromper,
            et4_botoes_confirmacao
        ]
    )
    
    # Clique no botão "✅ Sim" → confirma e executa classificação
    et4_botao_sim.click(
        confirmar_e_classificar,
        inputs=[
            gr.State("Sim"),
            df_state,
            et3_dropdown_coluna_quote,
            et3_dropdown_coluna_classificacao,
            et4_dropdown_modelo
        ],
        outputs=[
            et4_status_classificacao,
            et4_arquivo_resultado,
            et4_grupo,
            et5_grupo,
            et4_botao_interromper,
            et4_botoes_confirmacao
        ]
    )
    
    # Clique no botão "❌ Não" → cancela a classificação
    et4_botao_nao.click(
        confirmar_e_classificar,
        inputs=[
            gr.State("Não"),
            df_state,
            et3_dropdown_coluna_quote,
            et3_dropdown_coluna_classificacao,
            et4_dropdown_modelo
        ],
        outputs=[
            et4_status_classificacao,
            et4_arquivo_resultado,
            et4_grupo,
            et5_grupo,
            et4_botao_interromper,
            et4_botoes_confirmacao
        ]
    )
    
    # Clique no botão "❌ Interromper Classificação"
    et4_botao_interromper.click(
        interromper_classificacao,
        inputs=[],
        outputs=[et3_texto_status]
    )

#Etapa 4 – Few-Shot Learning,

    # Mudança de modelo → decide se campos de few-shot devem ser exibidos
    et4_dropdown_modelo.change(
        mostrar_upload_exemplos,
        inputs=[et4_dropdown_modelo],
        outputs=[
            et4_upload_exemplos,
            et4_grupo_exemplos,
            et4_mensagem_exemplos
        ]
    )
    
    # Upload da planilha de exemplos anotados → popula os dropdowns de mapeamento
    et4_upload_exemplos.change(
        carregar_colunas_planilha_exemplos,
        inputs=[et4_upload_exemplos],
        outputs=[
            et4_dropdown_exemplo_quote,
            et4_dropdown_exemplo_constructo,
            et4_dropdown_exemplo_justificativa,
    
            et4_dropdown_exemplo_quote,
            et4_dropdown_exemplo_constructo,
            et4_dropdown_exemplo_justificativa,
    
            et4_botao_confirmar_exemplos
        ]
    )
    
    # Clique no botão "✅ Confirmar exemplos" → valida os exemplos e exibe resultado
    et4_botao_confirmar_exemplos.click(
        confirmar_exemplos,
        inputs=[
            et4_dropdown_exemplo_quote,
            et4_dropdown_exemplo_constructo,
            et4_dropdown_exemplo_justificativa
        ],
        outputs=[
            et4_status_exemplos_msg,
            et4_exemplos_formatados,
            et4_exemplos_formatados
        ]
    )
    
#Etapa 5 – Avaliação dos Resultados

    # Clique em "📥 Já tenho classificações. Quero avaliar!" → navega direto para a Etapa 5
    botao_ir_etapa5_avaliacao.click(
        ir_para_etapa5,
        inputs=[],
        outputs=[
            et1_grupo, et2_grupo, et3_grupo, et4_grupo, et5_grupo
        ]
    )
    
    # Upload da planilha classificada → exibe dropdowns com colunas detectadas
    et5_upload_planilha.change(
        carregar_planilha_avaliacao,
        inputs=[et5_upload_planilha],
        outputs=[
            et5_dropdown_col_manual,
            et5_dropdown_col_automatica,
            et5_dropdown_col_resultado,
            et5_botao_avaliar,
            et5_grupo,
            et5_status_carregamento
        ]
    )
    
    # Seleção de "Criar nova coluna..." → exibe input para nome personalizado
    et5_dropdown_col_resultado.change(
        mostrar_input_nova_coluna,
        inputs=[et5_dropdown_col_resultado],
        outputs=[et5_input_nova_coluna]
    )
    
    # Clique em "📊 Avaliar Classificação" → executa a avaliação e gera os arquivos
    et5_botao_avaliar.click(
        avaliar_classificacao_fn,
        inputs=[
            et5_dropdown_col_manual,
            et5_dropdown_col_automatica,
            et5_dropdown_col_resultado,
            et5_input_nova_coluna
        ],
        outputs=[
            et5_output_resultado,
            et5_download_planilha,
            et5_download_pdf
        ]
    )
    
    # Clique em "🔄 Iniciar nova análise" → reinicia tudo
    et5_botao_reiniciar.click(
        reiniciar_processo,
        inputs=[],
        outputs=[
            et1_input_escopo,
            et1_output_escopo,
            et2_upload_planilha,
            et3_upload_planilha,
            et3_dropdown_coluna_quote,
            et3_dropdown_coluna_classificacao,
            et3_texto_status,
            et4_dropdown_modelo,
            et4_texto_confirmacao,
            et4_status_classificacao,
            et4_arquivo_resultado,
            et4_botao_interromper,
            et4_botoes_confirmacao,
            et5_upload_planilha,
            et5_dropdown_col_manual,
            et5_dropdown_col_automatica,
            et5_dropdown_col_resultado,
            et5_input_nova_coluna,
            et5_status_carregamento,
            et5_download_planilha,
            et1_grupo,
            et2_grupo,
            et3_grupo,
            et4_grupo,
            et5_grupo,
            chat_coluna_lateral,
            df_state,
            *[campo for par in et2_campos_constructos for campo in par]
        ]
    )

# Executa o app
app.launch()   