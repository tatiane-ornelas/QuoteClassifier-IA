import gradio as gr
from main_controller import MainController
from dotenv import load_dotenv
from assistente_classificador import chain_assistente
import pandas as pd
# Carrega variáveis de ambiente (.env)
load_dotenv()

# Controlador principal do sistema
controller = MainController()

# === Interface ===
with gr.Blocks() as app:
    df_state = gr.State()

    with gr.Row():
        # === Coluna Lateral – Chat de Ajuda ===
        with gr.Group(visible=False) as coluna_lateral_ajuda:
            with gr.Column(scale=1):
                gr.Markdown("## ℹ️ Ajuda e Suporte")
                chatbot = gr.Chatbot(label="Assistente de Ajuda", height=300, type="messages")
                chat_input = gr.Textbox(placeholder="Digite sua dúvida e pressione Enter", show_label=False)
                botao_fechar_chat = gr.Button("❌ Fechar Chat")

        # === Coluna Principal ===
        with gr.Column(scale=4):
            gr.Markdown("## 🧠 QuoteClassifier-IA: Assistente Inteligente para Suporte à Análise Qualitativa")
            botao_deseja_chat = gr.Button("❓ Precisa de ajuda?", visible=True)
            botao_ir_para_avaliacao = gr.Button("📥 Já tenho classificações. Quero avaliar!", visible=True)

            # Etapa 1 – Escopo
            with gr.Group() as etapa1:
                gr.Markdown("### Etapa 1 – Escopo da Pesquisa")
                escopo_input = gr.Textbox(label="Digite o escopo da sua pesquisa")
                escopo_out = gr.Markdown()
                botao_escopo = gr.Button("Salvar escopo")

            # Etapa 2 – Constructos
            with gr.Group(visible=False) as etapa2:
                gr.Markdown("### Etapa 2 – Cadastro de Constructos")
                metodo_input = gr.Radio(["Manual", "Upload de planilha"], label="Como deseja cadastrar os constructos?")
                qtd_input = gr.Number(label="Quantos constructos deseja cadastrar?", precision=0, visible=False)
                botao_qtd = gr.Button("Confirmar quantidade", visible=False)

                MAX_CONSTRUCTOS = 10
                campos_constructos = [
                    (gr.Textbox(label=f"Nome do Constructo {i+1}", visible=False),
                     gr.Textbox(label=f"Definição do Constructo {i+1}", visible=False))
                    for i in range(MAX_CONSTRUCTOS)
                ]

                botao_salvar = gr.Button("Salvar constructos", visible=False)
                upload_constructos = gr.File(label="Upload de planilha (.xlsx)", file_types=[".xlsx"], visible=False)
                resumo_out = gr.Markdown()

            # Etapa 3 – Upload de quotes
            with gr.Group(visible=False) as etapa3:
                gr.Markdown("### Etapa 3 – Upload da Planilha e Seleção de Colunas")
                upload_file = gr.File(label="Faça upload da planilha (.xlsx)", interactive=True)
                dropdown_quote = gr.Dropdown(label="Coluna com os QUOTES", visible=False)
                dropdown_class = gr.Dropdown(label="Coluna para CLASSIFICAÇÃO", visible=False)
                status_text = gr.Markdown()

            # Etapa 4 – Classificação
            with gr.Group(visible=False) as etapa4:
                gr.Markdown("### Etapa 4 – Classificação")
                modelo_dropdown = gr.Dropdown(
                    label="Modelo de classificação",
                    choices=[
                        "EmbeddingQuoteClassifier", "openai-3.5", "openai-4",
                        "HybridQuoteClassifier", "ConstructSimilarityClassifier"
                    ]
                )
                #  NOVOS COMPONENTES – FEW-SHOT
                mensagem_exemplos = gr.Markdown(
                    "**ℹ️ Se desejar incluir exemplos anotados (few-shot learning)** para ajudar os modelos baseados em linguagem natural, envie uma planilha com as colunas: **quote**, **constructo** e **justificativa**.",
                    visible=False
                )            
                # Upload da planilha de exemplos
                upload_exemplos = gr.File(
                    label="📄 Upload de exemplos anotados (opcional)", 
                    file_types=[".xlsx"],
                    visible=False
                )  
                # Status do carregamento dos exemplos
                with gr.Row(visible=False) as grupo_exemplos:
                    texto_status_exemplos = gr.Markdown()        

              # Seleção de colunas da planilha de exemplos
                dropdown_col_quote_exemplo = gr.Dropdown(label="Coluna do Quote", visible=False, interactive=True)
                dropdown_col_constructo_exemplo = gr.Dropdown(label="Coluna do Constructo", visible=False, interactive=True)
                dropdown_col_justificativa_exemplo = gr.Dropdown(label="Coluna da Justificativa", visible=False, interactive=True)
                
                # Confirmação dos exemplos carregados
                botao_confirmar_exemplos = gr.Button("✅ Confirmar exemplos", visible=False)
                msg_exemplos = gr.Textbox(label="Status dos Exemplos", visible=False)
            
                # Visualização dos exemplos formatados
                exemplos_formatados = gr.Textbox(
                    label="Exemplos Confirmados",
                    visible=False,
                    interactive=False,
                    lines=15
                )
                # Botão principal de classificação
                botao_classificar = gr.Button("Iniciar Classificação com IA")
                confirmacao_texto = gr.Markdown(visible=False)
                
                with gr.Row(visible=False) as confirmacao_botoes:
                    botao_sim = gr.Button("✅ Sim")
                    botao_nao = gr.Button("❌ Não")
                
                status_class = gr.Markdown()
                file_output = gr.File(label="Download da Planilha Classificada")
                botao_interromper = gr.Button("❌ Interromper Classificação", visible=False)
                        
            # Etapa 5 – Avaliação
            with gr.Group(visible=False) as etapa5:
                gr.Markdown("### Etapa 5 – Avaliação dos Resultados")
                upload_avaliacao = gr.File(label="Upload da planilha classificada (.xlsx)", file_types=[".xlsx"])
                status_avaliacao = gr.Markdown(visible=False)
                dropdown_manual = gr.Dropdown(label="Coluna com Classificação Manual", visible=False)
                dropdown_automatica = gr.Dropdown(label="Coluna com Classificação Automática", visible=False)
                dropdown_resultado = gr.Dropdown(
                    label="Coluna para salvar Resultado (Certa/Errada)", 
                    visible=False)
                nova_coluna_input = gr.Textbox(
                    label="Nome da nova coluna de resultado",
                    placeholder="Ex: Resultado",
                    visible=False
                    )
                botao_avaliar = gr.Button("📊 Avaliar Classificação", visible=False)
                resultados_avaliacao = gr.Markdown()
                download_relatorio = gr.File(label="Download da Planilha Avaliada", visible=False)
                botao_reiniciar = gr.Button("🔄 Iniciar nova análise")
                download_pdf = gr.File(label="📄 Download do Relatório em PDF", visible=False)

    # === Funções ===
    def mostrar_input_nova_coluna(col_escolhida):
        if col_escolhida == "Criar nova coluna...":
            return gr.update(visible=True)
        return gr.update(visible=False)

    def ir_para_etapa5():
        return (
            gr.update(visible=False),  # etapa1
            gr.update(visible=False),  # etapa2
            gr.update(visible=False),  # etapa3
            gr.update(visible=False),  # etapa4
            gr.update(visible=True),   # etapa5
        )
    
    def carregar_planilha_avaliacao(arquivo):
        try:
            colunas = controller.carregar_planilha_quotes(arquivo.name)
            return (
                gr.update(choices=colunas, visible=True),
                gr.update(choices=colunas, visible=True),
                gr.update(choices=colunas + ["Criar nova coluna..."], visible=True),
                gr.update(visible=True),
                gr.update(visible=True),  # Etapa5
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



    def avaliar_classificacao_fn(col_manual, col_automatica, col_resultado, nome_novo):
        try:
            if col_resultado == "Criar nova coluna...":
                col_resultado = nome_novo.strip() or "Resultado"
    
            # Avaliação com múltiplos retornos
            markdown, caminho_arquivo, _, acertos, erros, pdf_path, relatorio_md = controller.avaliar_resultados(
                col_manual, col_automatica, col_resultado
            )
    
            markdown_total = f"{markdown.strip()}\n\n{relatorio_md.strip()}"
    
            return (
                markdown_total,
                gr.update(value=caminho_arquivo, visible=True),
                gr.update(value=pdf_path, visible=True)
            )
        except Exception as e:
            return (
                f"❌ Erro na avaliação: {e}",
                gr.update(visible=False),
                gr.update(visible=False)
            )

    
    def abrir_chat(): return gr.update(visible=True)
    def fechar_chat():  return gr.update(visible=False), []

    def responder_chat(msg, hist):
        resposta = chain_assistente.invoke({"pergunta": msg})
    
        # Converte tuplas antigas (se houver) para o formato correto
        hist_formatado = []
        for item in hist:
            if isinstance(item, tuple) and len(item) == 2:
                hist_formatado.append({"role": "user", "content": item[0]})
                hist_formatado.append({"role": "assistant", "content": item[1]})
            elif isinstance(item, dict) and "role" in item and "content" in item:
                hist_formatado.append(item)
    
        # Adiciona nova interação
        hist_formatado.append({"role": "user", "content": msg})
        hist_formatado.append({"role": "assistant", "content": resposta})
    
        return hist_formatado, ""



    def salvar_escopo_fn(txt):
        controller.salvar_escopo(txt)
        return gr.update(value=f"✅ Escopo salvo: {txt}"), gr.update(visible=True)

    def escolher_metodo(metodo):
        vis_manual = metodo == "Manual"
        vis_upload = metodo == "Upload de planilha"
        updates = [gr.update(visible=vis_manual), gr.update(visible=vis_manual)]
        updates += [gr.update(visible=vis_manual) for _ in campos_constructos for _ in range(2)]
        updates += [gr.update(visible=vis_manual), gr.update(visible=vis_upload)]
        return updates

    def mostrar_campos(qtd):
        updates = []
        for i, (n, d) in enumerate(campos_constructos):
            updates.extend([gr.update(visible=(i < qtd)), gr.update(visible=(i < qtd))])
        return updates + [gr.update(visible=True)]

    def processar_constructos_fn(*entradas):
        resumo = controller.carregar_constructos_manualmente(entradas)
        return resumo, gr.update(visible=True), gr.update(visible=True)

    def carregar_constructos_de_planilha(arquivo):
        try:
            resumo = controller.carregar_constructos_de_planilha(arquivo.name)
            return resumo, gr.update(visible=True), gr.update(visible=True)
        except Exception as e:
            return f"❌ Erro: {e}", gr.update(visible=False), gr.update(visible=False)

    def carregar_planilha(file):
        if file is None:
            return (
                gr.update(visible=False), gr.update(visible=False),
                gr.update(value="ℹ️ Nenhuma planilha carregada"), gr.update(value=None),
                gr.update(visible=False), gr.update(visible=False)
            )
        try:
            colunas = controller.carregar_planilha_quotes(file.name)
            df = controller.quote_loader.get_dataframe()
            return (
                gr.update(choices=colunas, visible=True),
                gr.update(choices=colunas, visible=True),
                gr.update(value="✅ Planilha carregada com sucesso"),
                gr.update(value=df),
                gr.update(visible=True), gr.update(visible=True)
            )
        except Exception as e:
            return (
                gr.update(visible=False), gr.update(visible=False),
                gr.update(value=f"❌ Erro ao carregar a planilha: {e}"),
                gr.update(value=None),
                gr.update(visible=False), gr.update(visible=False)
            )

    def confirmar_modelo_fn(modelo):
        return (
            gr.update(value=f"Você confirma a seleção do modelo: **{modelo}**?", visible=True),
            gr.update(visible=True),
            gr.update(visible=True)
        )

    def confirmar_e_classificar(resp, df, q, c, modelo):
        if resp != "Sim":
            return (
                "❌ Cancelado. Você pode escolher outro modelo ou revisar os dados e clicar em Iniciar Classificação com IA novamente.",
                None, gr.update(visible=True), 
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False)
            )
    
        controller.resetar_interrupcao()
    
        # Verifica se há exemplos confirmados (opcional)
        exemplos = controller.exemplos_usuario  # ✅ Corrigido aqui
    
        # Passa os exemplos para os classificadores que suportam
        suporta_exemplos = modelo in ["openai-3.5", "openai-4", "ConstructSimilarityClassifier"]
    
        status, arquivo = controller.classificar(
            col_quote=q,
            col_class=c,
            modelo=modelo,
            progress=None,
            deve_interromper=controller.verificar_interrupcao,
            exemplos=exemplos if suporta_exemplos else None
        )
    
        return (
            status, arquivo,
            gr.update(visible=True), gr.update(visible=True),
            gr.update(visible=False), gr.update(visible=False)
        )


    def interromper_classificacao():
        controller.solicitar_interrupcao()
        return gr.update(value="⚠️ Interrupção solicitada.")

    def reiniciar_processo():
        controller.resetar_interrupcao()
        updates_constructos = [gr.update(value="", visible=False) for pair in campos_constructos for item in pair]
    
        return (
            gr.update(value=""),  # escopo_input
            gr.update(value=""),  # escopo_out
            gr.update(value=None),  # upload_file
            gr.update(value=None),  # upload_constructos
            gr.update(choices=[], value=None, visible=False),  # dropdown_quote
            gr.update(choices=[], value=None, visible=False),  # dropdown_class
            gr.update(value=None),  # df_state
            gr.update(value=""),  # status_text
            gr.update(value=""),  # status_class
            gr.update(value=""),  # confirmacao_texto
            gr.update(value=""),  # resumo_out
            gr.update(value=None),  # modelo_dropdown
            gr.update(visible=False),  # etapa2
            gr.update(visible=False),  # etapa3
            gr.update(visible=False),  # etapa4
            gr.update(visible=False),  # etapa5
            gr.update(visible=False),  # botao_interromper 
            gr.update(visible=False),  # confirmacao_botoes
            gr.update(value=None),  # file_output
            gr.update(visible=True),  # etapa1
            gr.update(visible=False),  # coluna_lateral_ajuda
    
            # Limpeza da Etapa 5
            gr.update(visible=False),  # dropdown_manual
            gr.update(visible=False),  # dropdown_automatica
            gr.update(visible=False),  # dropdown_resultado
            gr.update(visible=False),  # nova_coluna_input
            gr.update(value="", visible=False),  # status_avaliacao
            gr.update(value=None, visible=False),  # download_relatorio
    
            *updates_constructos
        )

    def habilitar_botao_classificar(q, c, m): return gr.update(visible=bool(q and c and m))


    def mostrar_upload_exemplos(modelo):
        # Define os modelos que suportam few-shot
        visivel = modelo in ["openai-3.5", "openai-4", "HybridQuoteClassifier"]
        return (
            gr.update(visible=visivel),  # upload_exemplos
            gr.update(visible=visivel),  # grupo_exemplos
            gr.update(visible=visivel)   # mensagem_exemplos
        )
    
    
    def carregar_colunas_planilha_exemplos(arquivo):
        try:
            df = pd.read_excel(arquivo.name)
            colunas = df.columns.tolist()
            controller._df_exemplos_temp = df  # Armazena o DataFrame temporariamente
            return (
                gr.update(choices=colunas, visible=True),  # dropdown_col_quote_exemplo
                gr.update(choices=colunas, visible=True),  # dropdown_col_constructo_exemplo
                gr.update(choices=colunas, visible=True),  # dropdown_col_justificativa_exemplo
                gr.update(visible=True),                   # dropdown_col_quote_exemplo
                gr.update(visible=True),                   # dropdown_col_constructo_exemplo
                gr.update(visible=True),                   # dropdown_col_justificativa_exemplo
                gr.update(visible=True)                    # botao_confirmar_exemplos
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
    
    
    def confirmar_exemplos(col_quote, col_constructo, col_justificativa):
        df = controller._df_exemplos_temp.copy()
        exemplos = [
            {
                "quote": str(row[col_quote]),
                "constructo": str(row[col_constructo]),
                "justificativa": str(row[col_justificativa])
            }
            for _, row in df.iterrows()
        ]
        controller.setar_exemplos_usuario(exemplos)
    
        # 🔄 Formatação dos exemplos para exibição
        exemplos_formatados = ""
        for i, exemplo in enumerate(exemplos, start=1):
            exemplos_formatados += f"Exemplo {i}:\nQuote: {exemplo['quote']}\nConstructo: {exemplo['constructo']}\nJustificativa: {exemplo['justificativa']}\n\n"
    
        return (
            f"✅ {len(exemplos)} exemplos confirmados com sucesso!",
            exemplos_formatados.strip(),
            gr.update(visible=True)
        )



    # === Conexões ===
    dropdown_resultado.change(
    mostrar_input_nova_coluna,
    [dropdown_resultado],
    [nova_coluna_input]
    )
    botao_ir_para_avaliacao.click(
        ir_para_etapa5,
        inputs=[],
        outputs=[etapa1, etapa2, etapa3, etapa4, etapa5]
    )    
    upload_avaliacao.change(
        carregar_planilha_avaliacao,
        [upload_avaliacao],
        [dropdown_manual, dropdown_automatica, dropdown_resultado, botao_avaliar, etapa5, status_avaliacao]
    )

    botao_avaliar.click(
        avaliar_classificacao_fn,
        [dropdown_manual, dropdown_automatica, dropdown_resultado, nova_coluna_input],
        [resultados_avaliacao, download_relatorio, download_pdf]
    )
    botao_deseja_chat.click(abrir_chat, [], [coluna_lateral_ajuda])
    botao_fechar_chat.click(fechar_chat, [], [coluna_lateral_ajuda, chat_input])
    chat_input.submit(
        responder_chat, 
        [chat_input, chatbot], 
        [chatbot, chat_input])

    botao_escopo.click(salvar_escopo_fn, [escopo_input], [escopo_out, etapa2])
    metodo_input.change(escolher_metodo, [metodo_input], [qtd_input, botao_qtd] + [c for p in campos_constructos for c in p] + [botao_salvar, upload_constructos])
    botao_qtd.click(mostrar_campos, [qtd_input], [c for p in campos_constructos for c in p] + [botao_salvar])
    botao_salvar.click(processar_constructos_fn, [c for p in campos_constructos for c in p], [resumo_out, etapa3, modelo_dropdown])
    upload_constructos.change(carregar_constructos_de_planilha, [upload_constructos], [resumo_out, etapa3, modelo_dropdown])

    upload_file.change(carregar_planilha, [upload_file], [dropdown_quote, dropdown_class, status_text, df_state, botao_classificar, etapa4])
    dropdown_quote.change(habilitar_botao_classificar, [dropdown_quote, dropdown_class, modelo_dropdown], [botao_classificar])
    dropdown_class.change(habilitar_botao_classificar, [dropdown_quote, dropdown_class, modelo_dropdown], [botao_classificar])
    modelo_dropdown.change(habilitar_botao_classificar, [dropdown_quote, dropdown_class, modelo_dropdown], [botao_classificar])

    botao_classificar.click(confirmar_modelo_fn, [modelo_dropdown], [confirmacao_texto, botao_interromper, confirmacao_botoes])
    botao_sim.click(confirmar_e_classificar, [gr.State("Sim"), df_state, dropdown_quote, dropdown_class, modelo_dropdown], [status_class, file_output, etapa4, etapa5, botao_interromper, confirmacao_botoes])
    botao_nao.click(confirmar_e_classificar, [gr.State("Não"), df_state, dropdown_quote, dropdown_class, modelo_dropdown], [status_class, file_output, etapa4, etapa5, botao_interromper, confirmacao_botoes])

    botao_interromper.click(interromper_classificacao, [], [status_text])
    botao_reiniciar.click(reiniciar_processo, [], [
        escopo_input, escopo_out, upload_file, upload_constructos,
        dropdown_quote, dropdown_class, qtd_input,
        status_text, status_class, confirmacao_texto, resumo_out,
        modelo_dropdown, etapa2, etapa3, etapa4, etapa5,
        botao_interromper, confirmacao_botoes,
        df_state, etapa1, coluna_lateral_ajuda, file_output
    ] + [item for pair in campos_constructos for item in pair])


    # Quando o modelo é alterado, verifica se deve exibir os campos de few-shot
    modelo_dropdown.change(
        mostrar_upload_exemplos,
        inputs=[modelo_dropdown],
        outputs=[upload_exemplos, grupo_exemplos, mensagem_exemplos]
    )
    
    # Quando o usuário faz upload da planilha de exemplos anotados
    upload_exemplos.change(
        carregar_colunas_planilha_exemplos,
        inputs=[upload_exemplos],
        outputs=[
            dropdown_col_quote_exemplo,
            dropdown_col_constructo_exemplo,
            dropdown_col_justificativa_exemplo,
            
            dropdown_col_quote_exemplo,
            dropdown_col_constructo_exemplo,
            dropdown_col_justificativa_exemplo,
            
            botao_confirmar_exemplos
        ]
    )
    
    # Quando o botão "Confirmar exemplos" é clicado
    botao_confirmar_exemplos.click(
        confirmar_exemplos,
        inputs=[
            dropdown_col_quote_exemplo,
            dropdown_col_constructo_exemplo,
            dropdown_col_justificativa_exemplo
        ],
        outputs=[
            msg_exemplos,
            exemplos_formatados,
            exemplos_formatados
        ]
    )


# Executa o app
app.launch()
