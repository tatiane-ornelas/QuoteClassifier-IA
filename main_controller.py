from core.construct_loader import ConstructLoader
from core.dataset_loader import QuoteDatasetLoader
from core.pipeline import ClassificationPipeline
from classifiers.embedding import EmbeddingQuoteClassifier
from classifiers.llm import LLMQuoteClassifier
from classifiers.hybrid_classifier import HybridQuoteClassifier
from classifiers.similarity_classifier import ConstructSimilarityClassifier
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

# Garante que a pasta 'results/' exista
os.makedirs("results", exist_ok=True)

class MainController:
    """
    Classe principal de orquestração. Gerencia o fluxo de entrada de dados,
    seleção de modelo e execução da classificação de quotes com base nos constructos definidos.
    """

    def __init__(self):
        self.construct_loader = ConstructLoader()
        self.quote_loader = QuoteDatasetLoader()
        self.escopo_pesquisa = ""  # Texto do escopo definido pelo usuário
        self.interromper = False   # Flag para interrupção da classificação
        self.exemplos_usuario = []  # Exemplos fornecidos pelo usuário (para few-shot)

    def carregar_exemplos_usuario(self, file_path):
        """Carrega exemplos anotados pelo usuário a partir de um arquivo Excel."""
        try:
            df = pd.read_excel(file_path)
            colunas_esperadas = {"Quote", "Constructo", "Justificativa"}
            colunas_arquivo = set(df.columns.str.strip())
            if not colunas_esperadas.issubset(colunas_arquivo):
                raise ValueError("A planilha deve conter as colunas: Quote, Constructo e Justificativa")

            exemplos = []
            for _, row in df.iterrows():
                quote = str(row["Quote"]).strip()
                constructo = str(row["Constructo"]).strip()
                justificativa = str(row["Justificativa"]).strip()
                if quote and constructo and justificativa:
                    exemplos.append({
                        "quote": quote,
                        "constructo": constructo,
                        "justificativa": justificativa
                    })

            if not exemplos:
                raise ValueError("A planilha está vazia ou os exemplos estão incompletos.")

            self.exemplos_usuario = exemplos
            return f"✅ {len(exemplos)} exemplo(s) carregado(s) com sucesso."
        except Exception as e:
            return f"❌ Erro ao carregar exemplos: {str(e)}"

    def setar_exemplos_usuario(self, exemplos: list):
        """Define manualmente a lista de exemplos do usuário."""
        self.exemplos_usuario = exemplos

    def salvar_escopo(self, texto):
        """Armazena o escopo textual fornecido pelo usuário."""
        self.escopo_pesquisa = texto
        return f"Escopo salvo:\n\n**{self.escopo_pesquisa}**"

    def carregar_constructos_manualmente(self, entradas):
        """Registra os constructos inseridos manualmente pelo usuário."""
        pares = list(zip(entradas[::2], entradas[1::2]))
        self.construct_loader.load_manual(pares)
        return self.construct_loader.get_formatted_summary()

    def carregar_constructos_de_planilha(self, file_path):
        """Carrega constructos a partir de uma planilha Excel."""
        self.construct_loader.load_from_excel(file_path)
        return self.construct_loader.get_formatted_summary()

    def carregar_planilha_quotes(self, file_path):
        """Carrega os quotes de um arquivo Excel e retorna os nomes das colunas disponíveis."""
        colunas = self.quote_loader.load_excel(file_path)
        return colunas

    def carregar_exemplos_anotados(self, caminho_arquivo):
        """Carrega exemplos anotados diretamente em formato dicionário."""
        df = pd.read_excel(caminho_arquivo)
        exemplos = []
        for _, row in df.iterrows():
            exemplo = {
                "quote": str(row["quote"]).strip(),
                "constructo": str(row["constructo"]).strip(),
                "justificativa": str(row.get("justificativa", "")).strip()
            }
            exemplos.append(exemplo)
        return exemplos

    def resetar_interrupcao(self):
        self.interromper = False

    def solicitar_interrupcao(self):
        self.interromper = True

    def verificar_interrupcao(self):
        return self.interromper

    def avaliar_resultados(self, col_manual, col_automatica, col_resultado):
        """
        Avalia a acurácia da classificação automática comparando com a classificação manual.
        Também gera um relatório em PDF com a matriz de confusão e estatísticas.
    
        Parâmetros:
            col_manual (str): Nome da coluna com os rótulos manuais.
            col_automatica (str): Nome da coluna com os rótulos gerados pelo modelo.
            col_resultado (str): Nome da coluna onde será registrada a comparação ("Certa"/"Errada").
    
        Retorna:
            - markdown (str): Resumo da avaliação formatado em Markdown.
            - nome_saida (str): Nome do arquivo Excel salvo com os resultados.
            - matriz_confusao (DataFrame): Matriz de confusão calculada.
            - acertos (int): Quantidade de classificações corretas.
            - erros (int): Quantidade de classificações incorretas.
            - pdf_path (str): Caminho do arquivo PDF gerado com o relatório.
            - relatorio_md (str): Relatório de classificação formatado em Markdown.
        """
        # Verifica se há dados carregados
        if not self.quote_loader or self.quote_loader.get_dataframe().empty:
            raise ValueError("❌ Nenhum dado foi carregado.")
    
        # Carrega o DataFrame com os quotes
        df = self.quote_loader.get_dataframe()
    
        # Garante que as colunas informadas existem no DataFrame
        for col in [col_manual, col_automatica]:
            if col not in df.columns:
                raise ValueError(f"❌ A coluna '{col}' não foi encontrada na planilha.")
    
        # Cria a coluna de resultado, se ainda não existir
        if col_resultado not in df.columns:
            df[col_resultado] = ""
    
        # Normaliza os valores: converte para minúsculas e remove espaços extras
        y_true = df[col_manual].astype(str).str.strip().str.lower()
        y_pred = df[col_automatica].astype(str).str.strip().str.lower()
    
        # Compara rótulos: marca "Certa" ou "Errada"
        df[col_resultado] = ["Certa" if p == t else "Errada" for p, t in zip(y_pred, y_true)]
    
        # Calcula acurácia, total de acertos e erros
        acuracia = accuracy_score(y_true, y_pred)
        acertos = sum(df[col_resultado] == "Certa")
        erros = sum(df[col_resultado] == "Errada")
    
        # Insere a acurácia ao final da coluna de classificação automática
        idx_nova_linha = len(df)
        df.loc[idx_nova_linha, col_automatica] = f"Acurácia: {acuracia:.2%}"
    
        # Calcula a matriz de confusão e o relatório de classificação
        matriz_confusao = pd.crosstab(y_true, y_pred, rownames=["Manual"], colnames=["Automática"])
        relatorio_classificacao = classification_report(y_true, y_pred, zero_division=0)
    
        # Gera nome do novo arquivo Excel
        caminho_original = self.quote_loader.file_path
        nome_base = os.path.splitext(os.path.basename(caminho_original))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nome_saida = os.path.join("results", f"{nome_base}_avaliado_{timestamp}.xlsx")
    
        # Salva os resultados no Excel
        df.to_excel(nome_saida, index=False)
    
        # Prepara o Markdown com resumo da avaliação (para exibir no Gradio)
        markdown = f"""
    ### ✅ Avaliação Gravada com Sucesso
    
    - Coluna de resultado: `{col_resultado}` com "Certa"/"Errada"
    - Acurácia: `{acuracia:.2%}` (gravada ao final da coluna `{col_automatica}`)
    - Total de registros: **{len(df) - 1}**
    - Acertos: **{acertos}**
    - Erros: **{erros}**
    """
    
        # Formata o relatório de classificação como Markdown para o app
        relatorio_md = f"### 📊 Relatório de Classificação\n```\n{relatorio_classificacao}\n```"
    
        # Gera um PDF com as principais informações da avaliação
        pdf_path = os.path.join("results", f"{nome_base}_avaliacao_{timestamp}.pdf")
        with PdfPages(pdf_path) as pdf:
            fig, ax = plt.subplots(figsize=(8.3, 11.7))  # Formato A4
            ax.axis("off")
            texto = f"""
                AVALIAÇÃO DE CLASSIFICAÇÃO
                
                Acurácia: {acuracia:.2%}
                Total de registros: {acertos + erros}
                Acertos: {acertos}
                Erros: {erros}
                
              
                Relatório de Classificação:
                {relatorio_classificacao}
                """
            ax.text(0.05, 0.95, texto, va='top', family='monospace', fontsize=9, wrap=True)
            pdf.savefig(fig)
            plt.close()
    
        # Retorna todos os dados úteis para exibição e download
        return markdown, nome_saida, matriz_confusao, acertos, erros, pdf_path, relatorio_md


    def get_coluna_classificacao(self):
        """Retorna a primeira coluna com 'class' no nome (útil para autocompletar)."""
        for col in self.quote_loader.get_dataframe().columns:
            if 'class' in col.lower():
                return col
        return None


    def classificar(self, col_quote, col_class, modelo, progress=None, deve_interromper=None, exemplos=None):

        """
        Executa a classificação dos quotes utilizando o modelo selecionado.
        Retorna mensagem de sucesso e nome do arquivo Excel gerado.
        """
        constructos = self.construct_loader.get_constructs()
        df = self.quote_loader.get_dataframe()

        # Seleciona e instancia o classificador conforme o modelo informado
        if modelo == "EmbeddingQuoteClassifier":
            classifier = EmbeddingQuoteClassifier(constructos)
        elif modelo == "HybridQuoteClassifier":
            classifier = HybridQuoteClassifier(
                constructos,
                escopo=self.escopo_pesquisa,
                modelo="gpt-4",
                exemplos=exemplos
            )
        elif modelo == "ConstructSimilarityClassifier":
            classifier = ConstructSimilarityClassifier(
                constructos,
                escopo=self.escopo_pesquisa,
                modelo="gpt-4"
                
            )
        else:
            classifier = LLMQuoteClassifier(
                constructos,
                escopo=self.escopo_pesquisa,
                modelo="gpt-3.5-turbo" if modelo == "openai-3.5" else "gpt-4",
                exemplos=exemplos

            )

        # Executa o pipeline
        pipeline = ClassificationPipeline(df, col_quote, col_class, classifier)
        pipeline.run(progress=progress, deve_interromper=deve_interromper)

        caminho_original = self.quote_loader.file_path
        nome_base = os.path.splitext(os.path.basename(caminho_original))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        sufixo_few_shot = "_FEW-SHOT" if exemplos else ""
        nome_arquivo = os.path.join("results", f"{nome_base}_{modelo.replace('-', '_')}{sufixo_few_shot}_{timestamp}.xlsx")
        

        pipeline.export(nome_arquivo)
        return "✅ Classificação concluída!", nome_arquivo
