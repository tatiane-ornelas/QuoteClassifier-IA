import pandas as pd

class ConstructLoader:
    """
    Classe responsável por carregar e armazenar constructos teóricos,
    utilizados na classificação de trechos (quotes).
    Permite tanto o carregamento manual quanto por planilha Excel.
    """

    def __init__(self):
        # Dicionário que armazena os constructos no formato {nome: definição}
        self.constructs = {}

    def load_manual(self, names_and_defs):
        """
        Carrega os constructos manualmente a partir de uma lista de tuplas (nome, definição).
        Ignora entradas vazias.

        Parâmetro:
        - names_and_defs: lista de tuplas (nome, definição)

        Retorna:
        - Dicionário com os constructos carregados
        """
        self.constructs = {n: d for n, d in names_and_defs if n and d}
        return self.constructs

    def load_from_excel(self, file_path):
        """
        Carrega os constructos de uma planilha Excel.

        Parâmetro:
        - file_path: caminho do arquivo Excel (.xlsx)

        Requisitos:
        - A planilha deve conter pelo menos duas colunas (nome e definição)

        Retorna:
        - Dicionário com os constructos carregados
        """
        df = pd.read_excel(file_path)
        if df.shape[1] < 2:
            raise ValueError("A planilha deve conter pelo menos duas colunas.")
        self.constructs = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
        return self.constructs

    def get_constructs(self):
        """
        Retorna o dicionário com os constructos carregados.
        """
        return self.constructs

    def get_formatted_summary(self):
        """
        Gera um resumo formatado dos constructos para exibição (ex: interface Gradio).

        Retorna:
        - String com os constructos formatados em Markdown
          Ex: "**Empatia**: Capacidade de se colocar no lugar do outro"
        """
        return "\n".join([f"**{k}**: {v}\n" for k, v in self.constructs.items()])