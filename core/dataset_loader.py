import pandas as pd

class QuoteDatasetLoader:
    """
    Classe responsável por carregar, armazenar e validar datasets (planilhas .xlsx)
    contendo quotes e colunas relacionadas à classificação.
    """

    def __init__(self):
        # Armazena o DataFrame carregado da planilha
        self.df = None

        # Caminho do arquivo original carregado, usado posteriormente para salvar versões modificadas
        self.file_path = None

    def load_excel(self, file_path):
        """
        Carrega uma planilha Excel e armazena o conteúdo em um DataFrame interno.

        Parâmetros:
        - file_path: caminho do arquivo Excel (.xlsx) a ser carregado

        Retorna:
        - Lista com os nomes das colunas da planilha, útil para dropdowns de seleção
        """
        self.df = pd.read_excel(file_path)
        self.file_path = file_path  # Guarda o caminho para uso futuro (ex: salvar versão classificada)
        return list(self.df.columns)

    def get_dataframe(self):
        """
        Retorna o DataFrame carregado para uso externo (ex: classificação).
        """
        return self.df
    
    def get_coluna_classificacao(self):
        """
        Retorna o nome da coluna mais provável de conter a classificação automática dos quotes.
        
        A função procura por colunas no DataFrame atual cujo nome contenha a palavra 'class'
        (ignorando maiúsculas/minúsculas), assumindo que essa coluna foi gerada pelo processo
        de classificação automatizada.
        
        Returns:
            str or None: O nome da coluna de classificação, se encontrada; caso contrário, None.
        """
        for col in self.df.columns:
            if 'class' in col.lower():
                return col
        return None

    def validate_columns(self, quote_col, class_col):
        """
        Verifica se as colunas fornecidas pelo usuário existem na planilha.

        Parâmetros:
        - quote_col: nome da coluna contendo os trechos (quotes)
        - class_col: nome da coluna onde as classificações serão salvas

        Levanta:
        - ValueError caso alguma das colunas não esteja presente no DataFrame
        """
        if quote_col not in self.df.columns or class_col not in self.df.columns:
            raise ValueError("Coluna não encontrada.")
    
    