import pandas as pd
from pathlib import Path
from classifiers.base import BaseQuoteClassifier

class ClassificationPipeline:
    """
    Classe responsável por executar o pipeline de classificação de quotes.
    Recebe um DataFrame, o nome da coluna com os quotes, o nome da coluna onde será
    inserida a classificação, e um classificador que implementa a interface BaseQuoteClassifier.
    """    
    def __init__(self, df, quote_column, class_column, classifier: BaseQuoteClassifier):
        # Criando uma cópia do DataFrame original para preservar os dados de entrada
        self.df = df.copy()
        self.quote_column = quote_column  # Nome da coluna com os trechos (quotes)
        self.class_column = class_column  #Nome da coluna onde será inserida a classificação
        self.classifier = classifier # Instância do classificador (LLM, embedding ou híbrido,ou outro)
        self.tempos = []  # Lista para armazenar o tempo de processamento de cada quote

    def run(self, progress=None, deve_interromper=None) -> pd.DataFrame:
        """
        Executa o pipeline de classificação dos quotes no DataFrame,
        utilizando o classificador fornecido.

        Parâmetros:
            progress (callable, opcional): função para indicar progresso ao usuário.
            deve_interromper (callable, opcional): função que retorna True se o processo
            deve ser interrompido (ex: clique do usuário em botão de parar).

        Retorna:
            pd.DataFrame: DataFrame com as colunas de classificação e justificativa adicionadas.
        """
        quotes = self.df[self.quote_column].tolist()

        resultados, justificativas, tempos = self.classifier.classify(quotes)

        resultados_finais = []
        justificativas_finais = []

        for i, t in enumerate(tempos):
            # Verifica se uma função de interrupção foi fornecida e se ela retorna True.
            # Isso permite que o pipeline seja interrompido de forma segura durante a execução,
            # útil, por exemplo, quando o usuário clica em um botão "Interromper" na interface.
            if deve_interromper and deve_interromper():
                print("⚠️ Interrupção solicitada. Encerrando processamento.")
                break

            print(f"Quote {i+1} levou {t:.2f} segundos")

            resultados_finais.append(resultados[i])
            justificativas_finais.append(justificativas[i])

            if progress:
                progress(i + 1, len(quotes))

        # Preenche os resultados no DataFrame (inclusive se incompleto por interrupção)
        self.df.loc[:len(resultados_finais)-1, self.class_column] = resultados_finais
        self.df.loc[:len(justificativas_finais)-1, self.class_column + "_justificativa"] = justificativas_finais

        return self.df
    
    def export(self, path: str | Path):
        """
        Exporta o DataFrame classificado para um arquivo Excel (.xlsx).
        Retorna o caminho do arquivo salvo.
        """
        self.df.to_excel(path, index=False)
        return str(path)