import time
from sentence_transformers import SentenceTransformer, util
from .base import BaseQuoteClassifier

class EmbeddingQuoteClassifier(BaseQuoteClassifier):
    """
    Classificador de trechos (quotes) baseado em similaridade semântica
    utilizando embeddings da biblioteca SentenceTransformers.
    
    Para cada quote, o classificador calcula a similaridade com os constructos
    e retorna o constructo mais similar.
    """

    def __init__(self, constructos: dict):
        """
        Inicializa o classificador com:
        - constructos: dicionário {nome: definição} com os constructos da pesquisa.
        """
        self.constructos = constructos

        # Modelo de embeddings escolhido para representar textos em vetores
        self.embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    def classify(self, quotes):
        """
        Classifica uma lista de quotes com base na maior similaridade de embedding
        entre o quote e os constructos definidos.

        Para cada quote:
        - Gera o embedding do texto
        - Compara com os embeddings dos constructos
        - Retorna o constructo mais similar e a justificativa da escolha

        Retorna:
        - Lista com nomes dos constructos mais similares
        - Lista com justificativas baseadas na similaridade
        - Lista com tempo (em segundos) de cada classificação
        """
        resultados = []
        justificativas = []
        tempos = []

        # Pré-processa os embeddings dos constructos
        constructo_embeddings = {
            nome: self.embedder.encode(nome + ". " + definicao, convert_to_tensor=True)
            for nome, definicao in self.constructos.items()
        }

        for quote in quotes:
            start = time.time()  # Início da medição de tempo

            # Gera embedding do quote
            quote_emb = self.embedder.encode(quote, convert_to_tensor=True)

            # Calcula similaridade com cada constructo
            similaridades = {
                nome: util.cos_sim(quote_emb, emb).item()
                for nome, emb in constructo_embeddings.items()
            }

            # Seleciona o constructo mais similar
            melhor_constructo = max(similaridades.items(), key=lambda x: x[1])

            end = time.time()
            tempos.append(end - start)

            resultados.append(melhor_constructo[0])
            justificativas.append(f"Similaridade: {melhor_constructo[1]:.4f}")

        return resultados, justificativas, tempos