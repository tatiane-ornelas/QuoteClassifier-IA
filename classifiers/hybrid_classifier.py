from sentence_transformers import SentenceTransformer, util
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from .base import BaseQuoteClassifier
import time

class HybridQuoteClassifier(BaseQuoteClassifier):
    """
    Classificador híbrido com few-shot:
    Combina embeddings para selecionar constructos mais similares e usa exemplos anotados com FewShotPromptTemplate.
    """

    def __init__(self, constructos, escopo, modelo="gpt-4", top_n=2, exemplos=None):
        """
        - constructos: dicionário {nome: definição}
        - escopo: contexto da pesquisa
        - modelo: modelo LLM da OpenAI
        - top_n: número de constructos mais similares para o prompt
        - exemplos: lista de dicionários com campos "quote", "constructo", "justificativa"
        """
        self.constructos = constructos
        self.escopo = escopo
        self.modelo = modelo
        self.top_n = top_n
        self.exemplos = exemplos or []

        self.embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')

        # Prompt do exemplo individual
        self.exemplo_prompt = ChatPromptTemplate.from_messages([
            ("human", 'Quote: "{quote}"'),
            ("ai", "Constructo: {constructo}\nJustificativa: {justificativa}")
        ])

    def classify(self, quotes):
        tempos = []
        resultados = []
        justificativas = []

        constructo_embeddings = {
            nome: self.embedder.encode(f"{nome}. {definicao}", convert_to_tensor=True)
            for nome, definicao in self.constructos.items()
        }

        for quote in quotes:
            inicio = time.time()

            quote_emb = self.embedder.encode(quote, convert_to_tensor=True)
            similaridades = {
                nome: util.cos_sim(quote_emb, emb).item()
                for nome, emb in constructo_embeddings.items()
            }

            # Seleciona top-N constructos semanticamente mais próximos
            top_n_constructos = sorted(similaridades.items(), key=lambda x: x[1], reverse=True)[:self.top_n]
            top_definicoes = "\n".join([f"{nome}: {self.constructos[nome]}" for nome, _ in top_n_constructos])

            # Cria prompt few-shot com estrutura de mensagens
            few_shot_prompt = FewShotChatMessagePromptTemplate(
                example_prompt=self.exemplo_prompt,
                examples=self.exemplos
            )

            prompt = ChatPromptTemplate.from_messages([
                ("system", (
                    "Você é um assistente treinado em análise qualitativa.\n"
                    "Escopo: {escopo}\n"
                    "Constructos mais similares:\n{top_definicoes}"
                )),
                few_shot_prompt,
                ("human", 'Quote: "{quote}"\nClassifique o trecho com Constructo e Justificativa.')
            ])

            chain = prompt | ChatOpenAI(model=self.modelo, temperature=0.4) | StrOutputParser()

            resposta = chain.invoke({
                "escopo": self.escopo,
                "top_definicoes": top_definicoes,
                "quote": quote
            })

            if "Justificativa:" in resposta:
                partes = resposta.split("Justificativa:")
                constructo_nome = partes[0].replace("Constructo:", "").strip()
                justificativa = partes[1].strip()
            else:
                constructo_nome = resposta.strip()
                justificativa = "Justificativa não fornecida."

            resultados.append(constructo_nome)
            justificativas.append(justificativa)
            tempos.append(time.time() - inicio)

        return resultados, justificativas, tempos
