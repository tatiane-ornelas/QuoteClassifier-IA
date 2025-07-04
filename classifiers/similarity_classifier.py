from langchain_core.prompts import ChatPromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time, re

from .base import BaseQuoteClassifier

class ConstructSimilarityClassifier(BaseQuoteClassifier):
    """
    Classificador híbrido que utiliza embeddings para pré-seleção
    e LLM (via LangChain) para avaliação semântica contextual.
    Pode utilizar exemplos fornecidos pelo usuário.
    Retorna os dois constructos mais aderentes, com percentuais e justificativas.
    """
    def __init__(self, constructos, modelo="gpt-4o", peso_emb=0.4, peso_llm=0.6, escopo=None, exemplos=None):
        self.constructos = constructos
        self.peso_emb = peso_emb
        self.peso_llm = peso_llm
        self.escopo = escopo or "Sem escopo definido."
        self.exemplos = exemplos or []

        self.embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')

        self.llm = ChatOpenAI(model=modelo, temperature=0, request_timeout=30)
        self.parser = StrOutputParser()

        # Pré-calcula os embeddings dos constructos
        self.constructo_embeddings = {
            nome: self.embedder.encode(definicao)
            for nome, definicao in self.constructos.items()
        }

        if self.exemplos:
            # Prompt com exemplos (few-shot)
            self.prompt_template = FewShotPromptTemplate(
                examples=self.exemplos,
                example_prompt=ChatPromptTemplate.from_template(
                    'Quote: "{quote}"\nConstructo: {constructo}\nJustificativa: {justificativa}\n'
                ),
                prefix=("Contexto da pesquisa: {escopo}\n\n"
                        "Avalie o grau de correspondência entre a definição de um constructo teórico e um trecho de entrevista (quote).\n"
                        "Veja os exemplos abaixo para entender o padrão de resposta.\n"),
                suffix=("Agora avalie este novo trecho:\n"
                        "Definição do Constructo:\n{definicao}\n\n"
                        'Quote: "{quote}"\n'
                        "Responda com o seguinte formato:\n"
                        "Similaridade: <número>%\nJustificativa: <texto explicativo>"),
                input_variables=["escopo", "definicao", "quote"]
            )
        else:
            # Prompt padrão
            self.prompt_template = ChatPromptTemplate.from_template("""
                Contexto da pesquisa: {escopo}

                Avalie o grau de correspondência entre a definição de um constructo teórico e um trecho de entrevista (quote).
                Indique:
                - Um valor percentual de similaridade (0 a 100)
                - Uma justificativa breve explicando por que esse constructo se aplica ao quote

                Definição do Constructo:
                {definicao}

                Trecho de entrevista (quote):
                {quote}

                Responda com o seguinte formato:
                Similaridade: <número>%
                Justificativa: <texto explicativo>
            """)

    def classify(self, quotes):
        resultados = []
        justificativas = []
        tempos = []

        for idx, quote in enumerate(quotes, start=1):
            print(f"\n🔍 Classificando quote {idx}/{len(quotes)}: {quote[:60]}...")

            start = time.time()
            quote_emb = self.embedder.encode(quote)

            # Similaridade com constructos
            similaridades_emb = {
                nome: cosine_similarity([quote_emb], [emb])[0][0]
                for nome, emb in self.constructo_embeddings.items()
            }

            # Top 2 constructos
            top_constructos = sorted(similaridades_emb.items(), key=lambda x: x[1], reverse=True)[:2]

            constructos_resultado = []
            justificativas_quote = []

            for nome, sim_emb in top_constructos:
                definicao = self.constructos[nome]
                try:
                    sim_llm, just_llm = self._avaliar_similaridade_modelo(definicao, quote)
                except Exception as e:
                    sim_llm = 0.0
                    just_llm = f"⚠️ Erro ao consultar o modelo: {str(e)}"

                media = self._calcular_media(sim_emb * 100, sim_llm)
                constructos_resultado.append(f"{nome} ({media:.2f}%)")
                justificativas_quote.append(
                    f"→ {nome}:\n  Emb: {sim_emb * 100:.2f}%, LLM: {sim_llm:.2f}%, Média: {media:.2f}%\n  Justificativa: {just_llm}"
                )

            resultados.append(", ".join(constructos_resultado))
            justificativas.append("\n".join(justificativas_quote))
            tempos.append(time.time() - start)

        return resultados, justificativas, tempos

    def _avaliar_similaridade_modelo(self, definicao, quote):
        """Executa LLM para avaliar correspondência entre quote e definição"""
        chain = self.prompt_template | self.llm | self.parser
        resposta = chain.invoke({
            "escopo": self.escopo,
            "definicao": definicao,
            "quote": quote
        })

        match = re.search(r"Similaridade:\s*(\d+(?:\.\d+)?)%", resposta)
        just_match = re.search(r"Justificativa:\s*(.*)", resposta, re.DOTALL)

        similaridade = float(match.group(1)) if match else 0.0
        justificativa = just_match.group(1).strip() if just_match else "Justificativa não identificada."

        return similaridade, justificativa

    def _calcular_media(self, sim_emb, sim_llm):
        return self.peso_emb * sim_emb + self.peso_llm * sim_llm
