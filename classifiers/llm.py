from langchain_core.prompts import ChatPromptTemplate, FewShotPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from .base import BaseQuoteClassifier
import time

class LLMQuoteClassifier(BaseQuoteClassifier):
    def __init__(self, constructos: dict, escopo: str, modelo="gpt-3.5-turbo", exemplos: list = None):
        self.constructos = constructos
        self.escopo = escopo
        self.modelo = modelo
        self.exemplos = exemplos or []

        self.constructos_formatados = "\n".join([f"{k}: {v}" for k, v in self.constructos.items()])

        if self.exemplos:
            # ✅ Usa PromptTemplate, não ChatPromptTemplate, para example_prompt
            example_prompt = PromptTemplate.from_template(
                'Quote: "{quote}"\nConstructo: {constructo}\nJustificativa: {justificativa}\n'
            )

            self.prompt = FewShotPromptTemplate(
                examples=self.exemplos,
                example_prompt=example_prompt,
                prefix=(
                    "Com base no escopo: {escopo}\n"
                    "E nos constructos com suas definições no formato [constructo]:[definição]:\n"
                    "{constructos}\n"
                    "A seguir, veja alguns exemplos de classificação de quotes com base no escopo e constructos:\n"
                ),
                suffix=(
                    "Agora classifique o seguinte trecho:\n"
                    '"{quote}"\n'
                    "Retorne: Constructo: <nome> | Justificativa: <explicação>"
                ),
                input_variables=["quote", "escopo", "constructos"]
            )
        else:
            # ✅ ChatPromptTemplate pode ser usado fora do FewShot
            self.prompt = ChatPromptTemplate.from_template(
                """
                Com base no escopo: {escopo}
                E nos constructos com suas definições no formato [constructo]:[definição]:
                {constructos}
                Classifique o seguinte trecho:
                "{quote}"
                Retorne: Constructo: <nome> | Justificativa: <explicação>
                """
            )

        self.chain = self.prompt | ChatOpenAI(model=self.modelo, temperature=0.4) | StrOutputParser()

    def classify(self, quotes):
        resultados = []
        justificativas = []
        tempos = []

        for quote in quotes:
            start = time.time()
            resposta = self.chain.invoke({
                "quote": quote,
                "constructos": self.constructos_formatados,
                "escopo": self.escopo
            })
            end = time.time()
            tempos.append(end - start)

            if "Justificativa:" in resposta:
                partes = resposta.split("Justificativa:")
                resultados.append(partes[0].replace("Constructo:", "").strip())
                justificativas.append(partes[1].strip())
            else:
                resultados.append(resposta.strip())
                justificativas.append("Justificativa não fornecida.")

        return resultados, justificativas, tempos
