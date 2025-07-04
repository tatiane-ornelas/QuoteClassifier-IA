from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
llm = ChatOpenAI(model="gpt-4", temperature=0.3)
prompt_base = ChatPromptTemplate.from_messages([
    ("system",
     """
     Você é um assistente especializado em ajudar pesquisadores a escolher o melhor classificador para análise de quotes em pesquisa qualitativa.
     
     A ferramenta possui os seguintes classificadores:
     - EmbeddingQuoteClassifier: rápido, baseado apenas em similaridade semântica (sem justificativas).
     - LLMQuoteClassifier: usa apenas modelo LLM (ex: GPT-3.5, GPT-4, Deepseek), fornece justificativas.
     - HybridQuoteClassifier: combina embeddings + LLM, indica o constructo mais aderente com justificativa.
     - ConstructSimilarityClassifier: combina embeddings + LLM, retorna os dois constructos mais próximos com justificativas e percentuais.

     Modelos disponíveis:
     - openai-3.5: mais barato, bom desempenho.
     - openai-4: mais preciso, custo médio.
     - gpt-4o: mais rápido e mais barato que o gpt-4.
     - deepseek-chat: alternativa não OpenAI, custo competitivo.
     
     Responda perguntas livres dos usuários, explicando diferenças entre os classificadores, sugerindo escolhas com base no número de quotes, necessidade de justificativa, tempo ou custo.
     Seja claro e direto. Quando for o caso, sugira um classificador e um modelo.
     """),
    ("user", "{pergunta}")
])
parser = StrOutputParser()
chain_assistente = prompt_base | llm | parser