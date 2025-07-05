
=============================
QuoteClassifier-IA – Assistente Inteligente para Pesquisa Qualitativa
=============================

Este projeto é um assistente inteligente que auxilia pesquisadores na classificação automática de trechos de entrevistas (quotes) em constructos teóricos previamente definidos. O sistema aplica Inteligência Artificial Generativa (como LLMs da OpenAI e embeddings semânticos) e foi desenvolvido com uma arquitetura modular baseada nos princípios SOLID e padrão MVC.

-----------------------------
Objetivo
-----------------------------
Facilitar e acelerar a análise qualitativa de grandes volumes de dados textuais, fornecendo classificações automáticas justificadas com base em constructos fornecidos pelo usuário.

-----------------------------
Público-alvo
-----------------------------
Pesquisadores qualitativos nas áreas de:
- Educação
- Ciências Sociais
- Psicologia
- Engenharia de Software
- Áreas que utilizam teorias estruturadas em constructos

-----------------------------
Requisitos do Sistema
-----------------------------
Python 3.11+

Dependências (requirements.txt):

gradio==5.35.0
langchain_core==0.3.68
langchain_openai==0.3.27
llama_index==0.12.46
matplotlib==3.10.3
pandas==2.3.0
pytest==8.3.4
python-dotenv==1.1.1
scikit_learn==1.7.0
sentence_transformers==4.1.0
views==0.3

-----------------------------
Como Executar
-----------------------------
1. Instale as dependências:
   pip install -r requirements.txt

2. Configure sua chave da OpenAI:
   Crie um arquivo .env na raiz do projeto com o conteúdo:
   OPENAI_API_KEY=sk-sua-chave-aqui

3. Execute a aplicação:
   python app.py

A interface será aberta automaticamente no navegador via Gradio.

-----------------------------
Estrutura do Projeto
-----------------------------
QuoteClassifier-IA/
├── app.py                         # Interface principal Gradio (executa o sistema)
├── main_controller.py            # Controlador do fluxo entre as etapas
├── assistente_classificador.py   # Chatbot com LLM para ajuda interativa
├── requirements.txt              # Dependências do projeto
├── .env.example                  # Exemplo de configuração da OpenAI API
├── README.md / README.txt        # Este manual

├── classifiers/                  # Módulos de classificação
│   ├── base.py
│   ├── embedding.py
│   ├── llm.py
│   ├── hybrid_classifier.py
│   ├── similarity_classifier.py
│   └── __init__.py

├── core/
│   ├── construct_loader.py
│   ├── dataset_loader.py
│   ├── pipeline.py
│   └── __init__.py

├── data/
│   ├── exemplo_constructos.xlsx
│   └── exemplo_quotes.xlsx

├── results/                      # Saída automática das classificações e avaliações

-----------------------------
Exemplo de uso
-----------------------------
1. Defina o escopo da sua pesquisa
2. Cadastre seus constructos e definições
3. Carregue sua planilha de quotes
4. Escolha o modelo de IA para classificação (LLM, embeddings ou híbrido)
5. Execute a classificação
6. Avalie os resultados com base na sua classificação manual

-----------------------------
Tecnologias Utilizadas
-----------------------------
- Python 3.11+
- Gradio
- LangChain
- LlamaIndex
- OpenAI API
- SentenceTransformers
- Pandas
- dotenv

-----------------------------
Arquitetura
-----------------------------
- Modular (MVC + SOLID)
- Suporte a múltiplos classificadores
- Chatbot integrado com IA

-----------------------------
Licença
-----------------------------
Projeto acadêmico e livre para fins educacionais. Para uso comercial, contate a autora.

-----------------------------
Autoria
-----------------------------
Tatiane Ornelas – Projeto Final do Curso de Programação Aplicada à Pesquisa – PUC Minas.
