
# 📘 QuoteClassifier-IA – Assistente Inteligente para Pesquisa Qualitativa

> Ferramenta baseada em IA para auxiliar pesquisadores na classificação de trechos de entrevistas qualitativas em constructos teóricos.

---

## 🎯 Objetivo

Facilitar e acelerar a análise qualitativa de grandes volumes de dados textuais, fornecendo classificações automáticas **justificadas** com base em constructos definidos pelo usuário.

---

## 👥 Público-alvo

Pesquisadores qualitativos nas áreas de:

- Educação
- Ciências Sociais
- Psicologia
- Engenharia de Software
- Outras áreas com uso de teorias baseadas em constructos

---

## 🧰 Requisitos do Sistema

- Python 3.11+

**Dependências principais:**

```
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
```

---

## 🚀 Como Executar

1. Instale as dependências:

```bash
pip install -r requirements.txt
```

2. Configure sua chave da OpenAI:

Crie um arquivo `.env` na raiz do projeto com o conteúdo:

```
OPENAI_API_KEY=sk-sua-chave-aqui
```

3. Execute o sistema:

```bash
python app.py
```

A interface será aberta automaticamente no navegador via Gradio.

---

## 🗂️ Estrutura do Projeto

```
QuoteClassifier-IA/
├── app.py
├── main_controller.py
├── assistente_classificador.py
├── requirements.txt
├── .env.example
├── README.md / README.txt
│
├── classifiers/
│   ├── base.py
│   ├── embedding.py
│   ├── llm.py
│   ├── hybrid_classifier.py
│   ├── similarity_classifier.py
│
├── core/
│   ├── construct_loader.py
│   ├── dataset_loader.py
│   ├── pipeline.py
│
├── data/
│   ├── exemplo_constructos.xlsx
│   └── exemplo_quotes.xlsx
│
├── results/
```

---

## 🧪 Exemplo de Uso

1. Defina o escopo da pesquisa
2. Cadastre os constructos e definições
3. Faça upload da planilha com os quotes
4. Escolha o modelo de IA (LLM, embeddings ou híbrido)
5. Classifique os dados
6. Avalie os resultados com base em classificações manuais

---

## ⚙️ Tecnologias Utilizadas

- 🧠 LLMs (OpenAI GPT)
- LangChain
- LlamaIndex
- SentenceTransformers
- Gradio
- Pandas
- Python-dotenv

---

## 🧱 Arquitetura

- Modular (padrão MVC + princípios SOLID)
- Suporte a múltiplos classificadores
- Assistente de ajuda baseado em IA (chatbot com LLM)

---

## 📄 Licença

Projeto acadêmico e livre para fins educacionais. Para uso comercial, contate a autora.

---

## ✍️ Autoria

**Tatiane Ornelas**  
Projeto Final do Curso de Programação Aplicada à Pesquisa – PUC Minas
