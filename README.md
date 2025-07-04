# 🧠 QuoteClassifier-IA – Assistente Inteligente para Pesquisa Qualitativa

Este projeto é um assistente inteligente que auxilia pesquisadores na **classificação automática de trechos de entrevistas (quotes)** em **constructos teóricos** previamente definidos. O sistema aplica **Inteligência Artificial Generativa** (como LLMs da OpenAI e embeddings semânticos) e foi desenvolvido com uma arquitetura modular baseada nos princípios **SOLID** e padrão **MVC**.

---

## 🎯 Objetivo

Facilitar e acelerar a análise qualitativa de grandes volumes de dados textuais, fornecendo classificações automáticas justificadas com base em constructos fornecidos pelo usuário.

---

## 👥 Público-alvo

Pesquisadores qualitativos nas áreas de:
- Educação
- Ciências Sociais
- Psicologia
- Engenharia de Software
- Áreas que utilizam teorias estruturadas em constructos

---

## 🧱 Estrutura do Projeto

```
QuoteClassifier-IA/
├── app.py                         # Interface principal Gradio (executa o sistema)
├── main_controller.py            # Controlador do fluxo entre as etapas
├── assistente_classificador.py   # Chatbot com LLM para ajuda interativa
├── requirements.txt              # Dependências do projeto
├── .env.example                  # Exemplo de configuração da OpenAI API
├── README.md                     # Este documento

├── classifiers/                  # Módulos de classificação
│   ├── base.py                   # Classe base para classificadores
│   ├── embedding.py              # Classificador baseado em embeddings
│   ├── llm.py                    # Classificador baseado em LLMs (GPT)
│   ├── hybrid_classifier.py      # Classificador híbrido (LLM + Embeddings)
│   ├── similarity_classifier.py  # Similaridade direta quote-constructo
│   └── __init__.py

├── core/                         # Núcleo de carregamento e orquestração
│   ├── construct_loader.py       # Leitura dos constructos (manual ou planilha)
│   ├── dataset_loader.py         # Leitura dos quotes
│   ├── pipeline.py               # Encadeia o processo de classificação
│   └── __init__.py

├── data/
│   ├── exemplo_constructos.xlsx  # Constructos de exemplo
│   └── exemplo_quotes.xlsx       # Quotes de exemplo

├── results/                      # Saída automática das classificações e avaliações
```

---

## 🚀 Como Executar

1. **Instale as dependências:**
```bash
pip install -r requirements.txt
```

2. **Configure sua chave da OpenAI:**

Crie um arquivo `.env` na raiz do projeto (ou copie `.env.example`):

```env
OPENAI_API_KEY=sk-sua-chave-aqui
```

3. **Execute a aplicação:**
```bash
python app.py
```

A interface será aberta automaticamente no navegador usando Gradio.

---

## 🧪 Exemplo de uso

1. Defina o escopo da sua pesquisa
2. Cadastre seus constructos e definições
3. Carregue sua planilha de quotes
4. Escolha o modelo de IA para classificação (LLM, embeddings ou híbrido)
5. Execute a classificação
6. Avalie os resultados com base na sua classificação manual

---

## 🤖 Tecnologias Utilizadas

- **Python 3.11+**
- **Gradio** – Interface gráfica para web
- **LangChain** – Orquestração de LLMs
- **LlamaIndex** – Indexação semântica
- **OpenAI API** – Modelos GPT-3.5 e GPT-4
- **SentenceTransformers** – Vetorização e similaridade
- **Pandas** – Manipulação de dados
- **dotenv** – Variáveis de ambiente

---

## 🧠 Arquitetura e Padrões

- Arquitetura **modular** (MVC + SOLID)
- Separação clara entre lógica, dados e interface
- Suporte a múltiplos classificadores
- Chatbot integrado para suporte contextual

---

## 📂 Resultados

Todos os arquivos gerados (classificações automáticas, avaliações de acurácia, etc.) são salvos automaticamente na pasta `/results/`.

---

## 📄 Licença

Este projeto é de uso acadêmico e livre para estudos. Para uso comercial, consulte o autor.

---

## 👩‍💻 Autoria

Desenvolvido por Tatiane Ornelas como projeto final do curso de Programação Aplicada à Pesquisa – PUC Minas.

