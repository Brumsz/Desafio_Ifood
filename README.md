# Desafio_Ifood

# 🤖 Agente de Suporte Inteligente com RAG Local & Memória

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-LCEL-green)
![RAG](https://img.shields.io/badge/AI-RAG%20Local-orange)
![LM Studio](https://img.shields.io/badge/LLM-Llama%203-purple)

> **Uma Prova de Conceito (POC) de um assistente virtual para suporte operacional (iFood), capaz de responder dúvidas sobre reembolsos e cancelamentos consultando uma base de conhecimento privada e mantendo o contexto da conversa.**

---

## 🧠 Sobre o Projeto

Este projeto consiste em um **Agente de IA Generativa** construído com **Python** e **LangChain**, utilizando a arquitetura **RAG (Retrieval-Augmented Generation)**.

O diferencial deste projeto é o foco em **Privacidade e Custo Zero**: ele foi desenhado para rodar 100% localmente, conectando-se a um LLM (Llama 3) hospedado no **LM Studio**, sem necessidade de enviar dados para APIs externas (como OpenAI).

Além disso, o agente possui **Memória Conversacional**, permitindo que o usuário faça perguntas de acompanhamento (ex: "E qual o prazo para isso?") sem perder o contexto.

---

## ✨ Principais Funcionalidades

* **📚 RAG (Busca Semântica):** O agente não alucina respostas. Ele consulta um arquivo CSV (`base_conhecimento_ifood_genai.csv`) antes de responder.
* **🔒 100% Local & Seguro:** Utiliza `HuggingFaceEmbeddings` para vetorização local e conecta-se ao `LM Studio` para inferência, garantindo que dados sensíveis não saiam da máquina.
* **🧠 Memória de Contexto (History Aware):** Implementado com **LangChain LCEL**, o agente reescreve perguntas ambíguas com base no histórico da conversa.
* **🛡️ Guardrails (Fallback):** Instruído via Engenharia de Prompt a não inventar informações. Se a resposta não estiver na base, ele sugere abertura de ticket.
* **⚙️ Arquitetura Moderna:** Código construído utilizando a sintaxe declarativa **LCEL (LangChain Expression Language)**.

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python
* **Orquestração:** LangChain (Community, Core, OpenAI)
* **Banco Vetorial:** ChromaDB
* **Embeddings:** Sentence-Transformers (`all-MiniLM-L6-v2`)
* **LLM Server:** LM Studio (Rodando Meta Llama 3)

---

## 🚀 Como Rodar o Projeto

### Pré-requisitos
1.  **Python** instalado.
2.  **LM Studio** instalado e configurado.

### Passo 1: Configurar o LM Studio
1.  Baixe e instale o [LM Studio](https://lmstudio.ai/).
2.  Na aba de busca, baixe o modelo **Meta Llama 3 Instruct** (versão `Q4_K_M` recomendada).
3.  Vá na aba de Servidor (ícone `<->`).
4.  Selecione o modelo baixado no topo.
5.  Clique em **Start Server**. Mantenha a porta padrão `1234`.

### Passo 2: Instalação do Código
Clone este repositório e entre na pasta:

git clone [https://github.com/SEU-USUARIO/NOME-DO-REPO.git](https://github.com/SEU-USUARIO/NOME-DO-REPO.git)
cd NOME-DO-REPO

Crie um ambiente virtual (Recomendado):
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate

Instale as dependências:
pip install -r requirements.txt

Passo 3: Execução
Com o LM Studio rodando ao fundo, execute o agente:
python main.py

🧪 Exemplos de Teste
Tente fazer estas perguntas para validar o funcionamento:

1. Teste de Conhecimento:

"O restaurante cancelou o pedido. O reembolso é automático?" Resposta esperada: Sim, explicando as condições baseadas no CSV.

2. Teste de Memória (Contexto):

"Como peço reembolso por item faltante?" Agente: Explica o processo. "E qual é o prazo para fazer isso?" Agente: Deve responder o prazo do reembolso, provando que entendeu o contexto.

3. Teste de Segurança (Fallback):

"Qual a capital da França?" Resposta esperada: O agente deve negar a resposta e sugerir um ticket, pois isso foge do escopo do iFood.

📂 Estrutura do Projeto
├── base_conhecimento_ifood_genai.csv  # Base de dados simulada
├── main.py                            # Código principal (RAG + Memória)
├── requirements.txt                   # Lista de dependências
└── README.md                          # Documentação

👤 Autor
Desenvolvido por [João Brum]

