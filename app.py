import streamlit as st
import os
import time

# --- IMPORTS DO SEU CÓDIGO ORIGINAL ---
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory

# Imports do LCEL (Núcleo)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Suporte iFood IA",
    page_icon="🍔",
    layout="centered"
)

st.title("🍔 Assistente de Suporte iFood")
st.caption("Powered by RAG Local + Llama 3")

# --- CACHE DO SISTEMA (PARA NÃO RECARREGAR TUDO A CADA CLIQUE) ---
# O Streamlit roda o código todo de novo a cada interação. 
# Usamos @st.cache_resource para carregar o banco e a IA só uma vez.

@st.cache_resource
def configurar_agente():
    os.environ["OPENAI_API_KEY"] = "lm-studio"
    print("🔄 Carregando IA e Base de Dados...")

    # 1. Carregar Dados
    loader = CSVLoader(file_path="base_conhecimento_ifood_genai-exemplo.csv", encoding="utf-8")
    docs = loader.load()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(documents=docs, embedding=embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # 2. LLM
    llm = ChatOpenAI(
        base_url="http://127.0.0.1:1234/v1", 
        api_key="lm-studio", 
        temperature=0,
        model="meta-llama-3-8b-instruct"
    )

    # 3. Montagem da Chain (Manual/LCEL)
    
    # Reformulador
    prompt_reformulador = ChatPromptTemplate.from_messages([
        ("system", "Reescreva a pergunta do usuário para ser independente baseada no histórico."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    chain_reformuladora = prompt_reformulador | llm | StrOutputParser()

    def contextualize_q(input_dict):
        if input_dict.get("chat_history"): return chain_reformuladora
        else: return input_dict["input"]

    # Respondedor
    prompt_resposta = ChatPromptTemplate.from_messages([
        ("system", """Você é um agente interno que auxilia colaboradores a decidirem sobre reembolsos e cancelamentos.
        Sempre consulte a base de conhecimento antes de responder.
        Se não houver confiança suficiente, sugira validação manual ou abertura de ticket interno, em vez de gerar uma resposta incerta.
        Contexto:{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    def format_docs(docs): return "\n\n".join(d.page_content for d in docs)

    # A GRANDE CADEIA
    rag_chain = (
        RunnablePassthrough.assign(
            context=contextualize_q | retriever | format_docs
        )
        | prompt_resposta
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# Inicia a inteligência (apenas na primeira vez)
try:
    rag_chain = configurar_agente()
except Exception as e:
    st.error(f"Erro ao conectar com a IA. Verifique se o LM Studio está rodando! Erro: {e}")
    st.stop()

# --- GERENCIAMENTO DE MEMÓRIA (SESSÃO) ---

# Cria o histórico VISUAL (para aparecer na tela)
if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

# Cria o histórico TÉCNICO (para a IA lembrar)
if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session_id: str):
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# Monta o Agente com Memória
conversational_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

# --- INTERFACE DO CHAT ---

# 1. Desenha as mensagens antigas na tela
for msg in st.session_state.mensagens:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 2. Caixa de entrada do usuário
if pergunta := st.chat_input("Digite sua dúvida..."):
    # Adiciona a pergunta na tela
    st.session_state.mensagens.append({"role": "user", "content": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)

    # Gera a resposta
    with st.chat_message("assistant"):
        mensagem_placeholder = st.empty()
        mensagem_placeholder.markdown("▌ Pensando...")
        
        try:
            # Chama o Agente
            resposta = conversational_chain.invoke(
                {"input": pergunta},
                config={"configurable": {"session_id": "sessao_padrao"}}
            )
            
            # Atualiza o placeholder com a resposta final
            mensagem_placeholder.markdown(resposta)
            
            # Salva no histórico visual
            st.session_state.mensagens.append({"role": "assistant", "content": resposta})
            
        except Exception as e:
            mensagem_placeholder.markdown(f"❌ Erro: {e}")