import os
import sys


from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory


os.environ["OPENAI_API_KEY"] = "lm-studio"

def main():
    #abrindo o arquivo, e fazendo o processo de RAG
    try:
        loader = CSVLoader(file_path="base_conhecimento_ifood_genai-exemplo.csv", encoding="utf-8")
        docs = loader.load()
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_db = Chroma.from_documents(documents=docs, embedding=embeddings)
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        print(f"❌ Erro ao carregar base: {e}")
        return

    # 2. LLM
    llm = ChatOpenAI(
        base_url="http://127.0.0.1:1234/v1", 
        api_key="lm-studio", 
        temperature=0,
        model="meta-llama-3-8b-instruct"
    )

    #reformula a pergunta
    prompt_reformulador = ChatPromptTemplate.from_messages([
        ("system", "Dada a conversa e a pergunta recente, reescreva a pergunta para ser completa. NÃO responda, apenas reescreva."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    #cria e limpa a resposta
    chain_reformuladora = prompt_reformulador | llm | StrOutputParser()

   
    def contextualize_q(input_dict):
        if input_dict.get("chat_history"):
            return chain_reformuladora
        else:
            return input_dict["input"]

    #falando quem o agente é, e dando memoria para ele
    prompt_resposta = ChatPromptTemplate.from_messages([
        ("system", """Você é um agente interno que auxilia colaboradores a decidirem sobre reembolsos e cancelamentos.
        Sempre consulte a base de conhecimento antes de responder.
        Se não houver confiança suficiente, sugira validação manual ou abertura de ticket interno, em vez de gerar uma resposta incerta.
        Contexto:{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    #juntando os textos do documento
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    #fazendo o rag_chain dando o contexto, depois formulando a resposta
    rag_chain = (
        RunnablePassthrough.assign(
            context=contextualize_q | retriever | format_docs
        )
        | prompt_resposta
        | llm
        | StrOutputParser()         
        | (lambda x: {"answer": x}) 
    )

    # criacão da memoria temporaria 
    store = {}

    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversational_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    print("(Digite 'sair' para fechar)")

    
    session_id = "usuario_teste"
    
    while True:
        pergunta = input("\n🤖 Pergunta: ")
        if pergunta.lower() in ["sair", "exit"]:
            break
        
        try:
            resposta = conversational_chain.invoke(
                {"input": pergunta},
                config={"configurable": {"session_id": session_id}}
            )
            print(f"💡 Resposta: {resposta['answer']}")
        except Exception as e:
            print(f"❌ Erro: {e}")

if __name__ == "__main__":
    main()