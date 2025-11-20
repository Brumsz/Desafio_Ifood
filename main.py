import os

os.environ["OPENAI_API_KEY"] = "lm-studio"


def main():
    # Imports atualizados para evitar erros de reconhecimento
    try:
        from langchain_community.document_loaders import CSVLoader
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma
        from langchain_openai import ChatOpenAI
        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import PromptTemplate
    except ImportError as e:
        print(f"❌ Erro de Importação: {e}")
        print("Rode: pip install langchain langchain-community langchain-openai chromadb sentence-transformers")
        return

    #carregando o arquivo
    try:
        loader = CSVLoader(file_path="base_conhecimento_ifood_genai-exemplo.csv", encoding="utf-8")
        documentos = loader.load()
    except Exception:
        print("❌ Arquivo CSV não encontrado.")
        return

    #Aqui eu estou transformandos palavras em vetores de numeros para que o computador possa entender melhor
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(
        documents = documentos,
        embedding = embeddings
    )
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})


    #Estou chamando minha Ia que esta rodando localmente e deixando ela na temperatura 0 para não ter alucinações
    llm = ChatOpenAI(
        base_url = "http://127.0.0.1:1234/v1",
        api_key="lm-studio",
        temperature=0,
        model = "meta-llama-3-8b-instruct"
    )

    #Estou mostrando para Ia oque ela é e para que serve
    template = """
        Você é um agente interno que auxilia colaboradores a decidirem sobre reembolsos e cancelamentos.
        Sempre consulte a base de conhecimento antes de responder.
        Se não houver confiança suficiente, sugira validação manual ou abertura de ticket interno, em vez de gerar uma resposta incerta.

        Base de Conhecimento(Contexto):{context}

        Pergunta do Usuário:{question}

        Resposta:
        """

    PROMPT = PromptTemplate.from_template(template)

    #Aqui ele junta os textos do documento
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    #Aqui vai ser o chain
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )


    #interação 
    while True:
            pergunta = input("\n🤖 Pergunta: ")
            if pergunta.lower() in ["sair", "tchau", "exit"]:
                break
            
            print("⏳ Consultando a base...")
            try:
                resposta = chain.invoke(pergunta)
                print(f"\n💡 Resposta:\n{resposta}")
            except Exception as e:
                print(f"❌ Erro: O LM Studio está aberto e com o servidor ligado? Detalhe: {e}")

#Isso aqui é uma trava de segurança para saber quando esse arquivo é o main ou não
if __name__ == "__main__":
    main()