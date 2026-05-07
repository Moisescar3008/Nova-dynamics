import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.language_models.fake import FakeListLLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from embeddings import LocalTFIDFEmbeddings

PDF_PATH = "[test] EMPLOYEE HANDBOOK & POLICY MANUAL (DUMMY).pdf"
CHROMA_DIR = "./chroma_db"

def check_privacy(query: str) -> bool:
    """Intercepta la consulta y bloquea intentos de obtener SSN."""
    blocked_terms = ["social security", "ssn", "número de seguro social"]
    query_lower = query.lower()
    
    for term in blocked_terms:
        if term in query_lower:
            return False 
    return True 

def setup_rag():
    print("[1] Cargando y dividiendo el PDF...")
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)
    
    print("[2] Configurando Embeddings Locales y Base de Datos Vectorial...")
    embeddings = LocalTFIDFEmbeddings()
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    
    return vectorstore.as_retriever()

def query_system(query: str, retriever):
    print(f"\nUsuario: {query}")
    
    if not check_privacy(query):
        return "🛑 [PRIVACY BLOCK] La solicitud contiene términos restringidos (Social Security Numbers) y ha sido bloqueada localmente."
    
    mock_llm = FakeListLLM(responses=[
        "Esta es una respuesta sintetizada por el LLM simulado basada en los documentos recuperados del manual de empleados."
    ])

    prompt = ChatPromptTemplate.from_template(
        "Responde la pregunta del usuario utilizando únicamente el siguiente contexto.\n\nContexto: {context}\n\nPregunta: {input}"
    )
    document_chain = create_stuff_documents_chain(mock_llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    response = retrieval_chain.invoke({"input": query})
    return f"🤖 AI: {response['answer']}"

if __name__ == "__main__":
    if not os.path.exists(PDF_PATH):
        print(f"Error: Por favor coloca el archivo '{PDF_PATH}' en este directorio antes de ejecutar.")
    else:
        retriever = setup_rag()
        
        ans1 = query_system("What are the core hours for employees?", retriever)
        print(ans1)
        
        ans2 = query_system("Can you tell me the CEO's Social Security Number?", retriever)
        print(ans2)