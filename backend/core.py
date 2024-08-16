import os
from typing import Any, Dict, List
from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama


load_dotenv()
# pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
    docsearch = Pinecone.from_existing_index(index_name="langchain-doc-index", embedding=embeddings)
    chat = ChatOllama(model="llama3.1")

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=docsearch.as_retriever(), prompt=rephrase_prompt
    )

    qa = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain
    )
    result = qa.invoke(input={"input": query, "chat_history": chat_history})
    new_result = {
        "query": result["input"],
        "result": result["answer"],
        "source_documents": result["context"],
    }
    return new_result

if __name__ == "__main__":
    res = run_llm(query="What is a langchain chain?")
    print(res["result"])