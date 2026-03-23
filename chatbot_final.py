from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import os
import streamlit as st
from dotenv import load_dotenv

# 1. This loads your local .env file if you are testing on your own computer
load_dotenv() 

# 2. This checks if you are running on Streamlit Cloud and pulls the secret key
if "OPENAI_API_KEY" in st.secrets:
    OPENAI_API_KEY= st.secrets["OPENAI_API_KEY"]


def create_chatbot(pdf_paths, chunk_size=1000, chunk_overlap=100):
    """
    Creates a chatbot using an OpenRouter-hosted GPT-4 model with FAISS retrieval from PDF documents.

    Args:
        pdf_paths (list): List of PDF file paths.
        chunk_size (int): Max characters per chunk.
        chunk_overlap (int): Overlap characters between chunks.

    Returns:
        RetrievalQA chatbot instance.
    """

    # 1️⃣ Read PDFs
    full_text = ""
    for path in pdf_paths:
        reader = PdfReader(path)
        for page in reader.pages:
            full_text += page.extract_text() + "\n"

    # 2️⃣ Split into manageable chunks
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(full_text)
    print(f"[INFO] Total chunks created: {len(chunks)}")

    # 3️⃣ Create embeddings (you can use OpenRouter-compatible ones)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=OPENAI_API_KEY,
    )

    # 4️⃣ Build FAISS vectorstore
    db = FAISS.from_texts(chunks, embedding=embeddings)

    # 5️⃣ Initialize OpenRouter LLM
    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",  # or "mistralai/mistral-7b-instruct"
        temperature=0,
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=OPENAI_API_KEY,
    )

    # 6️⃣ Create RetrievalQA pipeline
    chatbot = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        chain_type="stuff"
    )

    return chatbot


# Example usage
#if __name__ == "__main__":
#    bot = create_chatbot(["sample.pdf"])
#    query = "Summarize the main topic of this document."
#    print("💬 Answer:", bot.run(query))

