# chatbot.py
 
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
 
# --- Ayarlar ---
VECTOR_DB_PATH = "./chroma_db"
 
def main():
    # 1. Ortam degiskenlerini yukle (.env icindeki GOOGLE_API_KEY)
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY .env dosyasinda bulunamadi!")
 
    print("1️⃣ Embedding modeli yukleniyor...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
 
    print("2️⃣ Chroma veritabani yukleniyor...")
    vector_store = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings
    )
 
    # Retriever (bilgi getiren araci)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
 
    print("3️⃣ Google Gemini modeli hazirlaniyor...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",  
        temperature=0.2,
     convert_system_message_to_human=True
     
     
    )
 
    print("4️⃣ RetrievalQA zinciri olusturuluyor...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",  # tum chunk’lari birlestirip modele verir
        return_source_documents=True
    )
 
    print("\n💬 Chatbot is ready to help you! How can I help you with your questions about mental health?\n")
 
    while True:
        query = input("👤 You: ")
        if query.lower() in ["exit", "quit", "q", "cik", "cikis"]:
            print("👋 See you!")
            break
 
        result = qa_chain({"query": query})
        print("\n🤖 Bot:")
        print(result["result"])
        print("\n📚 Resources:")
        for doc in result["source_documents"]:
            print(" -", doc.metadata.get("source"))
        print("-" * 50)
 
if __name__ == "__main__":
    main()
 

 
