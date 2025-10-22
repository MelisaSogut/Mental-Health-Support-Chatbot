# create_knowledge_base.py 

import os
from dotenv import load_dotenv
from datasets import load_dataset
from langchain_core.documents import Document # LangChain'in temel Document sinifi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# ---  VERILERE GoRE GuNCELLEMELER ---
DATASET_NAME = "Ihssane123/Mental_Health_Dataset"
VECTOR_DB_PATH = "./chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
# ---------------------------------------------

def create_vector_store():
    # 1. Ortam degiskenlerini yukle
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY ortam degiskeni .env dosyasinda ayarlanmadi!")

    print(f"1. Dataset ({DATASET_NAME}) yukleniyor...")
    try:
        # Dataset'in sadece 'train' split'ini yukle
        hf_dataset = load_dataset(DATASET_NAME, split='train')
    except Exception as e:
        print(f"Dataset yuklenirken hata olustu: {e}")
        return

    # 2. Veri Setindeki ilgili sutunlari tek bir metin belgesinde birlestirme
    print("2. Veri sutunlari LangChain Document formatina donusturuluyor...")
    
    documents = []
    
    # Her satiri birlestirecek bir dongu
    for i, row in enumerate(hf_dataset):
        # Kaynak metni olustur
        content = (
           
           f"Detayli Bilgi : {row['input']}",
           f"Detayli Bilgi : {row['response']}"
        )
        
        # LangChain Document objesi olustur
        doc = Document(
            page_content=content,
            metadata={
                "source": row['response'], 
                "row_id": i
            }
        )
        documents.append(doc)

    print(f"Toplam olusturulan baslangic belgesi sayisi: {len(documents)}")

    # 3. Belgeleri parcalara ayir
    print("3. Belgeler parcalaniyor (chunking)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    texts = text_splitter.split_documents(documents)
    print(f"Toplam parca sayisi: {len(texts)}")

    # 4. Gomme (Embedding) Modelini Ayarla
    print("4. Embedding modeli baslatiliyor ve veriler vektorlere donusturuluyor...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    # 5. Parcalari gom ve Vektor Veritabanina (ChromaDB) kaydet
    print("5. ChromaDB'ye kaydetme islemi basliyor (Bu islem veri boyutuna gore zaman alabilir)...")
    vector_store = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH
    )

    vector_store.persist()
    print(f"6. Vektor veritabani basariyla olusturuldu ve {VECTOR_DB_PATH} dizinine kaydedildi.")

if __name__ == "__main__":
    create_vector_store()