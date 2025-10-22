#🧠 Mental Health Support Chatbot

(RAG - Retrieval Augmented Generation Temelli Bir Zihinsel Sağlık Asistanı)

##📘 Projenin Amacı

Bu proje, zihinsel sağlık (mental health) konularında bilgi ve destek sunan bir RAG (Retrieval Augmented Generation) temelli sohbet asistanı geliştirmeyi amaçlamaktadır.
Kullanıcılar, kaygı, depresyon, stres gibi psikolojik konularda ingilizce soru sorabilir; chatbot ise önceden gömülmüş bilgi tabanını kullanarak doğru, empatik ve bilgilendirici ingilizce cevaplar üretir.

Asistan, Google Gemini 2.0 Flash modeliyle desteklenmekte ve yanıtlarını doğal, sade ve destekleyici bir dille vermektedir.

⚠️ Bu uygulama yalnızca bilgilendirme amaçlıdır.
Profesyonel bir psikolojik teşhis veya tedavi aracı değildir.

##📂 Veri Seti Hakkında

Kullanılan veri seti: Ihssane123/Mental_Health_Dataset
Lisans: MIT Lisansı
Kaynak: Hugging Face Datasets
Veri Linki:https://huggingface.co/datasets/Ihssane123/Mental_Health_Dataset/viewer/default/train?views%5B%5D=train&row=65


İçerik: Gerçek kullanıcıların zihinsel sağlıkla ilgili soruları ve uygun destekleyici yanıtları içerir.

Amaç: Chatbot’un eğitim verisi olarak kullanılıp, RAG tabanlı bilgi tabanının oluşturulmasıdır.

Veri seti doğrudan Hugging Face’den çekildiği için repoda saklanmamıştır.

##🧩 Kullanılan Yöntemler ve Mimariler
🔹 RAG (Retrieval Augmented Generation) Mimarisi

Model, iki aşamalı bir yapı izler:

Retrieval (Bilgi Getirme):
ChromaDB üzerinde saklanan mental health belgelerinden en alakalı içerikler bulunur.
Gömme işlemi için Google Generative AI Embeddings (text-embedding-004) modeli kullanılmıştır.

Generation (Cevap Üretimi):
Elde edilen bilgiler, Gemini 2.0 Flash modeliyle birleştirilerek doğal bir yanıt oluşturulur.

##🔹 Kullanılan Teknolojiler
Teknoloji	Kullanım Amacı
LangChain	RAG zincirinin yönetimi
ChromaDB	Vektör veritabanı oluşturma ve sorgulama
Google Gemini 2.0 Flash	LLM (Büyük Dil Modeli) ile doğal yanıt üretimi
Google Generative AI Embeddings	Metinleri vektör uzayına gömme
Streamlit	Web tabanlı kullanıcı arayüzü
dotenv	Ortam değişkenlerini yönetme (.env dosyası)
Hugging Face Datasets	Mental Health veri setini yükleme


##⚙️ Kurulum ve Çalıştırma Kılavuzu
###1️⃣ Depoyu Klonla
git clone https://github.com/kullaniciadi/mental-health-chatbot.git
cd mental-health-chatbot

###2️⃣ Sanal Ortam (venv) Oluştur
python -m venv .venv
.venv\Scripts\activate   # Windows

###3️⃣ Gerekli Kütüphaneleri Kur
pip install -r requirements.txt

###4️⃣ .env Dosyasını Oluştur

Proje dizininde bir .env dosyası oluşturup içine kendi Google API anahtarını ekle:

GOOGLE_API_KEY=senin_api_keyin

###5️⃣ Bilgi Tabanını (Vector Store) Oluştur
python create_knowledge_base.py


Bu adım, Hugging Face veri setini indirir, metinleri parçalara böler ve ./chroma_db klasörüne kaydeder.

###6️⃣ Web Arayüzünü Başlat
streamlit run website.py

###💻 Alternatif (Terminal üzerinden Chatbot)
python chatbot.py

##🌐 Web Arayüzü & Kullanım Kılavuzu

Proje çalıştırıldıktan sonra terminalde şu bağlantı görünür:

Local URL: http://localhost:8501


Bu bağlantıya giderek sohbet ekranına ulaşabilirsiniz.

##Kullanım Adımları:

Sorunuzu yazın (örneğin: “Can people with mental illness recover?”).

Chatbot veritabanından ilgili bilgileri çeker.

Empatik, kısa ve bilgilendirici bir yanıt üretir.

##Arayüz Özellikleri:

🌙 Koyu tema (Dark Mode - Sakura tarzı)


🧠 Empatik dil modeli yanıtları

⚠️ Uyarı: Tıbbi tavsiye niteliği taşımadığı belirtilmiştir

##🧠 Proje Mimarisi
📦 mental-health-chatbot
 ┣ 📜 create_knowledge_base.py   → Veri setinden vektör veritabanı oluşturur
 ┣ 📜 website.py                 → Streamlit tabanlı web arayüzü
 ┣ 📜 chatbot.py                 → Terminal tabanlı chatbot sürümü
 ┣ 📜 requirements.txt           → Gerekli Python kütüphaneleri
 ┣ 📜 .env                       → Google API anahtarı
 ┗ 📁 chroma_db/                 → Oluşturulan vektör veritabanı

##📊 Sonuçlar ve Gözlemler

Model, mental health konularında kullanıcıya doğru ve destekleyici bilgiler sunmaktadır.

Yanıtlar, insan odaklı ve empatik biçimde düzenlenmiştir.

RAG yaklaşımı sayesinde chatbot yalnızca öğrenilmiş kalıpları değil, gerçek bilgi tabanından gelen verileri kullanarak yanıt üretmektedir.

##🚀 Canlı Demo


##💬 Geliştirici Notu

Bu proje, yapay zekânın toplumsal fayda potansiyelini göstermek amacıyla geliştirilmiştir.🌿
