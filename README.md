# 🧠 MediSkill AI
**Asisten AI untuk Medis, Kesehatan, Soft Skills & Produktivitas berbasis Flask + LangChain + RAG (ChromaDB).**

MediSkill AI adalah platform asisten virtual berbasis web yang dirancang untuk membantu pengguna dalam:
- 🩺 Edukasi medis & kesehatan berbasis evidence-based medicine  
- 💼 Pengembangan soft skills & produktivitas  
- 📊 Akses informasi fasilitas, layanan, biaya, dan pelatihan  
- ⚡ Interaksi cepat melalui QuickPanel dan panel visual interaktif  

Project ini dibangun sebagai **produk AI asisten modern yang siap digunakan publik**.

---

## ✨ Fitur Utama

- ✅ Chat AI berbasis **LangChain + OpenAI**
- ✅ **Retrieval-Augmented Generation (RAG)** dengan ChromaDB
- ✅ Mode:
  - **Medis & Kesehatan**
  - **Soft Skills & Produktivitas**
- ✅ **QuickPanel**:
  - Tanya Dokter  
  - Info Biaya  
  - Cek Fasilitas  
  - Pelatihan Soft Skills  
  - Bantuan
- ✅ Panel visual dinamis:
  - Biaya & Paket
  - Fasilitas & Layanan
  - Lokasi
  - Program Pelatihan
- ✅ Penyimpanan riwayat chat (JSON)
- ✅ Memori dinamis percakapan melalui Vector Database
- ✅ UI modern dan user-friendly
- ✅ Siap deploy online menggunakan Flask

---

## 🏗️ Arsitektur Teknologi

- **Frontend**:  
  - HTML, CSS, JavaScript (Vanilla)
- **Backend**:  
  - Flask (Python)
- **AI Engine**:
  - OpenAI API
  - LangChain
- **Vector Database**:
  - ChromaDB
- **Environment Management**:
  - python-dotenv

---

## 📁 Struktur Project

```text
MEDISKILL_AI/
├── assets/
│   └── icons/
├── chroma_db/                # Vector memory (tidak diupload ke GitHub)
├── interfaces/
│   ├── global/
│   │   └── global_quickpanel.json
│   ├── special/
│   │   ├── facilities_grid.json
│   │   ├── fee_and_packages.json
│   │   ├── location_directory.json
│   │   └── training_programs.json
│   └── utils/
│       ├── intent_rules.py
│       ├── interface_router.py
│       └── json_loader.py
├── static/
│   └── images/
├── templates/
│   └── index.html
├── index.py
├── chat_history.json         # Sebaiknya tidak diupload
├── kb_aurex.json
├── requirements.txt
├── .env                      # Tidak diupload
└── README.md
