from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import json
from datetime import datetime
from dotenv import load_dotenv

# LangChain (tetap seperti semula)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load env
load_dotenv()

app = Flask(__name__, static_folder='static', template_folder='templates')

# ================= PERSISTENT STORAGE =================

CHAT_HISTORY_FILE = "chat_history.json"
KB_FILE = "kb_aurex.json"


# ========= PER-SESSION CHAT HISTORY =========

def _load_all_histories():
    """
    Load seluruh history dari file.
    Struktur yang diharapkan:
    {
        "session_id_1": [...messages...],
        "session_id_2": [...messages...],
        ...
    }
    """
    if not os.path.exists(CHAT_HISTORY_FILE):
        return {}

    try:
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Backward compatibility: kalau masih berupa list lama → bungkus jadi satu session "global"
        if isinstance(data, list):
            return {"_legacy_global": data}

        if isinstance(data, dict):
            return data

        print("Warning: chat_history.json tidak dalam format yang diharapkan, reset ke dict kosong.")
        return {}
    except Exception as e:
        print(f"Error loading chat history: {e}")
        return {}


def _save_all_histories(data: dict):
    """Simpan seluruh history (semua session) ke file JSON."""
    try:
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving chat history: {e}")


def load_chat_history(session_id: str):
    """
    Ambil history untuk satu session_id tertentu.
    """
    all_histories = _load_all_histories()
    return all_histories.get(session_id, [])


def save_chat_history(session_id: str, messages):
    """
    Simpan history untuk satu session_id tertentu.
    """
    all_histories = _load_all_histories()
    all_histories[session_id] = messages
    _save_all_histories(all_histories)


def load_static_knowledge_base():
    """
    Load knowledge base statis dari file kb_aurex.json.
    Struktur file:
    {
        "kb": [
            {"id": "kb_0", "type": "general", "text": "..."},
            ...
        ]
    }
    """
    if not os.path.exists(KB_FILE):
        print(f"WARNING: {KB_FILE} not found, using empty knowledge base.")
        return []

    try:
        with open(KB_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        kb_items = data.get("kb", [])
        print(f">>> Loaded {len(kb_items)} KB items from {KB_FILE}")
        return kb_items
    except Exception as e:
        print(f"Error loading {KB_FILE}: {e}")
        return []


def add_to_vectorstore(user_message, ai_response):
    """
    Menambahkan percakapan baru ke ChromaDB sebagai knowledge tambahan (memori dinamis).
    Format: "User bertanya: [question]. Jawabannya: [answer]"
    """

    # ⛔ 1) Skip jika jawaban masih template penolakan yang tidak kita inginkan
    if ai_response:
        lower = ai_response.lower()
        if "maaf, sebagai MediSkill AI, fokus saya adalah" in lower:
            print(">>> Skip storing fallback / out-of-domain answer to vectorstore.")
            return

    try:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("WARNING: OPENAI_API_KEY not set, cannot add to vectorstore.")
            return

        embeddings = OpenAIEmbeddings(api_key=api_key)
        persist_dir = "./chroma_db"

        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )

        conversation_text = f"User bertanya: {user_message}. Jawabannya: {ai_response}"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        vectorstore.add_texts(
            texts=[conversation_text],
            metadatas=[{
                "source": "chat_history",
                "timestamp": timestamp,
                "type": "conversation"
            }]
        )

        print(f">>> Conversation added to vectorstore at {timestamp}")

    except Exception as e:
        print(f"Error adding to vectorstore: {e}")


# ================= RAG SETUP =================

def setup_rag_chain():
    """
    Inisialisasi ChromaDB + Conversational RAG Chain menggunakan kb_aurex.json.
    """
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("WARNING: OPENAI_API_KEY not found in environment")
        return None

    # 1. Load KB statis
    kb_items = load_static_knowledge_base()
    kb_texts = [item["text"] for item in kb_items]

    if not kb_texts:
        print("WARNING: Knowledge base kosong. RAG tetap dibuat, tapi tanpa konteks statis.")

    try:
        embeddings = OpenAIEmbeddings(api_key=api_key)
        persist_dir = "./chroma_db"

        # 2. Cek apakah DB Chroma sudah ada
        if os.path.exists(persist_dir):
            print(">>> Loading existing ChromaDB Vector Store...")
            vectorstore = Chroma(
                persist_directory=persist_dir,
                embedding_function=embeddings
            )
        else:
            if not kb_texts:
                print("ERROR: Tidak ada data KB untuk inisialisasi ChromaDB baru.")
                return None

            print(">>> Creating new ChromaDB Vector Store from kb_aurex.json...")
            metadatas = []
            for item in kb_items:
                metadatas.append({
                    "kb_id": item.get("id"),
                    "kb_type": item.get("type", "unknown"),
                    "source": "kb_aurex"
                })

            vectorstore = Chroma.from_texts(
                texts=kb_texts,
                embedding=embeddings,
                metadatas=metadatas,
                persist_directory=persist_dir
            )

        retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

        # 3. Model
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.4, api_key=api_key)

        # ========== HISTORY-AWARE RETRIEVER ==========
        contextualize_q_system_prompt = """
        Anda adalah MediSkill AI, asisten yang memahami konteks percakapan.
        Tugas Anda: memformulasikan ulang pertanyaan user agar lebih jelas untuk proses pencarian dokumen.
        - Gunakan riwayat percakapan jika perlu.
        - Jika pertanyaan sudah jelas, jangan diubah.
        - Jangan menambahkan informasi medis/softskill baru di sini, hanya perjelas maksud.

        Khusus untuk pertanyaan yang berkaitan dengan biaya, harga, tarif, atau paket layanan:
        - Pertahankan kata kunci seperti "biaya", "harga", "tarif", "paket", "promo", dan jenis layanan (misalnya konsultasi, fisioterapi, workshop).
        - Jika user tidak menyebut layanan dengan jelas, Anda boleh memperjelas secara netral,
            misalnya:
            "Berapa kisaran biaya untuk layanan konsultasi medis di klinik?"
            "Berapa kisaran biaya paket fisioterapi di cabang yang tersedia?"
        - Jangan mengubah pertanyaan biaya menjadi pertanyaan murni medis atau soft skill.
        - Pastikan hasil reformulasi tetap memicu pemanggilan interface 'fee_and_packages'
            jika konteks mengarah ke biaya layanan.

        Khusus untuk pertanyaan yang berkaitan dengan fasilitas atau layanan yang tersedia:
        - Pertahankan kata kunci seperti "fasilitas", "layanan", "cek fasilitas", "jenis layanan", "ada fasilitas apa saja".
        - Jika perlu memperjelas, Anda boleh menambahkan konteks netral, misalnya:
            "Fasilitas dan layanan apa saja yang tersedia di klinik secara umum?"
        - Jangan mengubah pertanyaan fasilitas menjadi topik lain yang tidak relevan
            (misalnya hanya gejala medis tanpa menyebut fasilitas atau layanan).
        - Ingat: pada tahap ini Anda HANYA memperjelas pertanyaan, bukan memberikan jawaban lengkap.
        """

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        # ========== QA CHAIN (SYSTEM PROMPT MedisSkill AI) ==========
        qa_system_prompt = """
        Anda adalah **MediSkill AI**, dokter profesional berbasis evidence-based medicine
        sekaligus pendamping pengembangan diri (soft skills & produktivitas).

        =============================
        🎛 MODE & KONTESK TOMBOL
        =============================
        - Mode aktif (dari UI): **{mode}**
            • "medis"  → fokus utama: Medis & Kesehatan
            • "soft"   → fokus utama: Soft Skills & Produktivitas

        - Tombol / topik aktif (dari UI): **{topic}**
            Contoh:
            • Medis: "Tanya Keluhan", "Rekomendasi Obat", "Edukasi Penyakit", "Gaya Hidup Sehat"
            • Soft:  "Time Management", "Habit Building", "Mental Wellness", "Goal Setting"

        Gunakan informasi **mode + topic** ini untuk:
        - Menentukan gaya jawaban
        - Menentukan apakah pertanyaan relevan atau di luar domain

        =============================
        📚 SUMBER PENGETAHUAN
        =============================
        1) Knowledge base terstruktur (medis & soft skills) yang diberikan ke Anda sebagai konteks:
            • Medis: general, internal_medicine, infectious_diseases, emergency_medicine,
                pediatrics, respiratory, gastroenterology, neurology, mental_health,
                pharmacology, prevention_lifestyle, dll.
            • Soft skills & produktivitas: softskills_productivity, communication_skills,
                learning_skills, mindset_discipline, dll.
            • Informasi biaya & paket layanan: entri bertipe "pricing" yang berisi penjelasan
                tentang struktur biaya, kisaran harga umum, dan cara menggunakan panel "Biaya & Paket".
        2) Memori percakapan sebelumnya dengan user (chat_history).

        Anda **WAJIB**:
        - Hanya menggunakan informasi yang ada di konteks (KB + memori).
        - Tidak mengarang fakta baru di luar konteks.
        - Jika informasi kurang → jelaskan bahwa datanya terbatas, dan lakukan eskalasi.

        ⚠️ CATATAN PENTING (ANTI JAWABAN TEMPLATE)
        ===========================================
        Dilarang menggunakan jawaban template seperti:
        - "Maaf, sebagai MediSkill AI, fokus kami adalah pada informasi medis dan kesehatan..."
        - atau kalimat serupa yang menyatakan bahwa MediSkill AI sama sekali tidak memiliki informasi
            tentang biaya layanan, fasilitas, atau layanan kesehatan spesifik,
            SELAMA di konteks tersedia entri KB yang relevan atau panel visual disediakan sistem.

        Jika informasi terbatas:
        - Tetap berikan gambaran umum berdasarkan KB (misalnya struktur biaya, jenis fasilitas, konsep layanan).
        - Anda boleh menjelaskan bahwa detail sangat spesifik (misalnya angka nominal pasti atau jadwal per cabang)
            perlu dikonfirmasi ke admin/CS atau fasilitas terkait.
        - Namun tetap berikan jawaban yang membantu, jangan menolak mentah-mentah.

        =============================
        🩺 ATURAN UNTUK MODE "MEDIS"
        =============================
        Jika mode = "medis":
        - Fokus menjelaskan keluhan, penyakit, obat, gaya hidup sehat, dan edukasi pasien.
        - Boleh menyinggung soft skills hanya sebagai pendukung (contoh: disiplin minum obat, manajemen stres).
        
        Tombol:
        - Jika topic mengarah ke "Tanya Keluhan":
            → Bantu user memahami kemungkinan penyebab **secara umum**, edukasi red flags,
                dan sarankan kapan harus ke tenaga medis.
        - Jika topic mengarah ke "Rekomendasi Obat":
            → Jelaskan prinsip penggunaan obat yang aman, cara kerja, efek samping umum.
            → Jangan memberikan dosis spesifik atau mengganti resep dokter.
            → Selalu sarankan konsultasi tenaga medis untuk keputusan akhir.
        - Jika topic mengarah ke "Edukasi Penyakit":
            → Jelaskan definisi, gejala, faktor risiko, dan pencegahan berdasar konteks.
        - Jika topic mengarah ke "Gaya Hidup Sehat":
            → Berikan saran nutrisi, tidur, olahraga, kebiasaan sehat, manajemen stres.

        Eskalasi Medis:
        1️⃣ Jika konteks cukup → Jawab jelas, terstruktur, dan edukatif.
        2️⃣ Jika konteks terbatas → Jelaskan apa yang ada di KB dulu, 
            lalu sarankan konsultasi tenaga medis / buat tiket konsultasi.
        3️⃣ Jika ada tanda gawat darurat (sesak berat, nyeri dada hebat, kejang, penurunan kesadaran, perdarahan hebat):
            → Sarankan segera ke IGD atau fasilitas kesehatan terdekat, jangan tunda.

        =============================
        💼 ATURAN UNTUK MODE "SOFT"
        =============================
        Jika mode = "soft":
        - Fokus pada: time management, habit building, mental wellness, goal setting,
            komunikasi efektif, cara belajar efisien, mindset & disiplin.
        - Jika pertanyaan menyentuh medis berat → arahkan user pindah ke mode medis atau ke tenaga medis.

        Contoh:
        - Time Management:
            → gunakan konsep to-do list, time blocking, Pomodoro, prioritas.
        - Habit Building:
            → gunakan konsep langkah kecil, konsistensi, habit stacking, lingkungan.
        - Mental Wellness:
            → gunakan konsep manajemen stres, journaling, tidur, batasan (boundaries).
        - Goal Setting:
            → gunakan konsep SMART goals, breakdown, review berkala.

        Eskalasi Soft:
        - Jika topik masih berkaitan dengan kehidupan sehat, produktivitas, belajar → jawab dari KB.
        - Jika topik di luar itu (misalnya finansial murni, politik, gosip, dll):
            → Jelaskan bahwa MediSkill AI fokus di kesehatan & pengembangan diri, sarankan mencari profesional terkait.

        =============================
        💰 ATURAN KHUSUS PERTANYAAN BIAYA & PAKET LAYANAN
        =============================
        Jika user menanyakan biaya, harga, tarif, atau paket layanan:

        - Gunakan entri KB bertipe **pricing** untuk memberikan gambaran umum
            (misalnya: struktur biaya, faktor penentu harga, jenis paket layanan).
        - Jangan menjawab "tidak memiliki informasi biaya" selama konteks mengandung entri pricing.
        - TEGASKAN bahwa:
            • Nominal spesifik bisa berbeda tiap cabang.
            • Detail lengkap dapat dilihat di **panel 'Biaya & Paket' yang muncul di bawah**.
        - Berikan jawaban yang ramah, jelas, dan selaras dengan interface visual.
        - Jelaskan cara membaca panel jika diperlukan (contoh: tabel, kategori layanan, paket).
        - Jika konteks benar-benar tidak memuat entri pricing:
            → Jelaskan secara jujur bahwa data nominal tidak tersedia,
                namun tetap berikan gambaran umum faktor yang menentukan biaya,
                dan arahkan untuk melihat panel jika tersedia.

        =============================
        🧭 INTEGRASI DENGAN QUICKPANEL GLOBAL
        =============================
        QuickPanel menyediakan tombol:
        - "Tanya Dokter"      → intent: ask_doctor
        - "Info Biaya"        → intent: ask_price
        - "Cek Fasilitas"     → intent: ask_facilities
        - "Pelatihan Soft Skills" → intent: ask_training
        - "Bantuan"           → intent: ask_help

        Jika pertanyaan tampak berasal dari QuickPanel (misalnya sama dengan template teks tersebut):

        - Untuk "Tanya Dokter":
            → Sambut user dengan hangat, jelaskan bahwa Anda siap membantu memahami keluhan.
            → Mintalah beberapa informasi dasar (lokasi keluhan, durasi, gejala penyerta).
            → Berikan edukasi umum dan red flags yang perlu diwaspadai.

        - Untuk "Info Biaya":
            → Ikuti aturan bagian BIAYA & PAKET di atas, dan arahkan ke panel Biaya & Paket jika tersedia.

        - Untuk "Cek Fasilitas":
            → Jelaskan secara ringkas jenis fasilitas dan layanan yang tersedia berdasarkan konteks KB.
            → Jika panel Fasilitas & Layanan muncul, arahkan user untuk melihat detail di panel tersebut.

        - Untuk "Pelatihan Soft Skills":
            → Jelaskan jenis pelatihan atau workshop yang mungkin tersedia (time management, manajemen stres, dsb).
            → Jika ada panel atau daftar pelatihan di bawah, arahkan user untuk melihat dan memilih yang sesuai.

        - Untuk "Bantuan":
            → Jelaskan secara singkat apa saja yang bisa MediSkill AI bantu (medis + soft skills).
            → Beri tips cara menggunakan mode, tombol topik, QuickPanel, dan panel-panel visual di bawah jawaban.

        =============================
        🏥 ATURAN KHUSUS FASILITAS & LAYANAN
        =============================
        Pertanyaan tentang fasilitas dan layanan (misalnya: 
        "Fasilitas apa saja yang tersedia?", 
        "Ada layanan apa saja di klinik ini?", 
        "Apakah ada layanan konseling atau fisioterapi?")
        DIANGGAP sebagai bagian dari domain MediSkill AI.

        Jika user menanyakan fasilitas atau layanan:
        - Gunakan entri KB bertipe "facility" atau informasi relevan lain dalam konteks.
        - Berikan gambaran umum jenis fasilitas yang mungkin tersedia
            (contoh: konsultasi dokter, laboratorium, fisioterapi, konseling psikologis, kelas edukasi, workshop soft skills).
        - Jelaskan bahwa detail sangat spesifik (misalnya nama alat, kapasitas ruangan, atau jadwal per fasilitas)
            dapat berbeda tiap cabang dan perlu dikonfirmasi langsung ke pengelola fasilitas.

        Jika panel "Fasilitas & Layanan" tampil di bawah jawaban:
        - Arahkan user untuk melihat panel tersebut sebagai ringkasan visual.
        - Boleh membantu menjelaskan cara membaca panel (kategori layanan, ikon, atau label yang muncul).

        Secara khusus, jika user menanyakan:
        - "Fasilitas apa saja yang tersedia?"
        - atau kalimat serupa yang secara langsung meminta daftar fasilitas,

        MAKA Anda WAJIB menyertakan kalimat berikut di dalam jawaban:

        "Untuk informasi lebih detail mengenai Fasilitas & Layanan, Anda dapat melihat ringkasan Fasilitas dan Layanan yang tersedia di bawah. Panel tersebut akan memberikan gambaran umum mengenai struktur layanan dan jenis layanan yang tersedia. Jika Anda memiliki pertanyaan lebih spesifik, silakan beri tahu saya! 😊"

        Kalimat tersebut boleh dilengkapi dengan penjelasan tambahan dari KB bertipe "facility",
        tetapi jangan dihilangkan, karena berfungsi sebagai arahan utama ke panel Fasilitas & Layanan.

        =============================
        🚫 PERTANYAAN DI LUAR DOMAIN
        =============================
        ❗ Pengecualian penting:
        - Pertanyaan tentang BIAYA LAYANAN dan FASILITAS/LAYANAN
            BUKAN termasuk luar domain, karena didukung oleh KB dan panel visual.

        Yang dianggap DI LUAR DOMAIN antara lain:
        - Politik dan isu pemerintahan.
        - Ekonomi dan finansial murni (investasi, saham, kripto, pajak, bisnis umum).
        - Gosip selebriti, hiburan, berita umum yang tidak terkait kesehatan/mental.
        - Teknologi umum yang tidak terkait kesehatan, kebugaran, atau produktivitas.
        - Topik lain yang sama sekali tidak berkaitan dengan:
            • medis & kesehatan,
            • fasilitas/layanan kesehatan,
            • soft skills, produktivitas, dan pengembangan diri.

        Jika pertanyaan termasuk luar domain:
        - JANGAN menjawab isi atau memberikan analisis detail tentang topik tersebut.
        - Berikan jawaban singkat dan jelas bahwa:
            MediSkill AI hanya difokuskan untuk membantu di bidang:
            (1) Medis & kesehatan, dan
            (2) Soft skills & produktivitas.
        - Anda boleh menambahkan kalimat singkat seperti:
            "Untuk topik tersebut, sebaiknya Anda berkonsultasi dengan ahli yang relevan atau mencari referensi khusus di luar sistem ini."

        =============================
        💬 GAYA BAHASA & KOMUNIKASI
        =============================

        - Gunakan Bahasa Indonesia yang:
        • Hangat
        • Profesional
        • Mudah dipahami
        • Tidak kaku
        • Tidak terlalu formal

        - Gunakan gaya bahasa:
        • Default: "aku – kamu" (santai profesional)
        • Jika user menggunakan "lu – gue / lo – gua", sistem WAJIB mengikuti gaya tersebut.
        • Jika user menggunakan bahasa formal, sistem ikut formal.
        • Jika user santai, sistem ikut santai.

        - Sistem harus MENYESUAIKAN:
        • Gaya ketikan user
        • Tingkat keseriusan user
        • Pilihan kata user

        - DILARANG menggunakan bahasa yang:
        • Terlalu kaku seperti dokumen resmi
        • Terlalu baku seperti jurnal akademik
        • Terlalu robotik

        - BOLEH menggunakan:
        • Emoji ringan (😊🙂💡✅) secukupnya
        • Bullet point jika membantu kejelasan
        • Bahasa percakapan alami

        - Prioritas utama:
        ✅ Nyaman dibaca
        ✅ Terasa manusiawi
        ✅ Terasa seperti asisten pribadi, bukan mesin

        =============================
        KONTEKS DARI KNOWLEDGE BASE:
        -----------------------------
        {context}
        """

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        print(">>> ChromaDB & Conversational RAG Initialized Successfully (MediSkill AI)")
        return rag_chain

    except Exception as e:
        print(f"Error initializing RAG: {e}")
        return None


# Initialize global RAG chain
rag_chain = setup_rag_chain()


# ================= Interface router integration (NEW but non-intrusive) =================
# We import choose_interfaces but do it in try/except so if the module isn't present,
# the rest of the app (RAG etc.) continues to function.
try:
    from interfaces.utils.interface_router import choose_interfaces
    _HAS_INTERFACE_ROUTER = True
    print(">>> Interface router loaded (choose_interfaces available).")
except Exception as e:
    choose_interfaces = None
    _HAS_INTERFACE_ROUTER = False
    print(f"WARNING: Could not import choose_interfaces: {e}")
    print(">>> Interface routing disabled; frontend will only receive textual responses.")


# ================= ROUTES =================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get_history', methods=['GET'])
def get_history():
    try:
        session_id = request.args.get("session_id", "").strip()
        if not session_id:
            return jsonify({"success": False, "error": "session_id is required"}), 400

        messages = load_chat_history(session_id)
        return jsonify({"success": True, "messages": messages})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/send_message', methods=['POST'])
def send_message():
    """
    Body JSON dari frontend:
    {
        "message": "...",
        "mode": "medis" | "soft",
        "topic": "Tanya Keluhan" | "Rekomendasi Obat" | ...,
        "quickpanel_intent": "ask_doctor" | "ask_training" | "ask_help" | ...,
        "session_id": "..."
    }
    """
    try:
        # 🔥 supaya kita bisa re-init rag_chain
        global rag_chain

        data = request.get_json()
        user_message = data.get("message", "").strip()
        mode = data.get("mode", "medis")           # default: medis
        topic = data.get("topic", "").strip()      # boleh kosong
        quickpanel_intent = data.get("quickpanel_intent")
        session_id = data.get("session_id", "").strip()

        if not session_id:
            return jsonify({"error": "session_id is required"}), 400

        if not user_message:
            return jsonify({"error": "Message cannot be empty"}), 400

        if not os.getenv("OPENAI_API_KEY"):
            return jsonify({"error": "OpenAI API key is not set!"}), 500

        # 🔥 kalau rag_chain tiba-tiba None (misal gagal init sebelumnya),
        # coba re-init lagi dulu
        if rag_chain is None:
            app.logger.warning("rag_chain is None, trying to reinitialize...")
            rag_chain = setup_rag_chain()

        # history sekarang tergantung session_id
        messages = load_chat_history(session_id)

        # ======================================================
        # ✅ 1. HANDLE QUICKPANEL INTENT DULU
        # ======================================================
        if quickpanel_intent in ["ask_doctor", "ask_training", "ask_help"]:
            if quickpanel_intent == "ask_doctor":
                answer = (
                    "Halo! 👋 Saya **MediSkill AI**, siap membantu Anda memahami keluhan.\n\n"
                    "Agar saya bisa membantu lebih tepat, boleh ceritakan dulu:\n"
                    "- Di bagian mana keluhan dirasakan?\n"
                    "- Sudah berapa lama keluhan ini muncul?\n"
                    "- Apakah ada gejala lain yang menyertai (demam, sesak napas, nyeri hebat, muntah, dll)?\n\n"
                    "Sebagai edukasi umum, segera ke IGD atau fasilitas kesehatan terdekat jika muncul tanda bahaya seperti:\n"
                    "- Sesak napas berat atau napas sangat cepat\n"
                    "- Nyeri dada hebat\n"
                    "- Penurunan kesadaran, kejang, atau kebingungan berat\n"
                    "- Perdarahan hebat yang sulit berhenti\n\n"
                    "Silakan ceritakan keluhan Anda, saya bantu jelaskan secara umum ya\n"
                    "Anda juga bisa berkonsultasi secara pribadi dengan klik tombol Panel yang ada dibawah 😊"
                )

            elif quickpanel_intent == "ask_training":
                answer = (
                    "Halo! 👋 Saya **MediSkill AI** untuk pelatihan soft skills, biasanya tersedia beberapa jenis topik, seperti:\n"
                    "- Manajemen waktu (time management)\n"
                    "- Manajemen stres & keseimbangan kerja-hidup\n"
                    "- Komunikasi efektif & kerja tim\n"
                    "- Produktivitas & habit building\n"
                    "- Goal setting & perencanaan karier\n\n"
                    "Jika di bawah ini muncul panel atau daftar pelatihan, silakan lihat dan pilih yang paling sesuai "
                    "dengan kebutuhan Anda. Kalau Anda ceritakan tujuan Anda (misalnya mau lebih fokus, mengelola stres, atau "
                    "meningkatkan performa kerja), saya bisa bantu merekomendasikan jenis pelatihan yang cocok 😊"
                )

            elif quickpanel_intent == "ask_help":
                answer = (
                    "Halo! 👋 Saya **MediSkill AI** bisa membantu Anda di dua area utama:\n\n"
                    "1️⃣ Medis & Kesehatan\n"
                    "- Menjelaskan keluhan/gejala secara umum\n"
                    "- Edukasi penyakit & gaya hidup sehat\n"
                    "- Penjelasan obat dan pemeriksaan medis secara umum\n\n"
                    "2️⃣ Soft Skills & Produktivitas\n"
                    "- Time management & prioritas kerja\n"
                    "- Manajemen stres & mental wellness\n"
                    "- Habit building, fokus, dan pengembangan diri\n\n"
                    "Cara menggunakan MediSkill AI:\n"
                    "- Gunakan tombol mode & topik di atas untuk mulai cepat.\n"
                    "- Gunakan QuickPanel (Tanya Dokter, Info Biaya, Cek Fasilitas, Pelatihan Soft Skills, Bantuan) "
                    "untuk langsung ke kebutuhan tertentu.\n"
                    "- Panel visual di bawah jawaban (misalnya Biaya & Paket atau Fasilitas) bisa Anda gunakan sebagai ringkasan informasi.\n\n"
                    "Silakan tuliskan kebutuhan Anda sekarang, saya bantu arahkan 😊"
                )

            # Simpan history seperti biasa (PER SESSION)
            timestamp = datetime.now().isoformat()
            new_messages = [
                {"is_user": True, "q": user_message, "timestamp": timestamp},
                {"is_user": False, "a": answer, "timestamp": timestamp},
            ]
            messages.extend(new_messages)
            save_chat_history(session_id, messages)

            # Opsional: simpan juga ke vectorstore dinamis
            add_to_vectorstore(user_message, answer)

            # Kalau mau, tetap boleh pilih interfaces (misal Info Biaya, dll)
            interfaces_payload = []
            if _HAS_INTERFACE_ROUTER and callable(choose_interfaces):
                try:
                    interfaces_payload = choose_interfaces(user_message)
                except Exception as e:
                    app.logger.warning(f"Interface router error: {e}")
                    interfaces_payload = []

            return jsonify({
                "success": True,
                "response": answer,
                "timestamp": timestamp,
                "interfaces": interfaces_payload
            })

        # ======================================================
        # ✅ 2. JIKA BUKAN QUICKPANEL → LANJUT KE RAG SEPERTI BIASA
        # ======================================================
        if rag_chain:
            # Ambil 20 pesan terakhir utk session ini
            recent_messages = messages[-20:]
            chat_history = []
            for msg in recent_messages:
                if msg.get("is_user"):
                    chat_history.append(HumanMessage(content=msg["q"]))
                else:
                    chat_history.append(AIMessage(content=msg["a"]))

            # Invoke RAG dengan tambahan mode & topic
            response = rag_chain.invoke({
                "input": user_message,
                "chat_history": chat_history,
                "mode": mode,
                "topic": topic
            })

            answer = response["answer"]

            # Simpan percakapan ke vectorstore dinamis
            add_to_vectorstore(user_message, answer)
        else:
            # Kalau setelah dicoba re-init tetap None
            answer = (
                "Maaf, sistem MediSkill AI sedang tidak dapat diinisialisasi. "
                "Silakan coba beberapa saat lagi atau hubungi admin."
            )

        # Simpan history ke file (PER SESSION)
        timestamp = datetime.now().isoformat()
        new_messages = [
            {"is_user": True, "q": user_message, "timestamp": timestamp},
            {"is_user": False, "a": answer, "timestamp": timestamp},
        ]
        messages.extend(new_messages)
        save_chat_history(session_id, messages)

        # --- choose_interfaces tetap seperti semula ---
        interfaces_payload = []
        if _HAS_INTERFACE_ROUTER and callable(choose_interfaces):
            try:
                interfaces_payload = choose_interfaces(user_message)
            except Exception as e:
                app.logger.warning(f"Interface router error: {e}")
                interfaces_payload = []

        response_json = {
            "success": True,
            "response": answer,
            "timestamp": timestamp,
            "interfaces": interfaces_payload
        }

        return jsonify(response_json)

    except Exception as e:
        app.logger.error(f"Error in send_message: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/get_interfaces', methods=['GET'])
def get_interfaces():
    """
    New endpoint: get interface JSON list.
    Query params:
        - q: optional user message text to choose relevant interfaces (ex: ?q=berapa+biaya)
    Returns:
        { "success": True, "interfaces": [...] }
    """
    try:
        if not _HAS_INTERFACE_ROUTER:
            return jsonify({"success": False, "error": "Interface router not available"}), 503

        q = request.args.get('q', '').strip()
        try:
            interfaces_payload = choose_interfaces(q)
        except Exception as e:
            app.logger.warning(f"choose_interfaces failed: {e}")
            interfaces_payload = []
        return jsonify({"success": True, "interfaces": interfaces_payload})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/reset', methods=['POST'])
def reset():
    """
    Reset hanya chat history UNTUK SATU SESSION,
    tidak menghapus vectorstore / KB statis.
    """
    try:
        data = request.get_json() or {}
        session_id = data.get("session_id", "").strip()
        if not session_id:
            return jsonify({"success": False, "error": "session_id is required"}), 400

        save_chat_history(session_id, [])
        return jsonify({"success": True, "message": "Chat history reset successfully"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/clear_all', methods=['POST'])
def clear_all():
    """
    Menghapus:
    - folder ./chroma_db (memori dinamis / vektor)

    TIDAK menghapus:
    - chat_history.json (riwayat percakapan tetap aman)

    Lalu mencoba re-init RAG dengan kb_aurex.json.
    Jika re-init gagal, rag_chain diset None dan akan dicoba lagi
    saat ada request /send_message berikutnya.
    """
    try:
        import shutil
        global rag_chain

        persist_dir = "./chroma_db"
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)

        # ❗ Penting: jangan pakai chain lama lagi, karena DB-nya sudah dihapus
        rag_chain = None

        # Coba re-init sekarang (opsional)
        new_chain = setup_rag_chain()
        if new_chain is not None:
            rag_chain = new_chain
            msg = "Dynamic memory cleared successfully and RAG reinitialized."
        else:
            msg = (
                "Dynamic memory cleared. RAG akan diinisialisasi ulang otomatis "
                "saat Anda mengirim pesan berikutnya."
            )
            app.logger.warning(msg)

        return jsonify({"success": True, "message": msg})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

