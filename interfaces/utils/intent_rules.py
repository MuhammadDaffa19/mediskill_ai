# interfaces/utils/intent_rules.py
import re

INTENT_KEYWORDS = {
    # =======================
    # BIAYA / HARGA / PAKET
    # =======================
    "ask_price": [
        "harga", "biaya", "tarif", "paket", "cost", "price",
        "berapa biaya", "berapa harga", "berapa tarif", "info biaya"
    ],

    # =======================
    # FASILITAS / LAYANAN
    # =======================
    "ask_facilities": [
        "fasilitas", "layanan", "peralatan", "konseling", "terapi", "workshop",
        "apa saja fasilitas", "ada fasilitas", "fasilitas apa", "cek fasilitas"
    ],

    # =======================
    # LOKASI / CABANG
    # =======================
    "ask_location": [
        "lokasi", "alamat", "cabang", "dekat saya", "where", "di mana", "dimana",
        "ada di mana", "lokasi mana", "alamatnya dimana", "alamat klinik"
    ],

    # =======================
    # TANYA DOKTER / KELUHAN
    # =======================
    "ask_doctor": [
        "tanya dokter", "konsultasi dokter", "konsultasi medis", "curhat keluhan",
        "saya sakit", "saya kurang enak badan", "keluhan saya", "butuh saran dokter",
        "check up", "periksa", "tanya penyakit", "tanya gejala"
    ],

    # =======================
    # PELATIHAN / WORKSHOP SOFT SKILLS
    # =======================
    "ask_training": [
        "pelatihan", "training", "workshop", "kelas soft skill", "kelas soft skills",
        "kursus produktivitas", "kelas produktivitas", "webinar", "coaching", "mentoring"
    ],

    # =======================
    # BANTUAN / HELP / CARA PAKAI
    # =======================
    "ask_help": [
        "bantuan", "help", "butuh bantuan", "cara pakai aurex", "cara menggunakan aurex",
        "apa yang bisa kamu lakukan", "fitur apa saja", "panduan", "tutorial", "cara menggunakan fitur"
    ],
}


def normalize(text: str) -> str:
    if not text:
        return ""
    t = text.lower()
    # remove extra whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


def extract_keywords(text: str):
    t = normalize(text)
    found = set()
    for kws in INTENT_KEYWORDS.values():
        for kw in kws:
            if kw in t:
                found.add(kw)
    return list(found)


def detect_intent(text: str):
    """
    Returns list of detected intents, e.g. ['ask_price','ask_location'].
    Multi-intent supported.
    """
    t = normalize(text)
    detected = []
    for intent, kwlist in INTENT_KEYWORDS.items():
        for kw in kwlist:
            if kw in t:
                detected.append(intent)
                break
    return detected
