# interfaces/utils/interface_router.py
from .json_loader import load_json
from .intent_rules import detect_intent, extract_keywords
import logging
import re

logger = logging.getLogger("interface_router")

# Priority order of special interfaces
# Priority order of special interfaces
SPECIAL_ORDER = ["fee_and_packages", "facilities_grid", "training_programs", "location_directory"]

SPECIAL_MAP = {
    "fee_and_packages": "special/fee_and_packages.json",
    "facilities_grid": "special/facilities_grid.json",
    "training_programs": "special/training_programs.json",
    "location_directory": "special/location_directory.json"
}

# ================================
# 🚀 Simple NER-like Jakarta Synonyms
# ================================
CITY_SYNONYMS = {
    # Jakarta Pusat
    "jakarta pusat": "Jakarta Pusat",
    "jakpus": "Jakarta Pusat",
    "central jakarta": "Jakarta Pusat",

    # Jakarta Barat
    "jakarta barat": "Jakarta Barat",
    "jakbar": "Jakarta Barat",
    "west jakarta": "Jakarta Barat",

    # Jakarta Utara
    "jakarta utara": "Jakarta Utara",
    "jakut": "Jakarta Utara",
    "north jakarta": "Jakarta Utara",

    # Jakarta Selatan
    "jakarta selatan": "Jakarta Selatan",
    "jaksel": "Jakarta Selatan",
    "south jakarta": "Jakarta Selatan",

    # Jakarta Timur
    "jakarta timur": "Jakarta Timur",
    "jaktim": "Jakarta Timur",
    "east jakarta": "Jakarta Timur",

    # All Jakarta
    "dki jakarta": "ALL_JAKARTA",
    "jakarta": "ALL_JAKARTA",
    "jakarta area": "ALL_JAKARTA",
    "sekitar jakarta": "ALL_JAKARTA",
    "jakarta aja": "ALL_JAKARTA"
}


def _normalize(text):
    return text.lower().strip() if text else ""


def _map_city_synonyms(user_input: str):
    """
    Detect synonyms from CITY_SYNONYMS.
    Returns a list: ["Jakarta Selatan"] or ["ALL_JAKARTA"] etc.
    """
    text = _normalize(user_input)
    found = set()

    for key, mapped in CITY_SYNONYMS.items():
        if key in text:
            found.add(mapped)

    return list(found)


def _extract_locations_from_text(user_input: str, location_json_obj: dict):
    """
    Extracts both city-level and district-level matches from user_input.

    Returns:
        {
            "cities": [...],          # list of matched city names (as in JSON 'city' field)
            "districts": [...]        # list of matched district names (as in JSON 'district' field)
        }

    Priority for filtering:
        1. If any districts found -> will filter rows by district
        2. Else if cities found -> filter rows by city
        3. Else -> no filtering (return empty lists)
    """
    if not user_input or not location_json_obj:
        return {"cities": [], "districts": []}

    text = _normalize(user_input)
    matched_cities = set()
    matched_districts = set()

    # 1) synonyms detection (Jakarta mapping)
    synonyms = _map_city_synonyms(text)
    if synonyms:
        # ALL_JAKARTA special handling
        if "ALL_JAKARTA" in synonyms:
            rows = location_json_obj.get("rows", [])
            all_cities = {r.get("city") for r in rows if r.get("city")}
            # map to cities
            matched_cities.update([c for c in all_cities if c])
        else:
            matched_cities.update([s for s in synonyms if s != "ALL_JAKARTA"])

    # 2) scan JSON rows for city/district/aliases matches
    rows = location_json_obj.get("rows", []) if isinstance(location_json_obj, dict) else []
    # Build sets for quick lookup
    json_cities = set(r.get("city") for r in rows if r.get("city"))
    json_districts = set(r.get("district") for r in rows if r.get("district"))

    # Direct exact match against city values
    for city in json_cities:
        if not city: 
            continue
        city_lower = city.lower()
        if re.search(r'\b' + re.escape(city_lower) + r'\b', text):
            matched_cities.add(city)

    # Direct exact match against district values
    for district in json_districts:
        if not district:
            continue
        district_lower = district.lower()
        if re.search(r'\b' + re.escape(district_lower) + r'\b', text):
            matched_districts.add(district)

    # 3) aliases & token fallback per row
    # iterate rows to check aliases that map to district or city
    for r in rows:
        aliases = r.get("aliases") or []
        district = r.get("district")
        city = r.get("city")

        for a in aliases:
            if not a:
                continue
            a_low = a.lower().strip()
            # full alias match
            if re.search(r'\b' + re.escape(a_low) + r'\b', text):
                if district:
                    matched_districts.add(district)
                elif city:
                    matched_cities.add(city)

        # token fallback for district or city (short tokens like 'pusat','selatan')
        # Only used if no matches found yet
    if not matched_cities and not matched_districts:
        # try token matching: look for tokens from district and city names
        for city in json_cities:
            for token in re.split(r'\W+', (city or "").lower()):
                if token and len(token) >= 3 and re.search(r'\b' + re.escape(token) + r'\b', text):
                    matched_cities.add(city)
        for district in json_districts:
            for token in re.split(r'\W+', (district or "").lower()):
                if token and len(token) >= 3 and re.search(r'\b' + re.escape(token) + r'\b', text):
                    matched_districts.add(district)

    return {
        "cities": list(matched_cities),
        "districts": list(matched_districts)
    }


def _load_and_filter_location(location_path, user_input):
    """
    Load location_directory.json and filter by district (preferred) or city.
    Returns a copy of the location object (with filtered rows) or the full obj if no filter applied.
    """
    try:
        loc_obj = load_json(location_path)
    except Exception as e:
        logger.warning(f"Failed to load location file {location_path}: {e}")
        return None

    matches = _extract_locations_from_text(user_input or "", loc_obj)
    matched_districts = matches.get("districts", []) or []
    matched_cities = matches.get("cities", []) or []

    # PRIORITIZE district filtering
    if matched_districts:
        logger.info(f"Filtering by districts: {matched_districts}")
        rows = loc_obj.get("rows", [])
        filtered = [r for r in rows if (r.get("district") in matched_districts)]
        if not filtered:
            logger.info("Districts matched but no rows found; returning full list.")
            return loc_obj
        new_obj = dict(loc_obj)
        new_obj["rows"] = filtered
        new_obj["_filtered_by_districts"] = matched_districts
        return new_obj

    # If no districts matched but cities matched
    if matched_cities:
        logger.info(f"Filtering by cities: {matched_cities}")
        rows = loc_obj.get("rows", [])
        filtered = [r for r in rows if (r.get("city") in matched_cities)]
        if not filtered:
            logger.info("Cities matched but no rows found; returning full list.")
            return loc_obj
        new_obj = dict(loc_obj)
        new_obj["rows"] = filtered
        new_obj["_filtered_by_cities"] = matched_cities
        return new_obj

    # No matches → return full object
    return loc_obj


def choose_interfaces(user_input: str):
    """
    Returns list of interface JSON objects:
    - Always include global panel
    - Optionally include fee, facilities, location, and training interfaces
    """
    interfaces = []

    # Selalu kirim Global QuickPanel
    try:
        interfaces.append(load_json("global/global_quickpanel.json"))
    except Exception as e:
        logger.warning(f"Global quickpanel load failed: {e}")

    intents = detect_intent(user_input or "")
    keywords = extract_keywords(user_input or "")

    matched = set()

    # =======================
    # PRICE / BIAYA
    # =======================
    price_kws = ["harga", "biaya", "tarif", "paket", "cost", "price"]
    has_price_kw = any(k in keywords for k in price_kws)

    if "ask_price" in intents or has_price_kw:
        matched.add("fee_and_packages")

    # =======================
    # FACILITIES / LAYANAN
    # (hanya kalau TIDAK ada sinyal harga)
    # =======================
    fac_kws = ["fasilitas", "layanan", "peralatan", "konseling", "terapi", "workshop"]
    has_fac_kw = any(k in keywords for k in fac_kws)

    if (
        "ask_facilities" in intents
        and "ask_price" not in intents
        and not has_price_kw
    ):
        matched.add("facilities_grid")
    else:
        if (
            "ask_price" not in intents
            and not has_price_kw
            and has_fac_kw
        ):
            matched.add("facilities_grid")

    # =======================
    # TRAINING / PELATIHAN SOFT SKILLS
    # =======================
    training_kws = ["pelatihan", "training", "workshop", "soft skill", "soft skills"]
    has_training_kw = any(k in keywords for k in training_kws)

    # tampilkan panel training kalau intent ask_training
    # ATAU ada keyword pelatihan/soft skill
    if "ask_training" in intents or has_training_kw:
        matched.add("training_programs")

    # =======================
    # LOCATION
    # =======================
    loc_kws = [
        "lokasi", "alamat", "cabang", "dekat saya",
        "where", "di mana", "dimana", "ada di mana"
    ]
    if "ask_location" in intents or any(k in keywords for k in loc_kws):
        matched.add("location_directory")

    # Kalau tidak ada interface khusus → cuma global
    if not matched:
        return interfaces

    # Load interface sesuai prioritas
    for key in SPECIAL_ORDER:
        if key in matched:
            try:
                if key == "location_directory":
                    loc = _load_and_filter_location(SPECIAL_MAP[key], user_input)
                    if loc:
                        interfaces.append(loc)
                else:
                    interfaces.append(load_json(SPECIAL_MAP[key]))
            except Exception as e:
                logger.warning(f"Failed to load {key}: {e}")

    return interfaces
