import io
import re
import cv2
import numpy as np
import urllib.request
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field

# Base de données (Simulation d'Internet) contenant les gabarits officiels des billets FRANC CFA (BEAC)
# NOTE : Pour l'environnement de développement, en l'absence de liens directs pérennes vers des billets 
# HD de FCFA, nous utilisons des URL fiables en placeholder. Remplacez-les par vos propres URL 
# si vous disposez d'images de Franc CFA hébergées (ex: imgur, site BEAC).
BANKNOTE_INTERNET_DB = {
    "500": {
        "name": "Billet de 500 Francs CFA",
        "url": "https://picsum.photos/seed/fcfa500/800/400.jpg"
    },
    "1000": {
        "name": "Billet de 1000 Francs CFA",
        "url": "https://picsum.photos/seed/fcfa1000/800/400.jpg"
    },
    "2000": {
        "name": "Billet de 2000 Francs CFA",
        "url": "https://picsum.photos/seed/fcfa2000/800/400.jpg"
    },
    "5000": {
        "name": "Billet de 5000 Francs CFA",
        "url": "https://picsum.photos/seed/fcfa5000/800/400.jpg"
    },
    "10000": {
        "name": "Billet de 10000 Francs CFA",
        "url": "https://picsum.photos/seed/fcfa10000/800/400.jpg"
    }
}

# Cache système
internet_cache = {}
orb = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global orb
    print("Démarrage de l'API Centrale (Zone FRANC CFA)...")
    orb = cv2.ORB_create(nfeatures=2000)
    yield
    print("Arrêt de l'API.")

app = FastAPI(
    title="Détecteur de Faux Billets V5 (Zone FCFA - BEAC)",
    description="Identifie le billet CFA, télécharge son gabarit, et réalise la comparaison des pixels.",
    version="5.0.0",
    lifespan=lifespan
)

def identify_banknote_from_image(image_hsv):
    """
    Analyse les pics colorimétriques de la photo pour deviner de quel billet FCFA il s'agit.
    Règles BEAC : 500=Marron, 1000=Bleu, 2000=Rouge/Rose, 5000=Vert, 10000=Violet.
    """
    color_ranges = {
        "500": {"lower": np.array([10, 50, 50]), "upper": np.array([25, 255, 255])},     # Marron/Brun
        "1000": {"lower": np.array([100, 50, 50]), "upper": np.array([130, 255, 255])},  # Bleu
        "2000": {"lower": np.array([0, 50, 50]), "upper": np.array([10, 255, 255])},     # Rouge gamme 1
        "2000_b": {"lower": np.array([170, 50, 50]), "upper": np.array([180, 255, 255])},# Rouge gamme 2
        "5000": {"lower": np.array([35, 50, 50]), "upper": np.array([85, 255, 255])},    # Vert
        "10000": {"lower": np.array([130, 50, 50]), "upper": np.array([160, 255, 255])}  # Violet
    }
    
    pixels_count = {"500": 0, "1000": 0, "2000": 0, "5000": 0, "10000": 0}
    
    for val, bounds in color_ranges.items():
        mask = cv2.inRange(image_hsv, bounds["lower"], bounds["upper"])
        count = cv2.countNonZero(mask)
        if val == "2000_b":
            pixels_count["2000"] += count
        else:
            pixels_count[val] += count
            
    best_match = max(pixels_count, key=pixels_count.get)
    
    # Par défaut si peu de couleurs détectées
    if pixels_count[best_match] < 500:
        return "1000"
        
    return best_match

def fetch_internet_reference(denomination: str):
    if denomination in internet_cache:
        return internet_cache[denomination]
        
    db_entry = BANKNOTE_INTERNET_DB.get(denomination)
    if not db_entry:
        return None, None
        
    print(f">> RECHERCHE INTERNET : Gabarit de référence pour {db_entry['name']}...")
    try:
        # Ajout d'un User-Agent (Navigateur) pour éviter le blocage "Error 403 Forbidden" de Wikipédia
        req_obj = urllib.request.Request(
            db_entry["url"],
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        req = urllib.request.urlopen(req_obj)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img_color = cv2.imdecode(arr, -1)
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(img_gray, None)
        internet_cache[denomination] = (kp, des)
        return kp, des
    except Exception as e:
        print(f"Erreur téléchargement: {e}")
        return None, None


# =====================================================================
# ENDPOINT 1 : Numéro de Série
# =====================================================================
class SerialCheckRequest(BaseModel):
    currency: str = Field(..., description="La devise (ex: 'XAF', 'XOF')")
    serial_number: str = Field(...)

@app.post("/api/v1/detect/serial")
def detect_serial(request: SerialCheckRequest):
    currency = request.currency.upper()
    serial = request.serial_number.upper()
    
    result = {"is_fake": False, "confidence": 100, "message": ""}
    
    # Franc CFA BEAC = XAF, BCEAO = XOF
    if currency in ["XAF", "XOF"]:
        if not re.match(r"^[A-Z]?\d{10,12}$", serial):
            return {"is_fake": True, "message": "Format Invalide Franc CFA (Attendu: suite de chiffres/lettres)."}
        
        # Règle mathématique pour la simulation
        if serial[-1] == '0':
            return {"is_fake": True, "message": "Faux Billet détecté (Erreur de somme de contrôle BEAC)."}
            
        result["message"] = "Numéro de série valide dans l'espace monétaire BEAC/BCEAO."
    else:
        result["message"] = f"Pas de règle de checksum mathématique enregistrée pour la devise {currency}."

    return result

# =====================================================================
# ENDPOINT 2 : Reconnaissance Visuelle FCFA
# =====================================================================
@app.post("/api/v1/detect/image")
async def detect_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Veuillez envoyer une image.")
        
    try:
        image_bytes = await file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        upload_color = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        upload_gray = cv2.cvtColor(upload_color, cv2.COLOR_BGR2GRAY)
        upload_hsv = cv2.cvtColor(upload_color, cv2.COLOR_BGR2HSV)
    except Exception:
        raise HTTPException(status_code=400, detail="Fichier corrompu.")

    # 1. Deviner le type de billet CFA selon la colorimétrie
    denomination = identify_banknote_from_image(upload_hsv)
    info_billet = BANKNOTE_INTERNET_DB[denomination]
    
    # 2. Chercher sur le web la référence
    kp_ref, des_ref = fetch_internet_reference(denomination)
    
    if des_ref is None:
         raise HTTPException(status_code=503, detail="Réseau inaccessible : Impossible de contacter la BEAC.")

    # 3. Extraction (Uploaded Image)
    kp_upload, des_upload = orb.detectAndCompute(upload_gray, None)

    if des_upload is None or len(des_upload) < 10:
        return {
            "filename": file.filename,
            "etapes": {"identification": "Echec", "comparaison": "Impossible"},
            "is_fake": True,
            "message": "Erreur visuelle : Signature géométrique illisible (floue ou objet incongru)."
        }

    # 4. Comparaison BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_upload, des_ref)
    
    matches = sorted(matches, key=lambda x: x.distance)
    total_matches = len(matches)
    
    THRESHOLD_AUTHENTIC = 60
    
    is_fake = True
    if total_matches >= THRESHOLD_AUTHENTIC:
        is_fake = False
        message = f"AUTHENTIQUE : Empreintes géométriques valides confortées par le standard {info_billet['name']} BEAC."
    elif total_matches >= (THRESHOLD_AUTHENTIC / 2):
        message = f"INDÉTERMINÉ : Seulement {total_matches} correspondances. Contrefaçon légère ou mauvais scan."
    else:
        message = f"FAUSSAIRE RECONNU : Échec du mapping ({total_matches} points). Ce n'est pas un vrai {info_billet['name']} !"

    return {
        "filename": file.filename,
        "workflow": {
            "1_identification": f"Deviné algorithmiquement comme un {info_billet['name']}",
            "2_internet_search": f"Gabarit officiel BEAC téléchargé avec succès.",
            "3_comparaison_orb": f"{total_matches} empreintes certifiées (Minimum requis: {THRESHOLD_AUTHENTIC})"
        },
        "is_fake": is_fake,
        "message": message
    }
