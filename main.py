import io
import re
import cv2
import numpy as np
import urllib.request
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field

# URL de l'image de référence sur Internet (Un véritable billet de 50 Euros HD pour l'exemple)
REFERENCE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4b/50_Euro_Serie_Europa_Vorderseite.jpg/800px-50_Euro_Serie_Europa_Vorderseite.jpg"

# Stockage pour le "cerveau" visuel OpenCV
reference_image = None
keypoints_ref = []
descriptors_ref = None
orb = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global reference_image, keypoints_ref, descriptors_ref, orb
    print("Démarrage de l'API / Préparation du scanner OpenCV...")
    
    # 1. Initialiser l'algorithme ORB (Oriented FAST and Rotated BRIEF) utilisé en Computer Vision
    orb = cv2.ORB_create(nfeatures=2000)
    
    # 2. Se connecter à Internet pour télécharger l'image officielle authentique
    print("Téléchargement du billet de référence parfait depuis Internet...")
    try:
        req = urllib.request.urlopen(REFERENCE_URL)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        ref_color = cv2.imdecode(arr, -1)
        
        # 3. Traiter le gabarit
        reference_image = cv2.cvtColor(ref_color, cv2.COLOR_BGR2GRAY)
        
        # 4. Extraction mathématique des points clés (les angles, filigranes) du VRAI billet
        keypoints_ref, descriptors_ref = orb.detectAndCompute(reference_image, None)
        print(f"Gabarit en mémoire : {len(keypoints_ref)} points géométriques extraits de wiki.")
    except Exception as e:
        print(f"Erreur téléchargement gabarit: {e}")
        
    yield
    print("Arrêt de l'API.")

# Instanciation de l'API
app = FastAPI(
    title="Détecteur de Faux Billets V3 (Vision Pro)",
    description="Identifie la monnaie via OpenCV par mise en correspondance de pixels clés avec un gabarit venant d'Internet.",
    version="3.0.0",
    lifespan=lifespan
)


# =====================================================================
# ENDPOINT 1 : Vérification Numéro de Série + Devise
# =====================================================================

class SerialCheckRequest(BaseModel):
    currency: str = Field(..., description="La devise (ex: 'EUR', 'USD')")
    serial_number: str = Field(..., description="Le numéro imprimé")

@app.post("/api/v1/detect/serial")
def detect_serial(request: SerialCheckRequest):
    currency = request.currency.upper()
    serial = request.serial_number.upper()
    
    result = {
        "is_fake": False,
        "confidence": 100,
        "message": ""
    }
    
    if currency == "EUR":
        pattern = r"^[A-Z]{2}\d{10}$"
        if not re.match(pattern, serial):
            return {"is_fake": True, "message": "Format Invalide EUR (2 lettres + 10 chiffres)."}
        
        # Checksum mock
        if serial[-1] == '0':
            result["is_fake"] = True
            result["confidence"] = 85
            result["message"] = "Somme de contrôle invalide. Billet Contrefait."
        else:
            result["message"] = "Le numéro de série respecte le validateur algorithmique."
            
    elif currency == "USD":
        pattern = r"^[A-Z]{2}\d{8}[A-Z]?$"
        if not re.match(pattern, serial):
            return {"is_fake": True, "message": "Format Invalide USD."}
        result["message"] = "Série USD formellement valide."
    else:
        result["message"] = f"Pas de règle stricte pour la devise {currency}."

    return result


# =====================================================================
# ENDPOINT 2 : Reconnaissance Visuelle par Téléchargement / Comparaison
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
    except Exception:
        raise HTTPException(status_code=400, detail="Image corrompue.")

    if descriptors_ref is None:
         raise HTTPException(status_code=503, detail="Le gabarit internet n'a pas pu être chargé.")

    # 1. Extraction des caractéristiques de l'image postée dans l'API
    kp_upload, des_upload = orb.detectAndCompute(upload_gray, None)

    if des_upload is None or len(des_upload) < 10:
        return {
            "filename": file.filename,
            "match_points_found": 0,
            "is_fake": True,
            "message": "SUSPECT : Impossible de calculer la géométrie du billet."
        }

    # 2. Création du moteur de comparaison (Brute Force Matcher de OpenCV)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # 3. L'intelligence est ici : est-ce que les points de l'UPLOAD se superposent à INTERNET ?
    matches = bf.match(des_upload, descriptors_ref)
    
    # Trier par similarité (la plus petite distance)
    matches = sorted(matches, key=lambda x: x.distance)
    total_matches = len(matches)
    
    # Fixer la limite d'authenticité pour le TP 
    # (Un nombre de correspondances élevé = même objet)
    THRESHOLD_AUTHENTIC = 60
    
    is_fake = True
    if total_matches >= THRESHOLD_AUTHENTIC:
        is_fake = False
        message = f"POSITIF: {total_matches} empreintes géométriques valides correspondant au gabarit officiel Internet."
    elif total_matches >= (THRESHOLD_AUTHENTIC / 2):
        message = f"INDÉTERMINÉ: {total_matches} correspondances. Contrefaçon ou scan de mauvaise qualité."
    else:
        message = f"FAUX : Échec du mapping ({total_matches} points). Les caractéristiques visuelles ne correspondent pas du tout."

    return {
        "filename": file.filename,
        "algo_details": "OpenCV ORB Feature Matching",
        "match_points_found": total_matches,
        "points_requis_minimaux": THRESHOLD_AUTHENTIC,
        "is_fake": is_fake,
        "message": message
    }
