from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from datetime import datetime, timedelta, timezone
import uvicorn
import numpy as np
from PIL import Image
import io
import time
import httpx

# --- CONFIGURATION SÉCURITÉ ---
SECRET_KEY = "TON_CODE_SECRET_TRES_LONG" 
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

USER_DB = {"admin": "1234"} 

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI(title="CFA Vision Scan API - Counterfeit Detection", version="3.1.0")

# --- MODÈLE DE DONNÉES SIMULÉ ---
DETECTION_HISTORY = []

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Token invalide")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Session expirée")

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    print(f"Tentative de connexion : user={form_data.username}, pass={form_data.password}")
    user_password = USER_DB.get(form_data.username)
    if not user_password or form_data.password != user_password:
        print(f"ÉCHEC : attendu={user_password}, reçu={form_data.password}")
        raise HTTPException(status_code=400, detail="Identifiants incorrects")
    
    print("SUCCÈS : Token généré")
    access_token = create_access_token(data={"sub": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}

async def call_external_detection_api(image_bytes):
    """
    Simule l'appel à une API externe de détection haute performance (Hausse API).
    """
    # Simulation d'un délai réseau pour l'API externe
    time.sleep(2.0) 
    
    # Logique de détection simulée basée sur l'analyse d'image
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    stat = np.array(img).mean()
    
    # Un billet est considéré comme authentique si la luminosité moyenne est dans une certaine plage
    is_genuine = True if 80 < stat < 200 else False
    confidence = float(np.random.uniform(0.95, 0.999))
    
    return {
        "status": "success",
        "is_genuine": is_genuine,
        "confidence": confidence,
        "api_source": "Hausse-Detection-Pro-V1"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...), current_user: str = Depends(get_current_user)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Fichier non supporté.")

    try:
        content = await file.read()
        # Appel à l'API de détection (simulé)
        detection_result = await call_external_detection_api(content)
        
        response = {
            "is_genuine": detection_result["is_genuine"],
            "result": "AUTHENTIQUE" if detection_result["is_genuine"] else "CONTREFAÇON",
            "confidence": round(detection_result["confidence"], 4),
            "api_used": detection_result["api_source"],
            "timestamp": datetime.now().isoformat(),
            "operator": current_user
        }
        
        # Enregistrement automatique de la démission (soumission)
        DETECTION_HISTORY.append(response)
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
async def get_history(current_user: str = Depends(get_current_user)):
    """Récupère l'historique des soumissions de détection"""
    return {"history": DETECTION_HISTORY, "count": len(DETECTION_HISTORY)}

if __name__ == "__main__":
    print("Démarrage du serveur CFA Vision Scan...")
    uvicorn.run(app, host="127.0.0.1", port=8000)