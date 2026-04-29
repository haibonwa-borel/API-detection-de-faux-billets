import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import httpx
import asyncio
import threading

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("VerifyCash Vision CFA")
        self.geometry("500x750")
        self.token = None  # Contiendra le jeton après connexion

        # --- UI DESIGN ---
        self.grid_columnconfigure(0, weight=1)
        self.title_lbl = ctk.CTkLabel(self, text="VERIFYCASH VISION", font=("Impact", 28, "italic"))
        self.title_lbl.pack(pady=30)

        self.img_frame = ctk.CTkFrame(self, width=400, height=300, corner_radius=20, border_width=2)
        self.img_frame.pack(pady=10, padx=20)
        self.img_label = ctk.CTkLabel(self.img_frame, text="AUCUNE IMAGE")
        self.img_label.place(relx=0.5, rely=0.5, anchor="center")

        self.btn_select = ctk.CTkButton(self, text="SÉLECTIONNER UN BILLET", height=45, command=self.select_file)
        self.btn_select.pack(pady=20)

        self.btn_run = ctk.CTkButton(self, text="LANCER L'EXPERTISE", fg_color="#27ae60", hover_color="#1e8449",
                                      height=50, state="disabled", command=self.trigger_analysis)
        self.btn_run.pack(pady=10)

        self.res_label = ctk.CTkLabel(self, text="Initialisation...", font=("Arial", 14))
        self.res_label.pack(pady=20)

        self.file_path = None
        
        # Tentative de connexion automatique au lancement
        threading.Thread(target=self.auto_login, daemon=True).start()

    def auto_login(self):
        """Récupère automatiquement le token JWT du backend"""
        try:
            # On envoie les identifiants au format form-data (comme dans Postman)
            resp = httpx.post("http://127.0.0.1:8000/token", 
                              data={"username": "admin", "password": "1234"})
            if resp.status_code == 200:
                self.token = resp.json()["access_token"]
                self.after(0, lambda: self.res_label.configure(text="Serveur connecté", text_color="gray"))
            else:
                self.after(0, lambda: self.res_label.configure(text="Erreur d'authentification", text_color="orange"))
        except Exception:
            self.after(0, lambda: self.res_label.configure(text="Serveur injoignable", text_color="red"))

    def select_file(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg")])
        if path:
            self.file_path = path
            raw_img = Image.open(path)
            img = ctk.CTkImage(light_image=raw_img, dark_image=raw_img, size=(380, 280))
            self.img_label.configure(image=img, text="")
            self.btn_run.configure(state="normal")

    def trigger_analysis(self):
        if not self.token:
            self.res_label.configure(text="Veuillez patienter : connexion en cours...", text_color="orange")
            return
        self.res_label.configure(text="Analyse microscopique en cours...", text_color="cyan")
        self.btn_run.configure(state="disabled")
        threading.Thread(target=self.async_request, daemon=True).start()

    def async_request(self):
        async def call():
            # On utilise le format Bearer Token avec le vrai jeton récupéré
            headers = {"Authorization": f"Bearer {self.token}"}
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                with open(self.file_path, "rb") as f:
                    return await client.post("http://127.0.0.1:8000/predict", 
                                             files={"file": f}, 
                                             headers=headers)
        try:
            resp = asyncio.run(call())
            self.after(0, self.show_result, resp.json())
        except Exception as e:
            self.after(0, lambda: self.res_label.configure(text="Erreur de connexion", text_color="red"))
        finally:
            self.after(0, lambda: self.btn_run.configure(state="normal"))

    def show_result(self, data):
        # Synchronisation avec les clés de ton backend : 'is_genuine' ou 'result'
        if "result" in data:
            # Détermine si c'est authentique pour la couleur
            verdict = data["result"]
            color = "#2ecc71" if verdict == "AUTHENTIQUE" else "#e74c3c"
            
            confiance = data.get("confidence", 0) * 100
            msg = f"VERDICT : {verdict}\nCONFIANCE : {confiance:.2f}%"
            self.res_label.configure(text=msg, text_color=color, font=("Arial", 18, "bold"))
        else:
            # Affiche l'erreur renvoyée par FastAPI (ex: 401 ou 422)
            detail = data.get("detail", "Erreur serveur")
            self.res_label.configure(text=f"ERREUR : {detail}", text_color="#f1c40f")

if __name__ == "__main__":
    app = App()
    app.mainloop()