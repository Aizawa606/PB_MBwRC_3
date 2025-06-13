import os
import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage import img_as_ubyte
from tkinter import Tk, filedialog, Button, Label, Frame, messagebox
from PIL import Image, ImageTk
from collections import defaultdict
import math
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from deepface import DeepFace

# -----------------------------------------------------------------------------
# Ustawienia ścieżek (zmodyfikuj według swojej struktury)
# -----------------------------------------------------------------------------
BASE_PATH = r"C:\Users\Admin\Desktop\sem_6\biomka\PB_MBwRC_3"
SIGNATURE_PATH = os.path.join(BASE_PATH, "Signature")
FACES_PATH = os.path.join(BASE_PATH, "Faces")

# -------------------------
# Signature recognition
# -------------------------
class SignatureRecognizer:
    def __init__(self):
        self.database_path = SIGNATURE_PATH
        self.signature_groups = defaultdict(list)
        self.load_database()

    def load_database(self):
        os.makedirs(self.database_path, exist_ok=True)
        for filename in os.listdir(self.database_path):
            file_path = os.path.join(self.database_path, filename)
            if not os.path.isfile(file_path):
                continue
            try:
                with Image.open(file_path) as img:
                    if img.format != 'PNG':
                        continue
            except:
                continue
            processed_img, minutiae = self.process_image(file_path)
            name = ''.join([c for c in os.path.splitext(filename)[0] if not c.isdigit()])
            self.signature_groups[name].append(minutiae)

    def process_image(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Cannot read {img_path}")
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        skeleton = skeletonize(binary // 255)
        skel_img = img_as_ubyte(skeleton)
        minutiae = self.extract_minutiae(skel_img)
        return skel_img, minutiae

    def extract_minutiae(self, skel):
        pts = []
        for i in range(1, skel.shape[0] - 1):
            for j in range(1, skel.shape[1] - 1):
                if skel[i, j] == 255:
                    neigh = skel[i-1:i+2, j-1:j+2]
                    count = np.sum(neigh) // 255 - 1
                    if count == 1 or count >= 3:
                        pts.append((j, i))
        return pts

    def compare_with_database(self, query):
        results = {}
        for name, sigs in self.signature_groups.items():
            matches = [self._match_count(query, s) for s in sigs]
            results[name] = sum(matches) / len(matches) if matches else 0
        return results

    def _match_count(self, m1, m2, thresh=10):
        used, count = set(), 0
        for p in m1:
            for idx, q in enumerate(m2):
                if idx in used:
                    continue
                if math.hypot(p[0] - q[0], p[1] - q[1]) < thresh:
                    count += 1
                    used.add(idx)
                    break
        return count

# -------------------------
# Face recognition via DeepFace
# -------------------------
class DeepFaceRecognizer:
    def __init__(self, database_path=FACES_PATH, model_name="Facenet", enforce_detection=True):
        self.database_path = database_path
        self.model_name = model_name
        self.enforce_detection = enforce_detection
        self.db = self._load_database()

    def _load_database(self):
        db = defaultdict(list)
        # obsługa podfolderów i plików bezpośrednio w katalogu Faces
        for item in os.listdir(self.database_path):
            path = os.path.join(self.database_path, item)
            if os.path.isdir(path):
                # podfolder traktowany jako osoba
                for f in os.listdir(path):
                    if f.lower().endswith((".png", ".jpg", ".jpeg")):
                        db[item].append(os.path.join(path, f))
            elif os.path.isfile(path) and item.lower().endswith((".png", ".jpg", ".jpeg")):
                # plik bez podfolderu, nazwij osobę na podstawie liter
                name = ''.join([c for c in os.path.splitext(item)[0] if not c.isdigit()])
                db[name].append(path)
        return {name: imgs for name, imgs in db.items() if imgs}

    def recognize(self, query_path):
        results = {}
        for person, imgs in self.db.items():
            dists = []
            for img in imgs:
                try:
                    res = DeepFace.verify(
                        img1_path=query_path,
                        img2_path=img,
                        model_name=self.model_name,
                        enforce_detection=self.enforce_detection
                    )
                    dists.append(res['distance'])
                except Exception:
                    continue
            if dists:
                results[person] = min(dists)
        return results

    @staticmethod
    def dist2conf(dist, maxd=1.0):
        c = 1.0 - (dist / maxd)
        return max(0.0, min(1.0, c))

# -------------------------
# GUI Application
# -------------------------
class SignatureGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Multimodal Biometric System")
        self.root.config(bg="#f0f0f0")
        self.sig = SignatureRecognizer()
        self.face = DeepFaceRecognizer()
        self._build_ui()

    def _build_ui(self):
        frm = Frame(self.root, bg="#f0f0f0")
        frm.pack(pady=10)
        Label(self.root,
              text="Signature + Face Recognition",
              font=("Segoe UI", 16, "bold"),
              bg="#f0f0f0").pack(pady=5)
        self.bsig = Button(frm,
                           text="Load Signature",
                           command=self._load_sig,
                           bg="#4CAF50", fg="white",
                           padx=10, pady=5)
        self.bsig.pack(side="left", padx=5)
        self.bface = Button(frm,
                            text="Load Face Image",
                            command=self._load_face,
                            bg="#FF9800", fg="white",
                            padx=10, pady=5)
        self.bface.pack(side="left", padx=5)
        self.bcmp = Button(frm,
                           text="Compare",
                           command=self._compare,
                           state="disabled",
                           bg="#2196F3", fg="white",
                           padx=10, pady=5)
        self.bcmp.pack(side="left", padx=5)
        self.imglbl = Label(self.root, bg="#f0f0f0")
        self.imglbl.pack(pady=10)
        self.reslbl = Label(self.root,
                            text="Load signature and face",
                            bg="#f0f0f0",
                            font=("Segoe UI", 11))
        self.reslbl.pack(pady=5)
        self.plotfrm = Frame(self.root, bg="#f0f0f0")
        self.plotfrm.pack(pady=10)

    def _load_sig(self):
        p = filedialog.askopenfilename(
            initialdir=SIGNATURE_PATH,
            filetypes=[("PNG images", "*.png")]
        )
        if not p:
            return
        _, self.qmin = self.sig.process_image(p)
        img = Image.fromarray(_)
        img.thumbnail((300, 300))
        ph = ImageTk.PhotoImage(img)
        self.imglbl.config(image=ph)
        self.imglbl.image = ph
        self.sload = True
        self.reslbl.config(text="Signature loaded.")
        if getattr(self, 'fload', False):
            self.bcmp.config(state="normal")

    def _load_face(self):
        p = filedialog.askopenfilename(
            initialdir=FACES_PATH,
            filetypes=[("Image files", "*.png *.jpg *.jpeg")]
        )
        if not p:
            return
        faces = DeepFace.extract_faces(
            img_path=p,
            enforce_detection=self.face.enforce_detection
        )
        if not faces:
            messagebox.showerror("Error", "No face detected")
            return
        self.qface = p
        self.fload = True
        self.reslbl.config(text="Face loaded.")
        if getattr(self, 'sload', False):
            self.bcmp.config(state="normal")

    def _compare(self):
        sc = self.sig.compare_with_database(self.qmin)
        mc = max(sc.values()) if sc else 1
        sc = {n: v / mc for n, v in sc.items()}
        fd = self.face.recognize(self.qface)
        fc = {n: DeepFaceRecognizer.dist2conf(d) for n, d in fd.items()}
        w1, w2 = 0.7, 0.3
        cmb = {n: w1 * fc[n] + w2 * sc[n] for n in set(sc) & set(fc)}
        for w in self.plotfrm.winfo_children():
            w.destroy()
        if cmb:
            b, s = max(cmb.items(), key=lambda x: x[1])
            if s > 0.5 and b == max(sc, key=sc.get) == max(fc, key=fc.get):
                res = f"IDENTITY: {b} (score: {s:.2f})"
            else:
                res = "No recognition or low confidence"
            self._plot(cmb)
        else:
            res = "No common candidates"
        self.reslbl.config(text=res)

    def _plot(self, c):
        fig = Figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(111)
        names, vals = zip(*c.items())
        ax.bar(names, vals)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Confidence")
        ax.set_title("Combined Scores")
        ax.tick_params(axis='x', rotation=45)
        can = FigureCanvasTkAgg(fig, master=self.plotfrm)
        can.draw()
        can.get_tk_widget().pack()

if __name__ == "__main__":
    root = Tk()
    SignatureGUI(root)
    root.mainloop()
