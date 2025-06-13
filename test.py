import os
import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage import img_as_ubyte
from tkinter import Tk, filedialog, Button, Label, Frame, messagebox, Scale, HORIZONTAL
from PIL import Image, ImageTk
from collections import defaultdict
import math
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from deepface import DeepFace

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
BASE_PATH = r"C:\Users\Admin\Desktop\sem_6\biomka\PB_MBwRC_3"
SIGNATURE_PATH = os.path.join(BASE_PATH, "Signature")
FACES_PATH = os.path.join(BASE_PATH, "Faces")

# -----------------------------------------------------------------------------
# Signature recognition
# -----------------------------------------------------------------------------
class SignatureRecognizer:
    def __init__(self):
        self.database_path = SIGNATURE_PATH
        self.signature_groups = defaultdict(list)
        self.load_database()

    def load_database(self):
        os.makedirs(self.database_path, exist_ok=True)
        for fn in os.listdir(self.database_path):
            path = os.path.join(self.database_path, fn)
            if not os.path.isfile(path): continue
            try:
                with Image.open(path) as img:
                    if img.format != 'PNG': continue
            except:
                continue
            _, minutiae = self.process_image(path)
            name = ''.join(c for c in os.path.splitext(fn)[0] if not c.isdigit())
            self.signature_groups[name].append(minutiae)

    def process_image(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: raise ValueError(f"Cannot read {img_path}")
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        skeleton = skeletonize(binary // 255)
        skel_img = img_as_ubyte(skeleton)
        minutiae = self.extract_minutiae(skel_img)
        return skel_img, minutiae

    def extract_minutiae(self, skel):
        pts = []
        h, w = skel.shape
        for i in range(1, h-1):
            for j in range(1, w-1):
                if skel[i,j] == 255:
                    neigh = skel[i-1:i+2, j-1:j+2]
                    cnt = np.sum(neigh)//255 - 1
                    if cnt == 1 or cnt >= 3:
                        pts.append((j,i))
        return pts

    def compare_with_database(self, query):
        res = {}
        for name, sigs in self.signature_groups.items():
            matches = [self._match(query, s) for s in sigs]
            res[name] = sum(matches)/len(matches) if matches else 0
        return res

    def _match(self, m1, m2, thresh=10):
        used, c = set(), 0
        for p in m1:
            for idx, q in enumerate(m2):
                if idx in used: continue
                if math.hypot(p[0]-q[0], p[1]-q[1]) < thresh:
                    c += 1; used.add(idx); break
        return c

# -----------------------------------------------------------------------------
# Face recognition
# -----------------------------------------------------------------------------
class DeepFaceRecognizer:
    def __init__(self, database_path=FACES_PATH, model_name="Facenet", enforce_detection=True):
        self.db = self._load_database(database_path)
        self.model_name = model_name
        self.enforce_detection = enforce_detection

    def _load_database(self, base):
        db = defaultdict(list)
        for it in os.listdir(base):
            path = os.path.join(base,it)
            if os.path.isdir(path):
                for f in os.listdir(path):
                    if f.lower().endswith(("png","jpg","jpeg")):
                        db[it].append(os.path.join(path,f))
            elif os.path.isfile(path) and path.lower().endswith(("png","jpg","jpeg")):
                nm = ''.join(c for c in os.path.splitext(it)[0] if not c.isdigit())
                db[nm].append(path)
        return {k:v for k,v in db.items() if v}

    def recognize(self, qpath):
        out = {}
        for name, imgs in self.db.items():
            dlist = []
            for img in imgs:
                try:
                    r = DeepFace.verify(img1_path=qpath, img2_path=img,
                                         model_name=self.model_name,
                                         enforce_detection=self.enforce_detection)
                    dlist.append(r['distance'])
                except: pass
            if dlist: out[name] = min(dlist)
        return out

    @staticmethod
    def dist2conf(d, maxd=1.0): return max(0.0, min(1.0, 1-d/maxd))

# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------
class SignatureGUI:
    def __init__(self, root):
        self.root = root; root.title("Biometric System"); root.config(bg="#f0f0f0")
        self.sig = SignatureRecognizer()
        self.face = DeepFaceRecognizer()
        self.w_face = 0.7; self.w_sig = 0.3
        self._build_ui()

    def _build_ui(self):
        # Title
        Label(self.root, text="Signature + Face Recognition", font=("Segoe UI",18,"bold"),
              bg="#f0f0f0").grid(row=0, column=0, columnspan=4, pady=10)
        # Buttons
        Button(self.root, text="Load Signature", bg="#4CAF50", fg="white",
               command=self._load_sig).grid(row=1,column=0,padx=5)
        Button(self.root, text="Load Face", bg="#FF9800", fg="white",
               command=self._load_face).grid(row=1,column=1,padx=5)
        self.btn_compare = Button(self.root, text="Compare", bg="#2196F3", fg="white",
                                  state="disabled", command=self._compare)
        self.btn_compare.grid(row=1,column=2,padx=5)
        # Weight sliders
        Label(self.root, text="Face weight", bg="#f0f0f0").grid(row=2,column=0)
        Scale(self.root, from_=0, to=1, resolution=0.05, orient=HORIZONTAL,
              command=self._update_w_face, length=150).set(self.w_face)
        Scale(self.root, from_=0, to=1, resolution=0.05, orient=HORIZONTAL,
              command=self._update_w_sig, length=150).grid(row=2,column=1)
        Label(self.root, text="Sig weight", bg="#f0f0f0").grid(row=2,column=2)
        # Image displays
        self.lbl_sig = Label(self.root, bg="#f0f0f0"); self.lbl_sig.grid(row=3,column=0)
        self.lbl_face = Label(self.root, bg="#f0f0f0"); self.lbl_face.grid(row=3,column=1)
        # Result
        self.lbl_res = Label(self.root, text="Awaiting input...", font=("Segoe UI",12),
                             bg="#f0f0f0"); self.lbl_res.grid(row=4,column=0,columnspan=4,pady=10)
        # Plot
        self.frm_plot = Frame(self.root, bg="#f0f0f0"); self.frm_plot.grid(row=5,column=0,columnspan=4)

    def _update_w_face(self, val): self.w_face=float(val)
    def _update_w_sig(self, val): self.w_sig=float(val)

    def _load_sig(self):
        p=filedialog.askopenfilename(initialdir=SIGNATURE_PATH,filetypes=[("PNG","*.png")])
        if not p: return
        skel,self.qmin=self.sig.process_image(p)
        im=Image.fromarray(skel); im.thumbnail((200,200))
        ph=ImageTk.PhotoImage(im); self.lbl_sig.config(image=ph); self.lbl_sig.image=ph
        self.sload=True; self.lbl_res.config(text="Signature loaded.")
        if getattr(self,'fload',False): self.btn_compare.config(state="normal")

    def _load_face(self):
        p=filedialog.askopenfilename(initialdir=FACES_PATH,filetypes=[("Img","*.png *.jpg")])
        if not p: return
        faces=DeepFace.extract_faces(img_path=p,enforce_detection=self.face.enforce_detection)
        if not faces: messagebox.showerror("Error","No face"); return
        im=Image.open(p); im.thumbnail((200,200))
        ph=ImageTk.PhotoImage(im); self.lbl_face.config(image=ph); self.lbl_face.image=ph
        self.qface=p; self.fload=True; self.lbl_res.config(text="Face loaded.")
        if getattr(self,'sload',False): self.btn_compare.config(state="normal")

    def _compare(self):
        sig_scores=self.sig.compare_with_database(self.qmin)
        maxs=max(sig_scores.values()) if sig_scores else 1
        sc={n:v/maxs for n,v in sig_scores.items()}
        fd=self.face.recognize(self.qface)
        fc={n:DeepFaceRecognizer.dist2conf(d) for n,d in fd.items()}
        cmb={n:self.w_face*fc[n]+self.w_sig*sc[n] for n in set(sc)&set(fc)}
        for w in self.frm_plot.winfo_children(): w.destroy()
        if cmb:
            b,s=max(cmb.items(),key=lambda x:x[1])
            txt = f"IDENTITY: {b} ({s:.2f})" if s>0.5 and b==max(sc,key=sc.get)==max(fc,key=fc.get) else "Uncertain"
            self.lbl_res.config(text=txt)
            self._plot(cmb)
        else:
            self.lbl_res.config(text="No match")

    def _plot(self,c):
        fig=Figure(figsize=(5,3),dpi=100)
        ax=fig.add_subplot(111)
        names,vals=zip(*c.items())
        ax.bar(names,vals)
        ax.set_ylim(0,1);ax.set_ylabel("Conf");ax.tick_params(axis='x',rotation=45)
        can=FigureCanvasTkAgg(fig,master=self.frm_plot);can.draw();can.get_tk_widget().pack()

if __name__=="__main__":
    root=Tk(); SignatureGUI(root); root.mainloop()
