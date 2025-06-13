# Importy bibliotek: przetwarzanie obrazu, GUI, obliczenia, wykresy itp.
import os
import cv2 
import numpy as np
from skimage.morphology import skeletonize  # do wyznaczania "szkieletu" obrazu
from skimage import img_as_ubyte  # konwersja obrazu do 8-bitowego formatu
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, Button, Label, Frame, messagebox
from PIL import Image, ImageTk  # obsługa obrazów w GUI
from collections import defaultdict
import math
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Klasa odpowiedzialna za logikę przetwarzania i porównywania podpisów
class SignatureRecognizer:
    def __init__(self):
        self.database_path = "Signature"  # Ścieżka do katalogu z podpisami
        self.signature_groups = defaultdict(list)  # Grupowanie podpisów według imienia
        self.load_database()  # Wczytanie bazy danych przy starcie
    
    def load_database(self):
        """Wczytuje wszystkie podpisy z katalogu, dzieli je na grupy wg nazw i przetwarza"""
        if not os.path.exists(self.database_path):
            os.makedirs(self.database_path)
            messagebox.showinfo("Info", f"Created database directory at {self.database_path}")
            return
            
        for filename in os.listdir(self.database_path):
            file_path = os.path.join(self.database_path, filename)

            # Sprawdzenie czy plik jest obrazem PNG (na podstawie zawartości, nie rozszerzenia)
            try:
                with Image.open(file_path) as img:
                    if img.format != 'PNG':
                        continue  # pomijamy inne formaty
            except:
                continue  # pomijamy uszkodzone pliki
                
            # Wyodrębnienie imienia i numeru z nazwy pliku, np. Jan3.png → Jan, 3
            name_part = os.path.splitext(filename)[0]
            name = ''.join([c for c in name_part if not c.isdigit()])
            number = ''.join([c for c in name_part if c.isdigit()])
            
            # Przetwarzanie obrazu
            processed_img, minutiae = self.process_image(file_path)
            
            # Dodanie do odpowiedniej grupy (wg imienia)
            self.signature_groups[name].append({
                'path': file_path,
                'processed_img': processed_img,
                'minutiae': minutiae,
                'number': number
            })
    
    def process_image(self, img_path):
        """Przetwarza obraz: binarizacja Otsu, szkieletyzacja, ekstrakcja minucji"""
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image file {img_path}")
        
        # Automatyczna binarizacja: podpis → białe linie na czarnym tle
        _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Szkieletyzacja: redukcja linii podpisu do 1-pikselowej szerokości
        skeleton = skeletonize(binary_img // 255)
        skeleton_img = img_as_ubyte(skeleton)
        
        # Ekstrakcja punktów charakterystycznych (końce i rozwidlenia linii)
        minutiae = self.extract_minutiae(skeleton_img)
        
        return skeleton_img, minutiae
    
    def extract_minutiae(self, skeleton_img):
        """Wyszukuje minucje (końce i rozwidlenia linii) na obrazie-szkielecie"""
        minutiae = []
        for i in range(1, skeleton_img.shape[0]-1):
            for j in range(1, skeleton_img.shape[1]-1):
                if skeleton_img[i,j] == 255:
                    # Wydzielenie sąsiadów (3x3) i zliczenie aktywnych pikseli
                    neighbors = skeleton_img[i-1:i+2, j-1:j+2]
                    num_neighbors = np.sum(neighbors) // 255 - 1
                    
                    # Punkt końcowy (1 sąsiad) lub rozwidlenie (3+ sąsiadów)
                    if num_neighbors == 1 or num_neighbors >= 3:
                        minutiae.append((j, i))  # Zapisujemy jako (x, y)
        return minutiae
    
    def compare_minutiae(self, minutiae1, minutiae2, threshold=10):
        """Porównuje dwa zestawy minucji, liczy dopasowania w zasięgu progu"""
        matches = 0
        used = set()
        
        for m1 in minutiae1:
            for idx, m2 in enumerate(minutiae2):
                if idx not in used:
                    # Obliczanie euklidesowej odległości pomiędzy punktami
                    distance = math.sqrt((m1[0]-m2[0])**2 + (m1[1]-m2[1])**2)
                    if distance < threshold:
                        matches += 1
                        used.add(idx)  # Unikamy wielokrotnego użycia tych samych punktów
                        break
        return matches
    
    def compare_with_database(self, query_minutiae):
        """Porównuje minucje zapytania z każdą grupą podpisów w bazie"""
        results = {}
        for name, signatures in self.signature_groups.items():
            total_matches = 0
            for sig in signatures:
                matches = self.compare_minutiae(query_minutiae, sig['minutiae'])
                total_matches += matches
            
            avg_matches = total_matches / len(signatures) if signatures else 0
            results[name] = avg_matches
        return results


# Klasa interfejsu graficznego
class SignatureGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Signature Recognition")
        self.root.configure(bg="#f0f0f0")
        self.recognizer = SignatureRecognizer()
        self.setup_ui()  # Konfiguracja UI
    
    def setup_ui(self):
        """Tworzy elementy interfejsu: przyciski, etykiety, wykres"""
        button_frame = Frame(self.root, bg="#f0f0f0")
        title_label = Label(self.root, text="Signature Recognition System",
                    font=("Segoe UI", 16, "bold"), bg="#f0f0f0", fg="#333")
        title_label.pack(pady=10)
        button_frame.pack(pady=10)
        
        self.select_btn = Button(button_frame, text="Select Signature", command=self.load_signature,
        font=("Segoe UI", 10, "bold"), bg="#4CAF50", fg="white", padx=10, pady=5)
        self.select_btn.pack(side="left", padx=5)
        
        self.compare_btn = Button(button_frame, text="Compare", command=self.compare_signature, state="disabled",
        font=("Segoe UI", 10, "bold"), bg="#2196F3", fg="white", padx=10, pady=5)
        self.compare_btn.pack(side="left", padx=5)
        
        self.image_label = Label(self.root)
        self.image_label.pack(pady=10)
        
        self.result_label = Label(self.root, text="Select signature to compare", bg="#f0f0f0", font=("Segoe UI", 11))
        self.result_label.pack(pady=10)
        
        self.plot_frame = Frame(self.root, bg="#f0f0f0")
        self.plot_frame.pack(pady=10)
    
    def load_signature(self):
        """Wczytuje podpis użytkownika do porównania z bazą"""
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.*")])
        if file_path:
            try:
                with Image.open(file_path) as img:
                    if img.format != 'PNG':
                        messagebox.showerror("Error", "Please select a PNG format image")
                        return
                
                self.processed_img, self.query_minutiae = self.recognizer.process_image(file_path)
                
                # Wyświetlenie przetworzonego obrazu w GUI
                img = Image.fromarray(self.processed_img)
                img.thumbnail((300, 300))
                photo = ImageTk.PhotoImage(img)
                
                self.image_label.config(image=photo)
                self.image_label.image = photo
                
                self.compare_btn.config(state="normal")
                self.result_label.config(text="Signature loaded. Click 'Compare' to analyze.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process image: {str(e)}")
    
    def compare_signature(self):
        """Porównuje aktualnie wczytany podpis z bazą danych"""
        if hasattr(self, 'query_minutiae'):
            if not self.recognizer.signature_groups:
                messagebox.showwarning("Warning", "Database is empty. Add signatures to the Signature folder.")
                return
                
            results = self.recognizer.compare_with_database(self.query_minutiae)
            self.plot_results(results)  # Wykres wyników
            
            # Znalezienie najlepszego dopasowania
            best_match = max(results.items(), key=lambda x: x[1]) if results else (None, 0)
            self.result_label.config(text=f"Best match: {best_match[0]} (score: {best_match[1]:.2f})")
    
    def plot_results(self, results):
        """Tworzy wykres słupkowy z wynikami dopasowania"""
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        fig = Figure(figsize=(10, 5), dpi=100)
        ax = fig.add_subplot(111)

        names = list(results.keys())
        values = list(results.values())

        ax.bar(names, values)
        ax.set_xlabel('Signature Groups')
        ax.set_ylabel('Similarity Score')
        ax.set_title('Signature Comparison Results')
        ax.tick_params(axis='x', rotation=45)

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

        # Ponowne wypisanie najlepszego wyniku (na wypadek użycia tej funkcji osobno)
        best_match = max(results.items(), key=lambda x: x[1]) if results else (None, 0)
        self.result_label.config(text=f"Best match: {best_match[0]} (score: {best_match[1]:.2f})")


# Uruchomienie aplikacji
if __name__ == "__main__":
    root = Tk()
    app = SignatureGUI(root)
    root.mainloop()
