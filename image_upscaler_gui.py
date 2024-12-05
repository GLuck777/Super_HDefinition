import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageTk
import os
import platform
from tkinter import simpledialog


class SuperResolutionModel(nn.Module):
    def __init__(self, scale_factor=2):
        super(SuperResolutionModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=5, padding=2)
        )
        self.scale_factor = scale_factor

    def forward(self, x):
        return self.conv_layers(x)

class ImageUpscalerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Super-Résolution d'Images")
        self.root.geometry("600x800")

        # Configuration du modèle
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SuperResolutionModel().to(self.device)

        # Variables
        self.input_image_path = tk.StringVar()
        self.output_image_path = tk.StringVar()

        # Création de l'interface
        self.create_widgets()

    def create_widgets(self):
        # Cadre de sélection d'image
        input_frame = tk.LabelFrame(self.root, text="Image Originale")
        input_frame.pack(padx=10, pady=10, fill="x")

        # Bouton de sélection d'image
        tk.Button(input_frame, text="Sélectionner Image", command=self.select_input_image).pack(pady=10)

        # Prévisualisation de l'image originale
        self.original_image_label = tk.Label(input_frame)
        self.original_image_label.pack(pady=10)

        # Informations sur l'image originale
        self.original_info_label = tk.Label(input_frame, text="")
        self.original_info_label.pack(pady=5)

        # Cadre de configuration
        config_frame = tk.LabelFrame(self.root, text="Configuration de la Super-Résolution")
        config_frame.pack(padx=10, pady=10, fill="x")

        # Mode de mise à l'échelle
        tk.Label(config_frame, text="Mode de mise à l'échelle :").pack()
        self.scale_mode = tk.StringVar(value="facteur")
        scale_mode_frame = tk.Frame(config_frame)
        scale_mode_frame.pack(pady=5)
        
        tk.Radiobutton(scale_mode_frame, text="Par facteur", variable=self.scale_mode, value="facteur").pack(side=tk.LEFT, padx=10)
        tk.Radiobutton(scale_mode_frame, text="Par résolution", variable=self.scale_mode, value="resolution").pack(side=tk.LEFT)

        # Sélection du facteur ou de la résolution
        self.scale_input = tk.StringVar()
        self.scale_entry = tk.Entry(config_frame, textvariable=self.scale_input, width=20)
        self.scale_entry.pack(pady=5)
        
        # Label dynamique
        self.scale_label = tk.Label(config_frame, text="Facteur de mise à l'échelle (ex: 2, 3, 4)")
        self.scale_label.pack()

        # Binding pour mettre à jour le label
        self.scale_mode.trace_add('write', self.update_scale_label)

        # Bouton de traitement
        tk.Button(self.root, text="Augmenter la Résolution", command=self.upscale_image).pack(pady=10)

        # Cadre de résultat
        result_frame = tk.LabelFrame(self.root, text="Image Upscalée")
        result_frame.pack(padx=10, pady=10, fill="x")

        # Prévisualisation de l'image upscalée
        self.upscaled_image_label = tk.Label(result_frame)
        self.upscaled_image_label.pack(pady=10)

        # Informations sur l'image upscalée
        self.upscaled_info_label = tk.Label(result_frame, text="")
        self.upscaled_info_label.pack(pady=5)

        # Bouton de sauvegarde
        tk.Button(result_frame, text="Sauvegarder Image", command=self.save_image).pack(pady=10)

    def update_scale_label(self, *args):
        """Met à jour le label en fonction du mode de mise à l'échelle"""
        if self.scale_mode.get() == "facteur":
            self.scale_label.config(text="Facteur de mise à l'échelle (ex: 2, 3, 4)")
        else:
            self.scale_label.config(text="Résolution cible (ex: 720, 1080)")

    def select_input_image(self):
        # Détermine le répertoire initial en fonction de l'OS
        if platform.system() == "Windows":
            initial_dir = os.path.join(os.environ['USERPROFILE'], 'Pictures') # Répertoire "Images" sur Windows
        else:
            initial_dir = os.path.expanduser("~/Pictures")  # Répertoire "Images" sur Linux/Unix

        if not os.path.exists(initial_dir):
            initial_dir = os.path.expanduser("~")
        # Ouvre la boîte de dialogue pour sélectionner une image
        file_path = filedialog.askopenfilename(
            initialdir=initial_dir,
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if file_path:
            self.input_image_path.set(file_path)
            self.display_input_image(file_path)
        # file_path = filedialog.askopenfilename(
        #     # initialdir='/home/zone01-m14/Pictures/',
        #     initialdir='/home/',
        #     filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        # )
        # if file_path:
        #     self.input_image_path.set(file_path)
        #     self.display_input_image(file_path)

    def display_input_image(self, file_path):
        image = Image.open(file_path)
        # Affichage de l'image originale
        display_image = image.copy()
        display_image.thumbnail((400, 300))
        photo = ImageTk.PhotoImage(display_image)
        self.original_image_label.config(image=photo)
        self.original_image_label.image = photo

        # Affichage des informations sur l'image
        info_text = f"Dimensions originales : {image.width}x{image.height}"
        self.original_info_label.config(text=info_text)

    def upscale_image(self):
        if not self.input_image_path.get():
            messagebox.showerror("Erreur", "Veuillez sélectionner une image")
            return

        try:
            # Charger l'image
            image = Image.open(self.input_image_path.get()).convert('RGB')
            
            # Calculer la nouvelle résolution
            if self.scale_mode.get() == "facteur":
                # Mode facteur
                try:
                    scale_factor = float(self.scale_input.get())
                    new_width = int(image.width * scale_factor)
                    new_height = int(image.height * scale_factor)
                except ValueError:
                    messagebox.showerror("Erreur", "Veuillez entrer un facteur valide")
                    return
            else:
                # Mode résolution cible
                try:
                    target_resolution = int(self.scale_input.get())
                    width_factor = target_resolution / image.width
                    height_factor = target_resolution / image.height
                    scale_factor = min(width_factor, height_factor)
                    new_width = int(image.width * scale_factor)
                    new_height = int(image.height * scale_factor)
                except ValueError:
                    messagebox.showerror("Erreur", "Veuillez entrer une résolution valide")
                    return

            # Redimensionner l'image
            upscaled_image = image.resize((new_width, new_height), Image.LANCZOS)
            
            # Sauvegarde temporaire
            # output_path = os.path.join(
            #     os.path.dirname(self.input_image_path.get()), 
            #     f"upscaled_{os.path.basename(self.input_image_path.get())}"
            # )
            # upscaled_image.save(output_path)
            # self.output_image_path.set(output_path)
            save_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("Tous fichiers", "*.*")])
            if save_path:
                upscaled_image.save(save_path)
                self.output_image_path.set(save_path)
                messagebox.showinfo("Succès", f"Image sauvegardée à : {save_path}")
            else:
                messagebox.showwarning("Attention", "Sauvegarde annulée")
                return

            # Affichage de l'image upscalée
            display_upscaled = upscaled_image.copy()
            display_upscaled.thumbnail((400, 300))
            photo = ImageTk.PhotoImage(display_upscaled)
            self.upscaled_image_label.config(image=photo)
            self.upscaled_image_label.image = photo

            # Affichage des informations sur l'image upscalée
            info_text = f"Nouvelles dimensions : {new_width}x{new_height}"
            self.upscaled_info_label.config(text=info_text)

            messagebox.showinfo("Succès", "Image upscalée avec succès!")

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors du traitement : {str(e)}")

    def save_image(self):
        if not self.output_image_path.get():
            messagebox.showerror("Erreur", "Aucune image à sauvegarder")
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("Tous fichiers", "*.*")]
        )
        if save_path:
            Image.open(self.output_image_path.get()).save(save_path)
            messagebox.showinfo("Succès", f"Image sauvegardée à : {save_path}")

def main():
    root = tk.Tk()
    app = ImageUpscalerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
