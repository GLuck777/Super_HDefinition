* Prérequis :
Assurez-vous d'avoir Python installé (version 3.8 ou supérieure recommandée) et les bibliothèques nécessaires. Créez un fichier requirements.txt dans votre dossier super_resolution :

```bash
torch
torchvision
pillow
numpy
```

* Installation des dépendances :
Ouvrez un terminal/invite de commandes, naviguez jusqu'au dossier super_resolution, et exécutez :

```bash
# Créer un environnement virtuel (optionnel mais recommandé)
python -m venv venv
# Activer l'environnement virtuel
# Sur Windows :
venv\Scripts\activate
# Sur macOS/Linux :
source venv/bin/activate
source .venv/bin/activate
#pour sortir de l'environnement virtuel -> deactivate


# Installer les dépendances
pip install -r requirements.txt
pip install 'git+https://github.com/xinntao/Real-ESRGAN.git'
```

* Exécution du projet :
Vous pouvez démarrer l'application en exécutant directement le script Python :
ver1.0
```bash
python image-upscaler-gui.py
```
ver1.5
```bash
python image-upscaler-gui-enhenced.py
```
ver2.0
```bash
python image-upscaler-ml.py
```
* Structure du projet :
```

super_resolution/
│
├── venv/                # Environnement virtuel (optionnel)
├── image_upscaler.py    # Script de traitement d'image
├── image_upscaler_gui.py# Interface graphique
├── requirements.txt     # Dépendances du projet
└── README.md            # Instructions d'utilisation
```

* Résolution de problèmes potentiels :


Assurez-vous d'avoir Python installé
Vérifiez que toutes les dépendances sont installées
Si vous avez une erreur liée à PyTorch, consultez le site officiel pour l'installation spécifique à votre système


Options d'installation de PyTorch :
Si l'installation via pip pose problème, visitez le site officiel de PyTorch (pytorch.org) pour des instructions spécifiques à votre système.

Commande rapide d'installation de PyTorch (à adapter selon votre système) :

```bash
pip3 install torch torchvision
pip install torch torchvision pillow
```