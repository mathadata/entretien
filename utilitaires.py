import pickle
from zipfile import ZipFile
from io import BytesIO, StringIO
import requests
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Javascript, HTML, display_html

### --- IMPORT DES DONNÉES ---
# Téléchargement et extraction des inputs contenus dans l'archive zip

print("Chargement de la base de donnée d'images en cours...")

inputs_zip_url = "https://raw.githubusercontent.com/akimx98/challenge_data/main/input_mnist_2.zip"
inputs_zip = requests.get(inputs_zip_url)
zf = ZipFile(BytesIO(inputs_zip.content))
zf.extractall()
zf.close()

# # Inputs
with open('mnist_2_x_train.pickle', 'rb') as f:
    ID_train, d_train = pickle.load(f).values()

image = d_train[10,:,:].copy()

print("Images chargées !") 

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            if np.isnan(obj):
                return None
            return int(obj)
        if isinstance(obj, np.floating):
            if np.isnan(obj):
                return None
            return float(obj)
        if isinstance(obj, np.ndarray):
            return np.where(np.isnan(obj), None, obj).tolist()
        return super(NpEncoder, self).default(obj)


def run_js(js_code):
    display(Javascript(js_code))

def imshow(ax, image, **kwargs):
    ax.imshow(image, cmap='gray', vmin=0, vmax=255, extent=[0, 28, 28, 0], **kwargs)

def affichage(image):
    fig, ax = plt.subplots()
    imshow(ax, image)
    ax.set_xticks(np.arange(0,28,5))
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top') 
    plt.show()
    plt.close()


pd.set_option('display.max_rows', 28)
pd.set_option('display.max_columns', 28)
def affichage_tableau(image):
    df = pd.DataFrame(image)
    display(df)
    return
