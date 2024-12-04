# Todo

## Train prev
- [x] Extraer dataset
  - [x] Generar datasets
- [ ] Generar imagenes de las caras
  - [ ] Pasarle el face detection al dataset
  - [ ] Guardar las subimagenes correspondientes a cada una de las caras
- [ ] Generar las entradas al model
  - [ ] Aplicar normalizacion a las subimagenes
  - [ ] Guardar las subimagenes de las caras tras la normalizacion
- **<p style="color:red;font-size: 1.5rem">Time to train!!</p>**

# Notas

* Podriamos utilizar uan red convolucional preentranada como extractor de caracteristicas
    * En ese caso, podriamos guardar las salidas de la red convolucional para cada una de las imagenes normalizadas
* **Filtros**:
  * Face alignment
  * Equalizado
  * Blanco y negro

# Data filetree
```
─── data/
   ├─── dataset/
   ├─────── selfie_id.csv
   ├─────── Selfies ID Images dataset/...
   │─── face_images/
   ├─────── dataset.csv
   ├─────── data/...
   │─── normalized_face_images/
   ├─────── dataset.csv
   └─────── data/...
```


# Code nn imports
```py
%matplotlib inline


import torch
import torch.nn as nn
import torn.nn.functional as F
from torch import optim, transforms
from torch.utils.data import Dataset, Dataloader
from torchvision.io import read_image, ImageReadMode

import sklearn.metrics as metrics
import seaborn as sns
```

```py
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"> Torch version: {torch.__version__}")
print(f"> Dispositivo configurado para usar: {device}")
print(torch.cuda.get_device_name(0))
```


```py
IMAGE_SIZE = 220
BATCH_SIZE = 32


# hyperparams ---

NUM_EPOCH = 100
LEARNING_RATE = .001
MOMENTUM = .9
EARLY_PATIENCE = 10
SCHEDULER_PATIENCE = 6

DROPOUTS = [.5, .75]
```


* data_procesing.ipynb
* face_extractor.ipynb
* train_prevs.ipynb
* model_generation.ipynb
* model.py