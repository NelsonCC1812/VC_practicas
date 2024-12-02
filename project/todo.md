# Todo

## Train prev
- [ ] Extraer dataset
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
   │─── face_images/
   └─── normalized_face_images/
```