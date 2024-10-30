enlace dataset: https://universe.roboflow.com/zalfon/platerecsystem-jk0bw/dataset/2

comando:
```bash
 yolo detect train model=../yolo11n-license_plate.pt data=data.yaml imgsz=416 batch=4 device=CPU epochs=100
```

# Práctica 4 - Reconocimiento de matrículas y trackeo de instancias

## Tabla de contenidos

- [Práctica 4 - Reconocimiento de matrículas y trackeo de instancias](#práctica-4---reconocimiento-de-matrículas-y-trackeo-de-instancias)
  - [Tabla de contenidos](#tabla-de-contenidos)
  - [Introducción](#introducción)
  - [Desarrollo](#desarrollo)
    - [Detectar las instancias que buscamos](#detectar-las-instancias-que-buscamos)
    - [Detectar la posición de la matrícula](#detectar-la-posición-de-la-matrícula)
      - [Entrenar el modelo](#entrenar-el-modelo)
      - [Usar el modelo](#usar-el-modelo)
  - [Conclusiones](#conclusiones)


## Introducción

Esta práctica consiste en, mediante el uso de diferentes tecnologías (entre ellas *redes neuronales*), trackear tanto los vehículos como las personas presentes en el vídeo que se nos proporciona. Además, en los vehículos que la posean, leer la matrícula.

## Desarrollo

Haremos uso de un modelo llamado **YOLO** y del OCR llamado **EasyOCR**.

### Detectar las instancias que buscamos

El primer paso es usar el modelo YOLO ya entrenado y usar las etiquetas que queremos para que nos seleccione todas las instancias.

Queremos que nos aparezca en la imagen original una caja que encierre a dicha instancia, asociarle un ID (para ello trackearemos). Asimismo, en el video mostraremos tanto el rectángulo que encierra a la instancia como su identificador y su clase, variando en color para cada instancia.

### Detectar la posición de la matrícula

#### Entrenar el modelo


#### Usar el modelo

Una vez detectado lo que queremos, en aquellos que sean vehículos (en nuestro caso las clases 2, 3 y 5), recortamos la caja que los contienen, y se la pasamos a nuestra segunda IA. Esta nos identificará (en el caso de que la tenga) la posición de la matrícula.

## Conclusiones
