# Practica 5

> Alumno: *Nelson Cabrera Cano*

# Tabla de contenidos

- [Practica 5](#practica-5)
- [Tabla de contenidos](#tabla-de-contenidos)
  - [Elecciones](#elecciones)
    - [Overlay](#overlay)
    - [Detector](#detector)
    - [Input](#input)
    - [Variables de control](#variables-de-control)
  - [Procedimiento](#procedimiento)
    - [1. Cargar la imagen que usaremos como filtro](#1-cargar-la-imagen-que-usaremos-como-filtro)
    - [2. Cargar el input](#2-cargar-el-input)
    - [3. Extraemos caras](#3-extraemos-caras)
    - [Extraemos los datos de cada cara y aplicamos el filtro](#extraemos-los-datos-de-cada-cara-y-aplicamos-el-filtro)
    - [Consideraciones](#consideraciones)
      - [Creación de un VideoWriter](#creación-de-un-videowriter)
      - [Contador](#contador)
  - [Resultados](#resultados)
  - [Notas](#notas)

## Elecciones
### Overlay

El primer paso es la selección de la imagen, en este caso, vamos a colocar una gorra sobre las cabezas. Para lo cual, hemos seleccionado la siguiente image:

![Gorra](./images/cap.png)

### Detector

En este caso, por recomendación del profesor, vamos a utilizar como detector **MTCNN**, el cua solo extrae los siguientes datos:

```
x
y
w
h
right_eye:
    x
    y
left_eye:
    x
    y
```

Por lo que solo podemos extraer el rectángulo que contiene a la cara objetivo y la posición puntual de los ojos.

### Input

Debido a la potencia que necesita el detector, y a que no disponemos de mucha potencia de cómputo, hemos decidido utilizar un video como input (tampoco me hacia mucha gracia grabarme, y menos poniéndome filtros). Del siguiente [video de YouTube](https://www.youtube.com/watch?v=YzcawvDGe4Y), hemos extraído un fragmento que cumpla con las especificaciones que se pedían (queríamos valorar como se comportaba con varias caras que se viesen claramente).

Como utilizamos un Mac para hacer los cortes, el archivo salió en formato **MOV**, y este formato daba muchos problemas al pasarlos por el programa (el programa dejaba de funcionar). Por lo que optamos por usar una herramienta online para convertir el video de formato **MOV** a formato **MP4**, el cual si funciono de la forma correcta.

Este es el [video](https://alumnosulpgc-my.sharepoint.com/:v:/g/personal/nelson_cabrera101_alu_ulpgc_es/ES4sZmMdk5BGnLw-qhZGvZkBeaNeU3sES_Stfqw76xodIA?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=XdERhZ) que utilizamos como input.


### Variables de control

Se han definido una serie de variables de control para facilitar el correcto uso del programa, el proceso de debugueo, así como los correctos ajustes finales de aplicación del filtro.

* **DEBUG_MODE** : EL modo debug muestra todos los parámetros obtenidos de las caras detectadas en la salida de video. Nótese que sin mostrar el video esto carece de sentido.
* **CREATE_VID** : En función de esta variable se creará o no un archivo de salida, con las modificaciones aplicadas por el filtro o la muestra del modo debug.
* **SHOW_VID** : En función de esta variable se mostrará o no la salida de video con forme el programa se esté ejecutando, si tenemos esta opción activada, podremos parar cuando queramos el programa pulsando la tecla **ESC**.
  
* **CAP_FACE_PROPORTION** : La proporción de la imagen de la gorra respecto al ancho de la cara detectada, en este caso, hemos usado un valor de **1.4**.
  
* **VIDEO_INPUT** : El nombre del archivo de video utilizado como entrada, en el caso de que no proporcionemos ninguno (**''**), se utilizara la entrada por defecto, es decir, se tomará la camada por defecto del dispositivo utilizado, que en nuestro caso se trata de la **WebCam**.
  
* **XOF** : Offset en el eje de la **X**, esto se resta al valor de la posición original. En este caso, hemos ajustado, hasta quedarnos con que el valor que nos gusta es **.1**. Nótese, que este valor es proporcional al **ancho** de la detección de la cara.
* **OF** : Offset en el eje de la **Y**, esto se resta al valor de la posición original. En este caso, hemos ajustado, hasta quedarnos con que el valor que nos gusta es **.5**. Nótese, que este valor es proporcional al **.5** de la detección de la cara. 

## Procedimiento

Aunque el código fue ideado en un inicio para usarlo en local, hemos tenido que acabar usando **Google Colab** por lo que se usaron algunas modificaciones, como añadir las líneas para que el interprete de **Colab** instalará en el entorno las librerías necesarias, y también para que usara la **GPU**.

### 1. Cargar la imagen que usaremos como filtro

En este caso hemos optado por mantener la imagen original, dado que si hacemos muchos reescalados a diferentes tamaños acabaremos con una imagen de muy baja resolución.

Es muy importante que en este paso nos ocupemos de guardar la proporción de la imagen, la relación de aspecto, la relación que existe entre el ancho y el alto; así cuando hagamos una modificación en uno de estos parámetros tendremos que calcular el otro en con este valor, evitando así que hayan deformaciones en la imagen.

```python
ocap = cv2.imread('images/cap.png', cv2.IMREAD_UNCHANGED)
cap_prop = len(ocap[0])/len(ocap)
```

> Es importante tener en cuenta que la imagen esta en formato **PNG**, tiene un canal extra para el alpha, la transparencia. Lo necesitamos para poder aplicar la máscara, por lo que debemos cargar la imagen completa sin transformaciones.

### 2. Cargar el input

Es importante, para varias cosas que en este paso, guardemos los FPS a los que esta la (y ya que no salen como un numero entero, que lo aproximemos como tal), como indicamos antes, podemos usar un video o la cámara.

```python
vid = cv2.VideoCapture( VIDEO_INPUT or 0)
FPS = int(vid.get(cv2.CAP_PROP_FPS))
```

### 3. Extraemos caras

Por cada frame, extraemos todas las caras que contiene, es importante poner el valor **enforce_detection** a **False** ya que, no en todas los frames encontraremos caras. En este caso, como ya hemos comentado usaremos como detector **MTCNN**.


### Extraemos los datos de cada cara y aplicamos el filtro

Extraemos todos los datos posibles de cada cara, y nos disponemos a aplicar el filtro.

1. Aplicamos un cambio de tamaño de la imagen para adecuarla a la cara detectada y manteniendo la relación de aspecto de la misma.
2. Debido a los problemas que habría en los bordes, hemos optado por utilizar la cláusula **try**, para evitar el fin del programa en los bordes.
3. Separamos los canales **rgb** y el canal alpha con la máscara.
4. Aplicamos el filtro (la imagen) por medio del uso de una máscara y su inversa, a la imagen original y a la imagen de overlay.


### Consideraciones

#### Creación de un VideoWriter

En el caso de que queramos crear un video de salida, debemos crear un VideoWriter, con los parámetros adecuados (ancho, alto y fpss de la imagen original).


#### Contador

Utilizamos un contador para poder ver el progreso de la creación del video de salida, en este caso, hemos optado porque avise de cada segundo de video creado.

## Resultados

Tras todo este proceso tenemos como resultado este [video](https://alumnosulpgc-my.sharepoint.com/:v:/g/personal/nelson_cabrera101_alu_ulpgc_es/EWK5TFsBQ4ZKmlgQ9ojqrMwBSqeWyIkIpPCQ0wizpgJMhQ?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=hEhaXq)

Cuando la cara se ve claramente, se tiene un muy buen resultado. Es si, en cabezas que no estén totalmente verticales y en aquellas personas que agachan la cabeza, podemos ver las costuras del sistema que estamos utilizando.

Aquí otro ejemplo:  
![Ejemplo](./output-gif.gif)


## Notas

Es un proceso altamente costoso en recursos y tiempo, en **Google Colab** con este sistema, y usando **GPU** tardo más de una hora, a un aproximado de 2min de trabajo por cada segundo de video creado.
