# Práctica 4

## Contador de monedas

### Introducción
Esta práctica se trata de, dada una imagen de un conjunto de monedas, saber cuanto dinero hay en ella. Para ello se utiliza como referencia una moneda de valor conocido (para tomar su tamaño como referencia), seleccionándola en la imagen.

#### Desarrollo

Utilizando la imagen proporcionada por el profesor como referencia, he implementado todo el algoritmo necesario, y ajustado el parámetro del gap, entre en valor real de la moneda y una pequeña variación respecto a este valor, hasta tomar el valor más alto posible que sea capaz de dar el valor correcto.

Tras este proceso, se ha acabado con un valor de gap de un **0.04**, es decir, un **4%**, quizás muy poco margen, pero las variaciones entre monedas son realmente muy justas.

Hecho esto, se ha probado con otras fotos, dando un resultado incluso mejor del esperado, ya que incluso tomando a las sombras como parte del contorno de la moneda, daba el valor correcto.

#### Uso del contador

1. Aparece una imagen con las monedas y sus contornos definidos.
2. **Debemos seleccionar una moneda de `1€`**
    * Al seleccionar un punto, se pintará de azul
    * Para confirmar la selección debemos pulsar la tecla **ESC**
    * Si no es un punto valido (dentro de un contorno), volverá a aparecer la imagen
    * Si es un punto válido desaparecerá la imagen y continuará el proceso.
3. Se mostrará el resultado de la suma de los valores de las monedas.

#### Demostración

Esta es la imagen que saque yo (y su versión en escala de grises):  
![Imagen de monedas](./readme/monedas.png)![Imagen en escala de grises](./readme/monedas_gris.png)  
Como puede verse no es la mejor de las imágenes, y más adelante veremos que no se detectan bien los contornos.  

Ahora sacamos el histograma, aunque en este caso hemos usado **OTSU**, esta bien saber como es el histograma de la imagen:  
![Histograma](./readme/monedas_histograma.png) 

Ahora calculamos la versión umbralizada de la imagen, en este caso usando **OTSU** que calcula de forma automática el umbral:  
![Imagen umbralizada](./readme/monedas_umbralizada.png)  
Como vemos, no detecta bien los bordes de la imagen, se vera aún mejor en la siguiente imagen. Una vez calculado el umbral, calculamos los contornos exteriores, dibujando dichos contornos a la imagen original, tenemos lo siguiente:  
![Imagen con los contornos detectados](./readme/monedas_contornos.png)  

Pese a no detectar los contornos de forma perfecta, tenemos que el resultado del valor total de las monedas ha sido correco, presumiblemente debido a la correcta elección del GAP y a que la imagen no ha sido tan mala. (Teniendo por supuesto en cuenta que no existen monedas superpuestas, y otros detalles y/o imperfecciones en las monedas o en la imagen).
