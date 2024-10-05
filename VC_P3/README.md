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