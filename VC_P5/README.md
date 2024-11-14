# Practica 5

El primer paso es la selección de la imagen, en este caso, vamos a colocar una gorra sobre las cabezas. Para lo cual, hemos seleccionado la siguiente image:

![Gorra](./images/cap.png)

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