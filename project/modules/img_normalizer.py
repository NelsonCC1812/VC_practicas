# para normalizar
normalizer()

# para aplicar 1 todos los filtros (lo mismo que normalizar)
normalizer.apply(img, filter=normalizer.filters.all)

# para aplicar 1 concreto
normalizer.apply(img, filter=normalizer.filters.filter_name)

con propiedad normalization[] con la lista ordenada de los filtros que se aplicaran ordenados