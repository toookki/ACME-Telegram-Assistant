"""
modules/embeddings.py
=====================
Módulo de embeddings semánticos.

Responsabilidades:
- Cargar el modelo de sentence-transformers.
- Generar embeddings para todos los productos del catálogo.
- Guardar y cargar caché en disco para evitar recalcular en cada inicio.
- Exponer una función para generar embeddings de consultas en tiempo real.

El embedding convierte texto en un vector numérico.
Dos textos semánticamente similares tendrán vectores cercanos en el espacio,
aunque no compartan palabras exactas, por ejemplo:
"algo para el frío" ≈ "parka invierno".
"""

import os
import json
import pickle

import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import (
    PRODUCTOS_PATH,
    EMBEDDINGS_CACHE_PATH,
    EMBEDDING_MODEL,
)


# Modelo de embeddings cargado en memoria una sola vez.
_model = None

# Matriz de embeddings de productos con shape:
# (N_productos, embedding_dim).
_product_embeddings = None

# Lista de diccionarios con los productos del catálogo.
_products = None


def _get_model() -> SentenceTransformer:
    """
    Carga y retorna el modelo de embeddings usando un patrón singleton.

    El modelo se inicializa solo la primera vez que se solicita y luego
    se reutiliza durante toda la ejecución del programa.

    Returns:
        Instancia cargada de SentenceTransformer.
    """
    global _model

    if _model is None:
        print(f"Cargando modelo de embeddings: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        print("Modelo cargado.")

    return _model


def _product_to_text(product: dict) -> str:
    """
    Convierte un producto en una representación textual para embeddear.

    Concatena los campos más relevantes para la búsqueda semántica:
    nombre, categoría, descripción, marca, tags y colores.

    Args:
        product: Diccionario con la información del producto.

    Returns:
        Texto consolidado utilizado para generar el embedding del producto.
    """
    parts = [
        product.get("nombre", ""),
        product.get("categoria", ""),
        product.get("descripcion", ""),
        product.get("marca", ""),
        " ".join(product.get("tags", [])),
        " ".join(str(c) for c in product.get("colores", [])),
    ]

    return " ".join(p for p in parts if p)


def load_embeddings() -> tuple[list, np.ndarray]:
    """
    Carga o genera los embeddings de todos los productos del catálogo.

    El procedimiento sigue este orden:
    1. Reutiliza los embeddings ya cargados en memoria, si existen.
    2. Intenta cargar productos y embeddings desde la caché en disco.
    3. Si no existe caché, carga el catálogo, genera los embeddings
       y guarda el resultado para futuras ejecuciones.

    Returns:
        Tupla con:
        - products: Lista de diccionarios con los productos del catálogo.
        - embeddings_matrix: Matriz NumPy con shape
          (N_productos, embedding_dim).
    """
    global _product_embeddings, _products

    # Reutilizar embeddings ya cargados en memoria.
    if _product_embeddings is not None:
        return _products, _product_embeddings

    # Intentar cargar desde caché en disco.
    if os.path.exists(EMBEDDINGS_CACHE_PATH):
        print("Cargando embeddings desde cache...")

        with open(EMBEDDINGS_CACHE_PATH, "rb") as f:
            cache = pickle.load(f)

        _products = cache["products"]
        _product_embeddings = cache["embeddings"]

        print(f"{len(_products)} productos cargados desde cache.")

        return _products, _product_embeddings

    # Generar embeddings desde cero.
    print("Generando embeddings de productos...")

    with open(PRODUCTOS_PATH, "r", encoding="utf-8") as f:
        _products = json.load(f)

    model = _get_model()
    texts = [_product_to_text(p) for p in _products]

    _product_embeddings = model.encode(
        texts,
        show_progress_bar=True,
    )

    # Guardar caché en disco.
    os.makedirs(
        os.path.dirname(EMBEDDINGS_CACHE_PATH),
        exist_ok=True,
    )

    with open(EMBEDDINGS_CACHE_PATH, "wb") as f:
        pickle.dump(
            {
                "products": _products,
                "embeddings": _product_embeddings,
            },
            f,
        )

    print(
        f"Embeddings generados y guardados para "
        f"{len(_products)} productos."
    )

    return _products, _product_embeddings


def embed_query(query: str) -> np.ndarray:
    """
    Genera el embedding de una consulta del usuario en tiempo real.

    Utiliza el mismo modelo empleado para los productos, de modo que
    consulta y catálogo queden representados en el mismo espacio vectorial.

    Args:
        query: Texto de la consulta del usuario.

    Returns:
        Vector NumPy correspondiente al embedding de la consulta.
    """
    model = _get_model()
    return model.encode([query])[0]