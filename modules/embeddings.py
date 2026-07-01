"""
modules/embeddings.py
=====================
Módulo de embeddings semánticos.

Responsabilidades:
- Cargar el modelo de sentence-transformers.
- Generar embeddings para todos los productos del catálogo.
- Guardar/cargar cache en disco para no recalcular en cada inicio.
- Exponer función para embeddear consultas del usuario en tiempo real.

El embedding convierte texto en un vector numérico.
Dos textos semánticamente similares tendrán vectores cercanos en el espacio,
aunque no compartan palabras exactas (ej: "algo para el frío" ≈ "parka invierno").
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


# Variable global para el modelo (se carga una sola vez)
_model = None
_product_embeddings = None  # np.array de shape (N_productos, embedding_dim)
_products = None            # Lista de dicts de productos


def _get_model() -> SentenceTransformer:
    """Carga el modelo de embeddings (singleton)."""
    global _model
    if _model is None:
        print(f"Cargando modelo de embeddings: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        print("Modelo cargado.")
    return _model


def _product_to_text(product: dict) -> str:
    """
    Convierte un producto en una cadena de texto para embeddear.
    Concatena los campos más relevantes para la búsqueda semántica.
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


def load_embeddings():
    """
    Carga o genera los embeddings de todos los productos.
    Si existe el cache en disco, lo usa. Si no, los calcula y guarda.

    Retorna:
        tuple: (products, embeddings_matrix)
            - products: lista de dicts con los datos de cada producto
            - embeddings_matrix: np.array shape (N, dim)
    """
    global _product_embeddings, _products

    if _product_embeddings is not None:
        return _products, _product_embeddings

    # Intentar cargar desde cache
    if os.path.exists(EMBEDDINGS_CACHE_PATH):
        print("Cargando embeddings desde cache...")
        with open(EMBEDDINGS_CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        _products = cache["products"]
        _product_embeddings = cache["embeddings"]
        print(f"{len(_products)} productos cargados desde cache.")
        return _products, _product_embeddings

    # Generar embeddings desde cero
    print("Generando embeddings de productos...")
    with open(PRODUCTOS_PATH, "r", encoding="utf-8") as f:
        _products = json.load(f)

    model = _get_model()
    texts = [_product_to_text(p) for p in _products]
    _product_embeddings = model.encode(texts, show_progress_bar=True)

    # Guardar cache
    os.makedirs(os.path.dirname(EMBEDDINGS_CACHE_PATH), exist_ok=True)
    with open(EMBEDDINGS_CACHE_PATH, "wb") as f:
        pickle.dump({"products": _products, "embeddings": _product_embeddings}, f)
    print(f"Embeddings generados y guardados para {len(_products)} productos.")

    return _products, _product_embeddings


def embed_query(query: str) -> np.ndarray:
    """
    Genera el embedding de una consulta del usuario en tiempo real.

    Args:
        query: Texto de la consulta.

    Returns:
        np.ndarray: Vector del embedding de la consulta.
    """
    model = _get_model()
    return model.encode([query])[0]
