"""
modules/search.py
=================
Motor de búzzsqueda híbrida de productos.

Combina tres estrategias:
  1. Búsqueda semántica (embeddings + cosine similarity).
  2. Filtros estructurados (marca, color, talla, precio).
  3. Búsqueda por palabras clave (en nombre, tags y categoría).

El score final se calcula como:
  score = 0.6 * semantic_score + 0.4 * keyword_score

Los filtros estructurados actúan como restricciones (reducen el conjunto),
no como puntaje adicional.

También expone búsqueda en políticas de la tienda.
"""

import json
import re
from typing import Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import POLITICAS_PATH, MAX_PRODUCTOS_RESULTADO
from modules.embeddings import load_embeddings, embed_query


# Cache de políticas (cargada una vez)
_policies = None


def _load_policies() -> list:
    """Carga las políticas de la tienda."""
    global _policies
    if _policies is None:
        with open(POLITICAS_PATH, "r", encoding="utf-8") as f:
            _policies = json.load(f)
    return _policies


def _extract_filters(query: str) -> dict:
    """
    Extrae filtros estructurados desde la consulta del usuario.
    Busca patrones de color, talla y precio máximo.

    Args:
        query: Texto del usuario.

    Returns:
        dict con claves: color, talla, precio_max, marca.
    """
    query_lower = query.lower()
    filters = {}

    # Colores
    colores = ["negro", "blanco", "azul", "rojo", "gris", "verde", "café", "cafe"
               "amarillo", "rosado", "naranja", "morado", "transparente", "dorado"]
    for color in colores:
        if color in query_lower:
            filters["color"] = color
            break

    # Tallas numéricas (ej: "talla 42", "número 42", "en 42")
    talla_match = re.search(r"talla\s*(\d+)|n[uú]mero\s*(\d+)|en\s+(\d{2})\b", query_lower)
    if talla_match:
        talla = talla_match.group(1) or talla_match.group(2) or talla_match.group(3)
        filters["talla"] = int(talla)

    # Precio máximo (ej: "menos de 50000", "no más de $80.000", "bajo 70")
    precio_match = re.search(
        r"menos de\s*\$?([\d.]+)|no m[aá]s de\s*\$?([\d.]+)|bajo\s*\$?([\d.]+)",
        query_lower
    )
    if precio_match:
        raw = precio_match.group(1) or precio_match.group(2) or precio_match.group(3)
        filters["precio_max"] = int(raw.replace(".", ""))

    # Marcas conocidas
    marcas = ["nike", "adidas", "puma", "reebok", "asics", "skechers",
              "new balance", "converse", "vans", "merrell", "columbia",
              "under armour", "fila", "north face", "columbia"]
    for marca in marcas:
        if marca in query_lower:
            filters["marca"] = marca
            break

    return filters


def _keyword_score(query: str, product: dict) -> float:
    """
    Calcula un score de coincidencia por palabras clave.
    Busca en nombre, tags, categoría y descripción.

    Returns:
        float entre 0 y 1.
    """
    words = set(re.findall(r"\w+", query.lower()))
    # Texto del producto donde buscar
    product_text = " ".join([
        product.get("nombre", ""),
        product.get("categoria", ""),
        product.get("descripcion", ""),
        " ".join(product.get("tags", [])),
    ]).lower()
    product_words = set(re.findall(r"\w+", product_text))

    # Stopwords básicas a ignorar
    stopwords = {"el", "la", "los", "las", "un", "una", "de", "que", "en",
                 "y", "a", "con", "para", "por", "como", "me", "se", "si"}
    words -= stopwords

    if not words:
        return 0.0

    matches = words & product_words
    return len(matches) / len(words)


def _apply_filters(products: list, filters: dict) -> list:
    """
    Aplica filtros estructurados y retorna solo los productos que los cumplen.
    Si no hay filtros, retorna todos.
    """
    if not filters:
        return products

    result = []
    for p in products:
        # Filtro color
        if "color" in filters:
            colores_prod = [c.lower() for c in p.get("colores", [])]
            if filters["color"] not in colores_prod:
                continue

        # Filtro talla (numérica o "unica")
        if "talla" in filters:
            tallas_prod = p.get("tallas", [])
            if filters["talla"] not in tallas_prod:
                continue

        # Filtro precio máximo
        if "precio_max" in filters:
            if p.get("precio", 0) > filters["precio_max"]:
                continue

        # Filtro marca
        if "marca" in filters:
            marca_prod = p.get("marca", "").lower()
            if filters["marca"] not in marca_prod:
                continue

        result.append(p)
    return result


def search_products(query: str, top_k: int = MAX_PRODUCTOS_RESULTADO) -> list:
    """
    Búsqueda híbrida de productos.

    Pasos:
    1. Extrae filtros estructurados de la consulta.
    2. Calcula score semántico con embeddings.
    3. Calcula score por palabras clave.
    4. Combina scores (60% semántico, 40% keyword).
    5. Aplica filtros como restricciones.
    6. Retorna top_k productos ordenados por score.

    Args:
        query: Consulta del usuario en lenguaje natural.
        top_k: Número máximo de resultados a retornar.

    Returns:
        Lista de dicts de productos ordenados por relevancia.
    """
    products, embeddings_matrix = load_embeddings()

    # Score semántico
    query_vec = embed_query(query).reshape(1, -1)
    semantic_scores = cosine_similarity(query_vec, embeddings_matrix)[0]

    # Score por palabras clave
    keyword_scores = np.array([_keyword_score(query, p) for p in products])

    # Score combinado
    combined_scores = 0.6 * semantic_scores + 0.4 * keyword_scores

    # Ordenar por score descendente
    ranked_indices = np.argsort(combined_scores)[::-1]
    ranked_products = [products[i] for i in ranked_indices]
    ranked_scores = [float(combined_scores[i]) for i in ranked_indices]

    # Extraer filtros y aplicarlos
    filters = _extract_filters(query)
    if filters:
        filtered = _apply_filters(ranked_products, filters)
        # Si no hay resultados con filtros, retornar de todos modos los mejores sin filtro
        if filtered:
            return filtered[:top_k]

    # Sin filtros o sin resultados filtrados: retornar top_k por score
    # Umbral mínimo de relevancia para evitar ruido
    threshold = 0.1
    relevant = [(p, s) for p, s in zip(ranked_products, ranked_scores) if s >= threshold]
    return [p for p, _ in relevant[:top_k]]


def search_policies(query: str) -> Optional[str]:
    """
    Busca en las políticas de la tienda el contenido más relevante para la consulta.

    Usa matching simple por palabras clave sobre los campos 'tema' y 'contenido'.

    Args:
        query: Consulta del usuario.

    Returns:
        String con el contenido de la política más relevante, o None si no hay match.
    """
    policies = _load_policies()
    query_lower = query.lower()

    best_match = None
    best_score = 0

    for policy in policies:
        policy_text = f"{policy['tema']} {policy['contenido']}".lower()
        words = set(re.findall(r"\w+", query_lower))
        policy_words = set(re.findall(r"\w+", policy_text))
        stopwords = {"el", "la", "los", "las", "un", "una", "de", "que",
                     "en", "y", "a", "con", "para", "por", "como"}
        words -= stopwords
        if not words:
            continue
        score = len(words & policy_words) / len(words)
        if score > best_score:
            best_score = score
            best_match = policy["contenido"]

    # Solo retornar si hay suficiente relevancia
    return best_match if best_score > 0.15 else None
