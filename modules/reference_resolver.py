"""
modules/reference_resolver.py
==============================
Resuelve referencias a productos que YA se le mostraron al usuario
("la segunda", "esa negra", "la más barata") sin volver a buscar en
el catálogo completo.

Por qué existe este módulo
---------------------------
Antes, cuando el usuario escribía "la segunda", el código no revisaba
la posición 2 de lo que le acababa de mostrar. En vez de eso, mandaba
el texto "la segunda" (pegado con los nombres de los productos
anteriores) a search_products(), que vuelve a hacer una búsqueda
semántica sobre TODO productos.json. El resultado era una lista nueva,
sin relación real con lo que el usuario vio, y que además se
renumeraba desde 1 otra vez. Por eso "la segunda" terminaba apuntando
a cualquier cosa.

Este módulo se ejecuta ANTES de tocar el buscador: si detecta que el
usuario está señalando algo por posición o por atributo, devuelve
directamente ese producto (tomado de last_products, que sí es la
lista real que se le mostró). Si no detecta ninguna referencia clara,
devuelve None y el flujo normal de búsqueda sigue como antes.
"""

import re

# Ordinales explícitos -> índice dentro de la lista mostrada (0 = primero).
# Solo se incluyen formas ordinales ("segunda/segundo"), NO números
# cardinales sueltos ("dos", "tres"), porque esos aparecen todo el
# tiempo en frases normales ("quiero dos pares") y generarían falsos
# positivos.
_ORDINAL_WORDS = {
    "primera": 0, "primero": 0, "primer": 0,
    "segunda": 1, "segundo": 1,
    "tercera": 2, "tercero": 2,
    "cuarta": 3, "cuarto": 3,
    "quinta": 4, "quinto": 4,
    "sexta": 5, "sexto": 5,
    "ultima": -1, "última": -1, "ultimo": -1, "último": -1,
}

# Solo dispara con "opción 2" / "producto 3", nunca con "el 42" o
# "número 42" a secas, porque en tiendas de calzado "número" casi
# siempre significa talla (ej: "uso el número 42"), no posición.
_NUMBER_PATTERN = re.compile(r"\b(?:opci[oó]n|producto)\s*#?\s*(\d+)\b")

_COLORES = [
    "negro", "blanco", "azul", "rojo", "gris", "verde",
    "café", "cafe", "amarillo", "rosado", "naranja",
    "morado", "transparente", "dorado",
]


def _extract_ordinal_index(text: str):
    """Busca un ordinal explícito (palabra o 'opción/producto N') en el texto."""
    text_l = text.lower()

    for word, idx in _ORDINAL_WORDS.items():
        if re.search(rf"\b{word}\b", text_l):
            return idx

    match = _NUMBER_PATTERN.search(text_l)
    if match:
        return int(match.group(1)) - 1  # "producto 2" -> índice 1

    return None


def _color_pattern(color: str) -> str:
    """
    Genera un patrón que reconoce ambos géneros ("negro"/"negra",
    "dorado"/"dorada"), ya que en la lista de colores de cada
    producto siempre se guarda la forma masculina, pero el usuario
    puede escribir "esa negra" en femenino.
    """
    if color.endswith("o"):
        return rf"\b{color[:-1]}[oa]s?\b"
    return rf"\b{color}s?\b"


def _extract_simple_filters(text: str) -> dict:
    """
    Extracción mínima de atributos, usada SOLO para filtrar entre los
    productos que ya están en last_products (no para buscar en el
    catálogo completo).
    """
    text_l = text.lower()
    filters = {}

    for color in _COLORES:
        if re.search(_color_pattern(color), text_l):
            filters["color"] = "cafe" if color == "café" else color
            break

    if re.search(r"m[aá]s barat|m[aá]s econ[oó]mic|menor precio", text_l):
        filters["orden_precio"] = "asc"
    elif re.search(r"m[aá]s car|premium", text_l):
        filters["orden_precio"] = "desc"

    return filters


def resolve_reference(user_text: str, last_products: list):
    """
    Intenta averiguar a cuál de los productos en `last_products` se
    refiere el usuario.

    Args:
        user_text: mensaje del usuario, tal cual (sin modificar).
        last_products: lista de productos mostrados en el turno anterior.

    Returns:
        list[dict] con el/los producto(s) referenciado(s), o None si no
        se detectó ninguna referencia clara (en ese caso el llamador
        debe hacer una búsqueda normal en el catálogo).
    """
    if not last_products:
        return None

    # 1. Referencia por posición: "la segunda", "el primero", "opción 3"
    idx = _extract_ordinal_index(user_text)
    if idx is not None:
        try:
            return [last_products[idx]]
        except IndexError:
            # El usuario pidió una posición que no existe
            # (ej. "la quinta" habiendo solo 3 productos).
            return None

    # 2. Referencia por atributo: "esa negra", "la más barata"
    filters = _extract_simple_filters(user_text)
    if filters:
        candidates = last_products

        if "color" in filters:
            candidates = [
                p for p in candidates
                if filters["color"] in [c.lower() for c in p.get("colores", [])]
            ]

        if "orden_precio" in filters and candidates:
            candidates = sorted(
                candidates,
                key=lambda p: p.get("precio", 0),
                reverse=(filters["orden_precio"] == "desc"),
            )
            candidates = candidates[:1]

        return candidates or None

    return None