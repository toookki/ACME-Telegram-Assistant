"""
modules/reference_resolver.py
=============================
Resuelve referencias directas a productos ya mostrados.

No intenta resolver refinamientos nuevos como:
- "¿Y tienes alguna Nike parecida?"
- "¿Hay algo más barato aunque abrigue menos?"

Esos casos deben volver al buscador usando el query_state anterior.
"""

import re
import unicodedata


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
    "septima": 6, "septimo": 6,
    "octava": 7, "octavo": 7,
    "ultima": -1, "ultimo": -1,
}

# Solo dispara con "opción 2" / "producto 3", nunca con "el 42" o
# "número 42" a secas, porque en tiendas de calzado "número" casi
# siempre significa talla (ej: "uso el número 42"), no posición.
_NUMBER_PATTERN = re.compile(
    r"\b(?:opci[oó]n|producto)\s*#?\s*(\d+)\b"
)

_COLORES = [
    "negro", "blanco", "azul", "rojo", "gris", "verde",
    "cafe", "amarillo", "rosado", "naranja", "morado",
    "transparente", "dorado", "beige",
]


def _normalize_text(text: str) -> str:
    """
    Normaliza un texto para facilitar comparaciones y búsquedas.

    Convierte el texto a minúsculas, elimina tildes y caracteres
    especiales, y reduce espacios consecutivos.

    Args:
        text: Texto que se desea normalizar.

    Returns:
        Texto normalizado.
    """
    text = str(text).lower()
    text = text.replace("ı", "i").replace("´", "").replace("`", "")
    text = unicodedata.normalize("NFD", text)
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch) != "Mn"
    )
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def _extract_ordinal_index(text: str):
    """
    Extrae el índice de una referencia ordinal explícita.

    Reconoce palabras como "segunda" o "último", y expresiones
    como "opción 2" o "producto 3".

    Args:
        text: Texto del usuario.

    Returns:
        Índice correspondiente dentro de la lista de productos,
        o None si no se detecta una referencia ordinal.
    """
    text_l = _normalize_text(text)

    for word, idx in _ORDINAL_WORDS.items():
        if re.search(rf"\b{word}\b", text_l):
            return idx

    match = _NUMBER_PATTERN.search(text_l)
    if match:
        return int(match.group(1)) - 1  # "producto 2" -> índice 1

    return None


def _color_pattern(color: str) -> str:
    """
    Genera un patrón regex para reconocer variantes de género y número.

    Permite detectar expresiones como "negro", "negra", "negros"
    o "negras", aunque los colores de los productos se almacenen
    normalmente en forma masculina.

    Args:
        color: Nombre del color base.

    Returns:
        Patrón regex asociado al color.
    """
    color = _normalize_text(color)

    if color == "cafe":
        return r"\bcaf(e|es)\b"

    if color.endswith("o"):
        return rf"\b{color[:-1]}[oa]s?\b"

    return rf"\b{color}s?\b"


def _extract_color(text: str):
    """
    Extrae un color reconocido desde el texto del usuario.

    Args:
        text: Texto del usuario.

    Returns:
        Nombre normalizado del color detectado,
        o None si no se encuentra ninguno.
    """
    text_l = _normalize_text(text)

    for color in _COLORES:
        if re.search(_color_pattern(color), text_l):
            return color

    return None


def _extract_size(text: str):
    """
    Extrae una talla explícita desde el texto del usuario.

    Reconoce expresiones como "talla 42", "número 42",
    "talla M" o "en 42".

    Args:
        text: Texto del usuario.

    Returns:
        Talla detectada como int si es numérica, como str si es
        alfabética, o None si no se detecta ninguna.
    """
    text_l = _normalize_text(text)

    match = re.search(
        r"talla\s*([a-z]{1,3}|\d{1,2})|"
        r"numero\s*([a-z]{1,3}|\d{1,2})|"
        r"en\s+(\d{2})\b",
        text_l,
    )

    if not match:
        return None

    raw = next(g for g in match.groups() if g)
    raw = raw.upper()

    return int(raw) if raw.isdigit() else raw


def _is_refinement_query(text: str) -> bool:
    """
    Determina si el texto corresponde a un refinamiento de búsqueda.

    Detecta expresiones que solicitan nuevos productos similares
    o alternativas, por lo que la consulta debe volver al buscador
    en lugar de resolverse como una referencia directa.

    Args:
        text: Texto del usuario.

    Returns:
        True si se detecta un refinamiento de búsqueda;
        False en caso contrario.
    """
    t = _normalize_text(text)

    patterns = [
        r"\bparecid\w*\b",
        r"\bsimilar\w*\b",
        r"\botra opcion\b",
        r"\botras opciones\b",
        r"\balguna?\b",
        r"\by tienes\b",
        r"\btienes alguna\b",
        r"\bque abrigue menos\b",
        r"\babrigue menos\b",
        r"\bmenos abrigad\w*\b",
        r"\baunque abrig\w* menos\b",
    ]

    return any(re.search(pattern, t) for pattern in patterns)


def _is_attribute_question(text: str) -> bool:
    """
    Determina si el usuario pregunta por atributos de un producto.

    Detecta consultas relacionadas con talla, color u otras
    expresiones que contienen esos atributos.

    Args:
        text: Texto del usuario.

    Returns:
        True si se detecta una pregunta por atributos;
        False en caso contrario.
    """
    t = _normalize_text(text)

    return bool(
        re.search(r"\btiene\b.*\btalla\b", t)
        or re.search(
            r"\besta\b.*\b("
            r"color|negro|blanco|azul|rojo|gris|verde|beige|cafe"
            r")\b",
            t,
        )
        or re.search(r"\bhay\b.*\b(en|color|talla)\b", t)
        or _extract_color(t)
        or _extract_size(t)
    )


def _product_has_color(product: dict, color: str) -> bool:
    """
    Verifica si un producto está disponible en un color determinado.

    Args:
        product: Diccionario con la información del producto.
        color: Color normalizado que se desea comprobar.

    Returns:
        True si el producto incluye el color;
        False en caso contrario.
    """
    colores = [
        _normalize_text(c)
        for c in product.get("colores", [])
    ]

    return color in colores


def _product_has_size(product: dict, size) -> bool:
    """
    Verifica si un producto está disponible en una talla determinada.

    Compara primero los valores originales y luego sus versiones
    normalizadas para admitir representaciones equivalentes.

    Args:
        product: Diccionario con la información del producto.
        size: Talla que se desea comprobar.

    Returns:
        True si el producto incluye la talla;
        False en caso contrario.
    """
    tallas = product.get("tallas", [])

    if size in tallas:
        return True

    size_norm = _normalize_text(str(size))
    tallas_norm = [
        _normalize_text(str(t))
        for t in tallas
    ]

    return size_norm in tallas_norm


def resolve_reference(user_text: str, last_products: list):
    """
    Intenta resolver una referencia a productos mostrados previamente.

    Puede identificar referencias por posición, precio, color, talla
    o atributos de un único producto mostrado. Las consultas que
    representan refinamientos nuevos se dejan sin resolver para que
    vuelvan al buscador.

    Args:
        user_text: Mensaje del usuario tal como fue recibido.
        last_products: Lista de productos mostrados previamente.

    Returns:
        Lista con el producto o los productos referenciados;
        lista vacía si la referencia ordinal está fuera de rango;
        o None si no se detecta una referencia clara.
    """
    if not last_products:
        return None

    if _is_refinement_query(user_text):
        return None

    idx = _extract_ordinal_index(user_text)
    if idx is not None:
        try:
            return [last_products[idx]]
        except IndexError:
            return []

    if len(last_products) == 1 and _is_attribute_question(user_text):
        return [last_products[0]]

    t = _normalize_text(user_text)

    if re.search(r"\b(mas barat\w*|mas economic\w*|menor precio)\b", t):
        return [
            min(
                last_products,
                key=lambda p: p.get("precio", 0),
            )
        ]

    if re.search(r"\b(mas car\w*|premium|gama alta)\b", t):
        return [
            max(
                last_products,
                key=lambda p: p.get("precio", 0),
            )
        ]

    color = _extract_color(user_text)
    size = _extract_size(user_text)

    candidates = last_products

    if color:
        candidates = [
            p for p in candidates
            if _product_has_color(p, color)
        ]

    if size is not None:
        candidates = [
            p for p in candidates
            if _product_has_size(p, size)
        ]

    if color or size is not None:
        return candidates

    return None