"""
modules/search.py
=================
Motor de búsqueda híbrida de productos y políticas.

Diseño:
- Usa un ranking híbrido: embeddings + keywords como base, reforzado con filtros, intención, precio y stock.
- Agrega normalización robusta para español y texto copiado desde PDF.
- Agrega parser de intención: categoría, uso, marca, color, talla, precio.
- Separa filtros duros de preferencias suaves.
- Usa una taxonomía general del catálogo, no reglas por producto específico.
"""

import json
import re
import unicodedata
from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import POLITICAS_PATH, MAX_PRODUCTOS_RESULTADO
from modules.embeddings import load_embeddings, embed_query


_policies = None
_policy_embeddings = None


def _normalize_text(text: str) -> str:
    """
    Normaliza texto para comparaciones y búsquedas robustas.

    Convierte a minúsculas, corrige caracteres problemáticos frecuentes,
    elimina tildes y normaliza espacios consecutivos.

    Args:
        text: Texto que se desea normalizar.

    Returns:
        Texto normalizado.
    """
    text = str(text).lower()

    replacements = {
        "ı": "i",
        "´": "",
        "`": "",
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "\u00a0": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokens(text: str) -> set[str]:
    """
    Extrae tokens alfanuméricos únicos desde un texto normalizado.

    Args:
        text: Texto desde el cual se extraerán los tokens.

    Returns:
        Conjunto de tokens alfanuméricos únicos.
    """
    return set(re.findall(r"[a-z0-9]+", _normalize_text(text)))


def _product_full_text(product: dict) -> str:
    """
    Construye una representación textual completa de un producto.

    Combina nombre, categoría, descripción, marca, tags y colores para
    búsquedas léxicas y cálculo de relevancia.

    Args:
        product: Diccionario con la información del producto.

    Returns:
        Texto normalizado con los principales campos del producto.
    """
    return _normalize_text(" ".join([
        product.get("nombre", ""),
        product.get("categoria", ""),
        product.get("descripcion", ""),
        product.get("marca", ""),
        " ".join(str(t) for t in product.get("tags", [])),
        " ".join(str(c) for c in product.get("colores", [])),
    ]))


def _product_type_text(product: dict) -> str:
    """
    Construye el texto utilizado para clasificar el tipo de producto.

    Combina nombre, categoría, marca y tags, evitando depender directamente
    de la descripción para la detección principal del tipo.

    Args:
        product: Diccionario con la información del producto.

    Returns:
        Texto normalizado utilizado para clasificar el producto.
    """
    return _normalize_text(" ".join([
        product.get("nombre", ""),
        product.get("categoria", ""),
        product.get("marca", ""),
        " ".join(str(t) for t in product.get("tags", [])),
    ]))


def _contains_any(text: str, terms: list[str]) -> bool:
    """
    Comprueba si un texto contiene alguno de los términos indicados.

    La comparación se realiza sobre texto normalizado y exige coincidencias
    por límites de palabra.

    Args:
        text: Texto en el que se buscarán coincidencias.
        terms: Lista de términos que se desean detectar.

    Returns:
        True si se encuentra al menos un término; False en caso contrario.
    """
    text = _normalize_text(text)
    return any(re.search(rf"\b{re.escape(term)}\b", text) for term in terms)


def _parse_price(raw: str) -> int:
    """
    Convierte una representación textual de precio a entero.

    Elimina separadores de miles y, para valores menores que 1000,
    interpreta el número como miles de unidades monetarias.

    Args:
        raw: Representación textual del precio.

    Returns:
        Precio convertido a entero.
    """
    value = int(str(raw).replace(".", ""))
    if value < 1000:
        value *= 1000
    return value


def _load_policies() -> list:
    """
    Carga las políticas de la tienda desde disco usando caché en memoria.

    La lectura del archivo se realiza solo la primera vez que se invoca
    la función.

    Returns:
        Lista de políticas cargadas desde el archivo configurado.
    """
    global _policies
    if _policies is None:
        with open(POLITICAS_PATH, "r", encoding="utf-8") as f:
            _policies = json.load(f)
    return _policies


def _product_family(product: dict) -> str:
    """
    Clasifica un producto dentro de una familia general del catálogo.

    Primero utiliza la categoría normalizada y luego aplica reglas de
    fallback sobre el texto de tipo del producto.

    Args:
        product: Diccionario con la información del producto.

    Returns:
        Nombre normalizado de la familia detectada.
    """
    text = _product_type_text(product)
    categoria = _normalize_text(product.get("categoria", ""))

    if any(term in categoria for term in ["zapatillas", "zapatos", "sandalias", "botines"]):
        return "calzado"
    if "ropa invierno" in categoria:
        return "ropa_invierno"
    if "accesorios invierno" in categoria:
        return "accesorio_invierno"
    if "ropa deportiva" in categoria:
        return "ropa_deportiva"
    if "ropa casual" in categoria:
        return "ropa_casual"
    if "mochilas" in categoria or "bolsos" in categoria:
        return "bolsos"
    if "tecnologia" in categoria:
        return "tecnologia"
    if "hogar" in categoria:
        return "hogar"
    if "belleza" in categoria or "cuidado personal" in categoria:
        return "belleza"
    if "accesorios deportivos" in categoria:
        return "accesorio_deportivo"

    if _contains_any(text, ["zapatilla", "zapato", "botin", "sandalia", "calzado"]):
        return "calzado"
    if _contains_any(text, ["mochila", "bolso", "banano", "cartera", "tote"]):
        return "bolsos"
    if _contains_any(text, ["audifonos", "notebook", "tablet", "smartwatch", "monitor"]):
        return "tecnologia"

    return "otro"


def _product_kind(product: dict) -> str:
    """
    Detecta el tipo concreto de producto.

    Primero revisa nombre, categoría y marca. Luego utiliza tags y otros
    campos como fallback. Los tipos concretos se evalúan antes que conceptos
    genéricos para evitar clasificaciones ambiguas.

    Ejemplos de errores evitados:
    - Polar Doite Thermic -> "abrigo" por tag "abrigo".
    - Bufanda Basement Knit -> "abrigo" por tag "abrigo".

    Args:
        product: Diccionario con la información del producto.

    Returns:
        Nombre normalizado del tipo concreto detectado, o "otro" si no
        existe una coincidencia.
    """
    primary_text = _normalize_text(" ".join([
        product.get("nombre", ""),
        product.get("categoria", ""),
        product.get("marca", ""),
    ]))

    full_text = _product_type_text(product)

    # Patrones concretos. El orden importa:
    # primero producto concreto, después conceptos genéricos.
    concrete_patterns = [
        # Calzado
        ("zapatilla", r"\bzapatilla\w*\b"),
        ("zapato", r"\bzapato\w*\b"),
        ("botin", r"\bbotin\w*\b"),
        ("sandalia", r"\bsandalia\w*\b"),

        # Invierno / abrigo: tipos concretos antes que "abrigo"
        ("parka", r"\bparka\w*\b"),
        ("chaqueta", r"\bchaqueta\w*\b"),
        ("impermeable", r"\bimpermeable\w*\b"),
        ("trench", r"\btrench\b"),
        ("primera_capa", r"\bprimera capa\b"),
        ("polar", r"\bpolar\w*\b"),
        ("poleron", r"\bpoleron\w*|hoodie\w*\b"),
        ("sweater", r"\bsweater\w*\b"),
        ("gorro", r"\bgorro\w*\b"),
        ("guantes", r"\bguante\w*\b"),
        ("bufanda", r"\bbufanda\w*\b"),
        ("calcetines", r"\bcalcetin\w*|calcetines\w*\b"),
        ("abrigo", r"\babrigo\w*\b"),

        # Ropa
        ("polera", r"\bpolera\w*\b"),
        ("pantalon", r"\bpantalon\w*|jean\w*|jogger\w*|chino\w*\b"),
        ("short", r"\bshort\w*|bermuda\w*\b"),
        ("calza", r"\bcalza\w*\b"),
        ("top", r"\btop\b"),
        ("camisa", r"\bcamisa\w*\b"),
        ("vestido", r"\bvestido\w*\b"),
        ("falda", r"\bfalda\w*\b"),
        ("blazer", r"\bblazer\w*\b"),

        # Bolsos
        ("mochila", r"\bmochila\w*\b"),
        ("bolso", r"\bbolso\w*|duffel\w*|tote\w*\b"),
        ("banano", r"\bbanano\w*\b"),
        ("cartera", r"\bcartera\w*\b"),

        # Tecnología
        ("audifonos", r"\baudifono\w*|airpods\w*\b"),
        ("notebook", r"\bnotebook\w*|laptop\w*\b"),
        ("tablet", r"\btablet\w*\b"),
        ("kindle", r"\bkindle\w*\b"),
        ("smartwatch", r"\bsmartwatch\w*|reloj inteligente\w*\b"),
        ("mouse", r"\bmouse\w*\b"),
        ("teclado", r"\bteclado\w*\b"),
        ("parlante", r"\bparlante\w*\b"),
        ("monitor", r"\bmonitor\w*\b"),
        ("impresora", r"\bimpresora\w*\b"),
        ("router", r"\brouter\w*\b"),
        ("control_gamer", r"\bcontrol gamer\w*|xbox\w*\b"),
        ("power_bank", r"\bpower bank\w*|bateria externa\w*\b"),
        ("cargador", r"\bcargador\w*\b"),
        ("webcam", r"\bwebcam\w*|camara web\w*\b"),
        ("ssd", r"\bssd\w*|disco\w*\b"),

        # Hogar
        ("freidora", r"\bfreidora\w*\b"),
        ("cafetera", r"\bcafetera\w*\b"),
        ("hervidor", r"\bhervidor\w*\b"),
        ("aspiradora", r"\baspiradora\w*|robot aspirador\w*\b"),
        ("sartenes", r"\bsarten\w*|sartenes\w*\b"),
        ("licuadora", r"\blicuadora\w*\b"),
        ("plancha_hogar", r"\bplancha vapor\w*|\bplancha philips\w*"),
        ("organizador", r"\borganizador\w*\b"),
        ("sabanas", r"\bsabana\w*|sabanas\w*\b"),
        ("almohada", r"\balmohada\w*\b"),
        ("toalla", r"\btoalla\w*\b"),
        ("lampara", r"\blampara\w*\b"),
        ("silla", r"\bsilla\w*\b"),
        ("mesa", r"\bmesa\w*\b"),

        # Belleza
        ("crema", r"\bcrema\w*\b"),
        ("protector_solar", r"\bprotector solar\w*|spf\w*\b"),
        ("shampoo", r"\bshampoo\w*|champu\w*\b"),
        ("acondicionador", r"\bacondicionador\w*\b"),
        ("secador_pelo", r"\bsecador\w*.*pelo|secador de pelo\w*\b"),
        ("plancha_pelo", r"\bplancha de pelo\w*|alisador\w*\b"),
        ("perfume", r"\bperfume\w*\b"),
        ("maquillaje", r"\bmaquillaje\w*\b"),
        ("rasuradora", r"\brasuradora\w*|gillette\w*\b"),
        ("cepillo_dental", r"\bcepillo\w*.*dental|oral-b\w*\b"),

        # Accesorios deportivos
        ("mat_yoga", r"\bmat\w*|yoga\w*\b"),
        ("botella", r"\bbotella\w*\b"),
        ("bandas", r"\bbanda\w* elastica\w*|bandas\w*\b"),
        ("pesas", r"\bpesa\w*|mancuerna\w*\b"),
    ]

    # 1) Buscar primero en nombre/categoría/marca.
    for kind, pattern in concrete_patterns:
        if re.search(pattern, primary_text):
            return kind

    # 2) Fallback en tags/descripción, manteniendo el mismo orden.
    for kind, pattern in concrete_patterns:
        if re.search(pattern, full_text):
            return kind

    return "otro"


def _product_role(product: dict) -> str:
    """
    Asigna un rol funcional al producto dentro de su familia.

    Distingue, entre otros, prendas principales para frío, capas livianas,
    accesorios de invierno y productos principales de otras familias.

    Args:
        product: Diccionario con la información del producto.

    Returns:
        Rol funcional normalizado del producto.
    """
    kind = _product_kind(product)
    family = _product_family(product)

    if kind in {"parka", "chaqueta", "abrigo", "impermeable", "trench"}:
        return "frio_principal"
    if kind in {"poleron", "polar", "sweater", "primera_capa"}:
        return "frio_liviano"
    if kind in {"gorro", "guantes", "bufanda", "calcetines"} and family == "accesorio_invierno":
        return "frio_accesorio"
    if family == "calzado":
        return "principal"
    if family in {"bolsos", "tecnologia", "hogar", "belleza"}:
        return "principal"
    if family in {"accesorio_deportivo", "accesorio_invierno"}:
        return "accesorio"

    return "principal"


_COLORS = [
    "negro", "blanco", "azul", "rojo", "gris", "verde", "cafe",
    "beige", "amarillo", "rosado", "naranja", "morado",
    "transparente", "dorado"
]


def _color_regex(color: str) -> str:
    """
    Genera un patrón regex para reconocer variantes de un color.

    Permite detectar variaciones de género y número, por ejemplo
    "negro", "negra", "negros" y "negras".

    Args:
        color: Nombre base del color.

    Returns:
        Patrón regex asociado al color.
    """
    color = _normalize_text(color)

    if color == "cafe":
        return r"\bcaf(e|es)\b"

    if color.endswith("o"):
        stem = color[:-1]
        return rf"\b{stem}[oa]s?\b"

    return rf"\b{color}s?\b"


def _extract_color(query_norm: str) -> Optional[str]:
    """
    Extrae un color reconocido desde una consulta normalizada.

    Args:
        query_norm: Consulta del usuario previamente normalizada.

    Returns:
        Nombre normalizado del color detectado, o None si no se encuentra
        ninguno.
    """
    for color in _COLORS:
        if re.search(_color_regex(color), query_norm):
            return color
    return None


def _extract_size(query_norm: str):
    """
    Extrae una talla explícita desde una consulta normalizada.

    Reconoce expresiones como "talla 42", "número 42", "nro M" o "en 42".

    Args:
        query_norm: Consulta del usuario previamente normalizada.

    Returns:
        Talla detectada como int si es numérica, como str si es alfabética,
        o None si no se detecta ninguna.
    """
    patterns = [
        r"\btalla\s*([a-z]{1,3}|\d{1,2})\b",
        r"\bnumero\s*([a-z]{1,3}|\d{1,2})\b",
        r"\bnro\s*([a-z]{1,3}|\d{1,2})\b",
        r"\ben\s+(\d{2})\b",
    ]

    for pattern in patterns:
        m = re.search(pattern, query_norm)
        if m:
            raw = m.group(1).upper()
            if raw.isdigit():
                return int(raw)
            return raw

    return None


def _extract_price_max(query_norm: str) -> Optional[int]:
    """
    Extrae un precio máximo explícito desde la consulta.

    Reconoce expresiones como "menos de", "no más de", "bajo", "hasta"
    o "máximo".

    Args:
        query_norm: Consulta del usuario previamente normalizada.

    Returns:
        Precio máximo como entero, o None si no se detecta uno.
    """
    m = re.search(
        r"menos de\s*\$?([\d.]+)|no mas de\s*\$?([\d.]+)|bajo\s*\$?([\d.]+)|hasta\s*\$?([\d.]+)|maximo\s*\$?([\d.]+)",
        query_norm
    )
    if not m:
        return None

    raw = next(g for g in m.groups() if g)
    return _parse_price(raw)


def _extract_price_preference(query_norm: str) -> Optional[str]:
    """
    Detecta una preferencia relativa de precio.

    Categorías:
    - cheapest: el usuario pide explícitamente la opción más barata.
    - economic: el usuario busca algo conveniente o no tan caro.
    - premium: el usuario busca una alternativa de gama alta.

    Args:
        query_norm: Consulta del usuario previamente normalizada.

    Returns:
        Preferencia detectada como "cheapest", "economic" o "premium";
        None si no existe una señal clara.
    """
    if re.search(
        r"\b(mas barato|mas economico|menor precio|lo mas barato|la mas barata|el mas barato|algo mas barato)\b",
        query_norm
    ):
        return "cheapest"

    if re.search(
        r"\b(barat\w*|economic\w*|no tan car\w*|no sea tan car\w*|que no sea tan car\w*|conveniente|precio bajo|accesible|sin gastar tanto)\b",
        query_norm
    ):
        return "economic"

    if re.search(
        r"\b(premium|gama alta|mejor calidad|mas car\w*|alta gama)\b",
        query_norm
    ):
        return "premium"

    return None


def _extract_brand(query_norm: str, products: list) -> Optional[str]:
    """
    Extrae una marca mencionada explícitamente en la consulta.

    Las marcas posibles se obtienen dinámicamente desde el catálogo y se
    ordenan por longitud para priorizar coincidencias más específicas.

    Args:
        query_norm: Consulta del usuario previamente normalizada.
        products: Lista completa de productos del catálogo.

    Returns:
        Marca normalizada detectada, o None si no existe coincidencia.
    """
    brands = sorted(
        {_normalize_text(p.get("marca", "")) for p in products if p.get("marca")},
        key=len,
        reverse=True
    )
    for brand in brands:
        if brand and re.search(rf"\b{re.escape(brand)}\b", query_norm):
            return brand
    return None


def _detect_explicit_kind_and_family(query_norm: str) -> tuple[Optional[str], Optional[str]]:
    """
    Detecta un tipo y una familia de producto mencionados explícitamente.

    Utiliza una taxonomía general de patrones para categorías como calzado,
    ropa, tecnología, hogar y belleza.

    Args:
        query_norm: Consulta del usuario previamente normalizada.

    Returns:
        Tupla con el tipo concreto y la familia detectados. Cada elemento
        puede ser None si no existe una coincidencia explícita.
    """
    checks = [
        ("zapatilla", "calzado", r"\bzapatill\w*|sneaker\w*\b"),
        ("zapato", "calzado", r"\bzapato\w*\b"),
        ("botin", "calzado", r"\bbotin\w*\b"),
        ("sandalia", "calzado", r"\bsandalia\w*|chala\w*\b"),
        (None, "calzado", r"\bcalzado\b"),

        ("parka", "ropa_invierno", r"\bparka\w*\b"),
        ("chaqueta", "ropa_invierno", r"\bchaqueta\w*\b"),
        ("abrigo", "ropa_invierno", r"\babrigo\w*\b"),
        ("impermeable", "ropa_invierno", r"\bimpermeable\w*\b"),
        ("polar", "ropa_invierno", r"\bpolar\w*\b"),
        ("poleron", "ropa_casual", r"\bpoleron\w*|hoodie\w*\b"),
        ("sweater", "ropa_casual", r"\bsweater\w*\b"),
        ("primera_capa", "ropa_invierno", r"\bprimera capa\b"),

        ("polera", "ropa_casual", r"\bpolera\w*|camiseta\w*|remera\w*\b"),
        ("pantalon", "ropa_casual", r"\bpantalon\w*|jean\w*|jogger\w*|chino\w*\b"),
        ("short", "ropa_deportiva", r"\bshort\w*|bermuda\w*\b"),
        ("calza", "ropa_deportiva", r"\bcalza\w*\b"),
        ("camisa", "ropa_casual", r"\bcamisa\w*\b"),
        ("vestido", "ropa_casual", r"\bvestido\w*\b"),
        ("falda", "ropa_casual", r"\bfalda\w*\b"),

        ("gorro", "accesorio_invierno", r"\bgorro\w*\b"),
        ("guantes", "accesorio_invierno", r"\bguante\w*\b"),
        ("bufanda", "accesorio_invierno", r"\bbufanda\w*\b"),
        ("calcetines", "accesorio_invierno", r"\bcalcetin\w*|calcetines\w*\b"),

        ("mochila", "bolsos", r"\bmochila\w*\b"),
        ("bolso", "bolsos", r"\bbolso\w*|duffel\w*|tote\w*\b"),
        ("banano", "bolsos", r"\bbanano\w*\b"),
        ("cartera", "bolsos", r"\bcartera\w*\b"),

        ("audifonos", "tecnologia", r"\baudifono\w*|airpods\w*\b"),
        ("notebook", "tecnologia", r"\bnotebook\w*|laptop\w*\b"),
        ("tablet", "tecnologia", r"\btablet\w*\b"),
        ("smartwatch", "tecnologia", r"\bsmartwatch\w*|reloj inteligente\w*\b"),
        ("mouse", "tecnologia", r"\bmouse\w*\b"),
        ("teclado", "tecnologia", r"\bteclado\w*\b"),
        ("parlante", "tecnologia", r"\bparlante\w*\b"),
        ("monitor", "tecnologia", r"\bmonitor\w*\b"),
        ("impresora", "tecnologia", r"\bimpresora\w*\b"),
        ("router", "tecnologia", r"\brouter\w*\b"),
        ("cargador", "tecnologia", r"\bcargador\w*\b"),
        ("power_bank", "tecnologia", r"\bpower bank\w*|bateria externa\w*\b"),

        ("freidora", "hogar", r"\bfreidora\w*\b"),
        ("cafetera", "hogar", r"\bcafetera\w*\b"),
        ("hervidor", "hogar", r"\bhervidor\w*\b"),
        ("aspiradora", "hogar", r"\baspiradora\w*|robot aspirador\w*\b"),
        ("sartenes", "hogar", r"\bsarten\w*|sartenes\w*\b"),
        ("licuadora", "hogar", r"\blicuadora\w*\b"),
        ("sabanas", "hogar", r"\bsabana\w*|sabanas\w*\b"),
        ("almohada", "hogar", r"\balmohada\w*\b"),
        ("toalla", "hogar", r"\btoalla\w*\b"),
        ("lampara", "hogar", r"\blampara\w*\b"),
        ("silla", "hogar", r"\bsilla\w*\b"),
        ("mesa", "hogar", r"\bmesa\w*\b"),

        ("crema", "belleza", r"\bcrema\w*\b"),
        ("protector_solar", "belleza", r"\bprotector solar\w*|spf\w*\b"),
        ("shampoo", "belleza", r"\bshampoo\w*|champu\w*\b"),
        ("acondicionador", "belleza", r"\bacondicionador\w*\b"),
        ("secador_pelo", "belleza", r"\bsecador\w*.*pelo|secador de pelo\w*\b"),
        ("plancha_pelo", "belleza", r"\bplancha de pelo\w*|alisador\w*\b"),
        ("perfume", "belleza", r"\bperfume\w*\b"),
        ("maquillaje", "belleza", r"\bmaquillaje\w*\b"),
        ("rasuradora", "belleza", r"\brasuradora\w*|afeitadora\w*\b"),
        ("cepillo_dental", "belleza", r"\bcepillo\w*.*dental|oral-b\w*\b"),
    ]

    for kind, family, pattern in checks:
        if re.search(pattern, query_norm):
            return kind, family

    return None, None


def _extract_use_cases(query_norm: str) -> list[str]:
    """
    Extrae casos de uso o contextos mencionados en la consulta.

    Detecta intenciones como caminar, running, trekking, frío, lluvia,
    oficina, estudio, hogar o cuidado del pelo.

    Args:
        query_norm: Consulta del usuario previamente normalizada.

    Returns:
        Lista de casos de uso detectados, sin inferir productos concretos.
    """
    use_cases = []
    checks = [
        ("caminar", r"\bcaminar\w*|caminata\w*|camino mucho|caminar mucho|largas distancias\b"),
        ("running", r"\bcorrer\w*|running|trote|trotar\w*\b"),
        ("training", r"\bgimnasio|training|entrenamiento|yoga|pilates|fuerza|funcional\b"),
        ("trekking", r"\btrekking|outdoor|cerro|senderismo|sendero|montana\b"),
        ("ciudad", r"\bciudad|urbano|uso diario|dia a dia|diario|casual\b"),
        ("frio", r"\bfrio|invierno|abrig\w*|bajas temperaturas|temperaturas bajas|clima fresco|termic\w*\b"),
        ("lluvia", r"\blluvia|impermeable|viento|cortaviento\b"),
        ("calor", r"\bverano|playa|piscina|calor|livian\w*\b"),
        ("oficina", r"\boficina|formal|trabajo|reunion|evento\b"),
        ("estudio", r"\bcolegio|universidad|estudio|notebook|clases\b"),
        ("hogar", r"\bcasa|hogar|cocina|dormitorio|limpieza\b"),
        ("pelo", r"\bpelo|cabello|hidratacion|reparacion capilar\b"),
    ]

    for name, pattern in checks:
        if re.search(pattern, query_norm):
            use_cases.append(name)

    return use_cases


def _infer_family_from_use_cases(state: dict) -> None:
    """
    Infiere una familia de producto a partir de los casos de uso.

    Solo actúa cuando el estado todavía no contiene una familia explícita.
    La función modifica el diccionario state en el lugar.

    Args:
        state: Estado estructurado de la consulta.

    Returns:
        None.
    """
    if state.get("product_family"):
        return

    use_cases = set(state.get("use_cases", []))

    if {"caminar", "running", "trekking"} & use_cases:
        state["product_family"] = "calzado"
        state["family_inferred"] = True
        return

    if "frio" in use_cases or "lluvia" in use_cases:
        state["product_family"] = "ropa_invierno"
        state["product_role"] = "frio_principal"
        state["family_inferred"] = True
        return

    if "pelo" in use_cases:
        state["product_family"] = "belleza"
        state["family_inferred"] = True
        return

    if "hogar" in use_cases:
        state["product_family"] = "hogar"
        state["family_inferred"] = True
        return


def _adjust_role_from_query(state: dict) -> None:
    """
    Ajusta el rol funcional del producto según la consulta y el estado.

    Permite distinguir, por ejemplo, entre abrigo principal, capa liviana
    y accesorio de invierno. La función modifica state en el lugar.

    Args:
        state: Estado estructurado de la consulta.

    Returns:
        None.
    """
    q = state["query_norm"]
    use_cases = set(state.get("use_cases", []))

    if re.search(r"\babrigue menos|abriga menos|menos abrigad\w*|aunque abrig\w* menos|clima fresco\b", q):
        state["product_role"] = "frio_liviano"
        state["product_family"] = "ropa_invierno"
        return

    if state.get("product_family") == "accesorio_invierno":
        state["product_role"] = "frio_accesorio"
        return

    if state.get("product_kind") in {"poleron", "polar", "sweater", "primera_capa"}:
        state["product_role"] = "frio_liviano"
        return

    if state.get("product_kind") in {"parka", "chaqueta", "abrigo", "impermeable", "trench"}:
        state["product_role"] = "frio_principal"
        return

    if ("frio" in use_cases or "lluvia" in use_cases) and not state.get("product_role"):
        state["product_role"] = "frio_principal"


def _should_merge_with_previous(state: dict, query_norm: str) -> bool:
    """
    Determina si la consulta actual debe combinarse con el estado anterior.

    Considera marcadores explícitos de refinamiento y consultas breves con
    poca información estructurada nueva.

    Args:
        state: Estado extraído desde la consulta actual.
        query_norm: Consulta actual previamente normalizada.

    Returns:
        True si conviene heredar información del estado anterior;
        False en caso contrario.
    """
    refinement_markers = [
        r"\bparecid\w*\b",
        r"\bsimilar\w*\b",
        r"\botra opcion\b",
        r"\balguna?\b",
        r"\by tienes\b",
        r"\bmas barato\b",
        r"\bmas economico\b",
        r"\bmenos abrigad\w*\b",
        r"\babrigue menos\b",
        r"\besta en\b",
        r"\btiene talla\b",
    ]

    if any(re.search(p, query_norm) for p in refinement_markers):
        return True

    meaningful = [
        state.get("product_kind"),
        state.get("product_family"),
        state.get("color"),
        state.get("talla"),
        state.get("marca"),
        state.get("price_preference"),
    ]

    if sum(x is not None for x in meaningful) <= 2 and len(query_norm.split()) <= 5:
        return True

    return False


def _merge_query_states(previous: dict, current: dict) -> dict:
    """
    Combina el estado conversacional anterior con el estado actual.

    Conserva restricciones previas y reemplaza los campos para los que la
    consulta actual entrega nueva información. También fusiona casos de uso.

    Args:
        previous: Estado estructurado de la consulta anterior.
        current: Estado estructurado de la consulta actual.

    Returns:
        Nuevo diccionario con ambos estados combinados.
    """
    merged = dict(previous)
    merged["raw_query"] = current.get("raw_query", "")
    merged["query_norm"] = current.get("query_norm", "")

    for key in [
        "product_kind", "product_family", "product_role",
        "color", "talla", "precio_max", "marca", "price_preference"
    ]:
        if current.get(key) is not None:
            merged[key] = current[key]

    prev_cases = previous.get("use_cases", [])
    curr_cases = current.get("use_cases", [])
    merged["use_cases"] = list(dict.fromkeys(prev_cases + curr_cases))

    if re.search(r"\babrigue menos|abriga menos|menos abrigad\w*|aunque abrig", current.get("query_norm", "")):
        merged["product_family"] = "ropa_invierno"
        merged["product_role"] = "frio_liviano"

    return merged


def parse_query(query: str, products: list, previous_state: Optional[dict] = None) -> dict:
    """
    Analiza una consulta y construye su estado estructurado.

    Extrae tipo, familia, rol, casos de uso, color, talla, precio, marca y
    preferencias de precio. Opcionalmente combina el resultado con el estado
    de una consulta anterior.

    Args:
        query: Consulta del usuario en lenguaje natural.
        products: Lista completa de productos del catálogo.
        previous_state: Estado estructurado de la consulta anterior,
            o None si no existe contexto previo.

    Returns:
        Diccionario con la intención y los atributos estructurados de
        la consulta.
    """
    q = _normalize_text(query)
    kind, family = _detect_explicit_kind_and_family(q)

    state = {
        "raw_query": query,
        "query_norm": q,
        "product_kind": kind,
        "product_family": family,
        "family_inferred": False,
        "product_role": None,
        "use_cases": _extract_use_cases(q),
        "color": _extract_color(q),
        "talla": _extract_size(q),
        "precio_max": _extract_price_max(q),
        "marca": _extract_brand(q, products),
        "price_preference": _extract_price_preference(q),
    }

    _infer_family_from_use_cases(state)
    _adjust_role_from_query(state)

    if previous_state and _should_merge_with_previous(state, q):
        state = _merge_query_states(previous_state, state)
        _adjust_role_from_query(state)

    return state


_STOPWORDS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "de", "del", "que", "en", "y", "a", "con", "para", "por",
    "como", "me", "se", "si", "o", "pero", "algo", "quiero",
    "busco", "necesito", "tienes", "tengo", "hay", "alguna",
    "alguno", "opcion", "opciones", "producto", "productos"
}


def _keyword_score(query: str, product: dict) -> float:
    """
    Calcula un score de coincidencia léxica entre consulta y producto.

    Elimina stopwords de la consulta y mide la proporción de tokens
    relevantes que aparecen en el texto completo del producto.

    Args:
        query: Consulta del usuario en lenguaje natural.
        product: Diccionario con la información del producto.

    Returns:
        Score de coincidencia por palabras clave entre 0 y 1.
    """
    words = _tokens(query) - _STOPWORDS
    if not words:
        return 0.0

    product_words = _tokens(_product_full_text(product))
    matches = words & product_words
    return len(matches) / len(words)


def _use_case_score(state: dict, product: dict) -> float:
    """
    Calcula un boost de relevancia según los casos de uso detectados.

    Premia productos cuyas características, familia, tipo o rol coinciden
    con contextos como caminar, running, trekking, frío, oficina o estudio.

    Args:
        state: Estado estructurado de la consulta.
        product: Diccionario con la información del producto.

    Returns:
        Puntaje adicional asociado a los casos de uso.
    """
    text = _product_full_text(product)
    kind = _product_kind(product)
    family = _product_family(product)
    role = _product_role(product)
    score = 0.0
    use_cases = set(state.get("use_cases", []))

    if "caminar" in use_cases:
        if _contains_any(text, ["caminar", "caminata", "walk", "largas distancias"]):
            score += 0.22
        if _contains_any(text, ["comodidad", "comodo", "soporte", "liviano", "uso diario"]):
            score += 0.12
        if family == "calzado":
            score += 0.10

    if "running" in use_cases:
        if _contains_any(text, ["running", "correr", "trote", "entrenamiento"]):
            score += 0.22
        if kind == "zapatilla":
            score += 0.08

    if "trekking" in use_cases:
        if _contains_any(text, ["trekking", "outdoor", "cerro", "senderismo", "agarre"]):
            score += 0.22
        if family == "calzado":
            score += 0.08

    if "ciudad" in use_cases:
        if _contains_any(text, ["ciudad", "urbano", "casual", "uso diario", "diario"]):
            score += 0.16

    if "frio" in use_cases:
        if _contains_any(text, ["frio", "invierno", "abrigo", "termico", "bajas temperaturas"]):
            score += 0.24
        if role == "frio_principal":
            score += 0.16
        elif role == "frio_liviano":
            score += 0.08

    if "lluvia" in use_cases:
        if _contains_any(text, ["lluvia", "impermeable", "viento", "cortaviento"]):
            score += 0.20

    if "calor" in use_cases:
        if _contains_any(text, ["verano", "playa", "piscina", "liviano", "respirable"]):
            score += 0.18

    if "oficina" in use_cases:
        if _contains_any(text, ["oficina", "formal", "trabajo", "reunion", "evento"]):
            score += 0.18

    if "training" in use_cases:
        if _contains_any(text, ["gimnasio", "training", "entrenamiento", "yoga", "fuerza", "deporte"]):
            score += 0.18

    if "estudio" in use_cases:
        if _contains_any(text, ["colegio", "universidad", "notebook", "estudio", "trabajo"]):
            score += 0.16

    if "hogar" in use_cases and family == "hogar":
        score += 0.20

    if "pelo" in use_cases and family == "belleza":
        score += 0.20

    return score


def _size_matches(product: dict, wanted_size) -> bool:
    """
    Comprueba si un producto coincide con una talla solicitada.

    Primero compara los valores originales y luego sus representaciones
    normalizadas para tolerar formatos equivalentes.

    Args:
        product: Diccionario con la información del producto.
        wanted_size: Talla solicitada por el usuario.

    Returns:
        True si la talla está disponible; False en caso contrario.
    """
    tallas = product.get("tallas", [])

    if wanted_size in tallas:
        return True

    wanted_norm = _normalize_text(str(wanted_size))
    tallas_norm = [_normalize_text(str(t)) for t in tallas]
    return wanted_norm in tallas_norm


def _kind_matches(product: dict, wanted_kind: str) -> bool:
    """
    Comprueba si el tipo de un producto coincide con el solicitado.

    La intención genérica "zapato" acepta cualquier producto perteneciente
    a la familia de calzado.

    Args:
        product: Diccionario con la información del producto.
        wanted_kind: Tipo de producto solicitado.

    Returns:
        True si existe coincidencia; False en caso contrario.
    """
    actual = _product_kind(product)

    if wanted_kind == "zapato":
        return _product_family(product) == "calzado"

    return actual == wanted_kind


def _role_matches(product: dict, wanted_role: str) -> bool:
    """
    Comprueba si el rol funcional de un producto coincide con el solicitado.

    Args:
        product: Diccionario con la información del producto.
        wanted_role: Rol funcional solicitado.

    Returns:
        True si los roles coinciden; False en caso contrario.
    """
    return _product_role(product) == wanted_role


def _family_matches(product: dict, state: dict) -> bool:
    """
    Comprueba si un producto coincide con la familia solicitada.

    Cuando existe un rol específico de frío, la coincidencia se delega al
    rol funcional para mantener la distinción entre prendas y accesorios.

    Args:
        product: Diccionario con la información del producto.
        state: Estado estructurado de la consulta.

    Returns:
        True si el producto coincide con la familia o rol requerido;
        False en caso contrario.
    """
    wanted = state.get("product_family")
    actual = _product_family(product)

    if wanted is None:
        return True

    if state.get("product_role") in {"frio_principal", "frio_liviano", "frio_accesorio"}:
        return _role_matches(product, state["product_role"])

    return actual == wanted


def _structure_score(state: dict, product: dict) -> float:
    """
    Calcula un boost según coincidencias estructuradas.

    Premia coincidencias de familia, tipo, rol, marca, color y talla sin
    reemplazar los filtros duros aplicados posteriormente.

    Args:
        state: Estado estructurado de la consulta.
        product: Diccionario con la información del producto.

    Returns:
        Puntaje adicional por coincidencias estructuradas.
    """
    score = 0.0

    if state.get("product_family") and _family_matches(product, state):
        score += 0.18

    if state.get("product_kind") and _kind_matches(product, state["product_kind"]):
        score += 0.22

    if state.get("product_role") and _role_matches(product, state["product_role"]):
        score += 0.18

    if state.get("marca"):
        if state["marca"] in _normalize_text(product.get("marca", "")):
            score += 0.25

    if state.get("color"):
        colores = [_normalize_text(c) for c in product.get("colores", [])]
        if state["color"] in colores:
            score += 0.12

    if state.get("talla") is not None and _size_matches(product, state["talla"]):
        score += 0.12

    return score


def _price_score(products: list, product: dict, preference: Optional[str]) -> float:
    """
    Calcula un boost según la preferencia relativa de precio.

    Normaliza el precio del producto dentro del rango disponible y premia
    alternativas económicas o premium según la intención detectada.

    Args:
        products: Lista de productos usada para obtener el rango de precios.
        product: Producto cuyo precio se evaluará.
        preference: Preferencia de precio detectada, o None.

    Returns:
        Puntaje adicional asociado al precio.
    """
    if not preference:
        return 0.0

    prices = [p.get("precio", 0) for p in products if p.get("precio", 0) > 0]
    if not prices:
        return 0.0

    min_p, max_p = min(prices), max(prices)
    price = product.get("precio", max_p)

    if max_p == min_p:
        normalized_low = 1.0
    else:
        normalized_low = 1 - ((price - min_p) / (max_p - min_p))

    if preference in {"economic", "cheapest"}:
        return 0.18 * normalized_low

    if preference == "premium":
        return 0.12 * (1 - normalized_low)

    return 0.0


def _stock_score(product: dict) -> float:
    """
    Calcula un ajuste de score según disponibilidad de stock.

    Penaliza productos agotados y aplica un pequeño boost a productos
    disponibles.

    Args:
        product: Diccionario con la información del producto.

    Returns:
        Ajuste numérico asociado al stock.
    """
    return -0.35 if product.get("stock", 0) <= 0 else 0.03


def _is_kids_product(product: dict) -> bool:
    """
    Determina si un producto está orientado a público infantil.

    Args:
        product: Diccionario con la información del producto.

    Returns:
        True si se detectan señales de producto infantil;
        False en caso contrario.
    """
    text = _product_full_text(product)
    return bool(re.search(r"\b(nino|nina|ninos|ninas|infantil|kids)\b", text))


def _query_requests_kids(state: dict) -> bool:
    """
    Determina si la consulta solicita explícitamente productos infantiles.

    Args:
        state: Estado estructurado de la consulta.

    Returns:
        True si la consulta contiene señales de intención infantil;
        False en caso contrario.
    """
    q = state.get("query_norm", "")
    return bool(re.search(r"\b(nino|nina|ninos|ninas|infantil|kids|hijo|hija)\b", q))


def _passes_hard_filters(
    product: dict,
    state: dict,
    *,
    relax_role: bool = False,
) -> bool:
    """
    Comprueba si un producto satisface los filtros duros de la consulta.

    Evalúa tipo, rol, familia, color, talla, precio máximo y marca. También
    excluye productos infantiles cuando la consulta no los solicita.

    Si relax_role es True, omite únicamente la restricción exacta de rol
    para permitir un fallback controlado.

    Args:
        product: Diccionario con la información del producto.
        state: Estado estructurado de la consulta.
        relax_role: Indica si debe relajarse la restricción exacta de rol.

    Returns:
        True si el producto supera los filtros duros;
        False en caso contrario.
    """
    wanted_kind = state.get("product_kind")
    wanted_family = state.get("product_family")
    wanted_role = state.get("product_role")

    # Evitar productos infantiles si el usuario no pidió algo para niños.
    if _is_kids_product(product) and not _query_requests_kids(state):
        return False

    if wanted_kind and not _kind_matches(product, wanted_kind):
        return False

    # Filtro fuerte por rol, cuando existe.
    if wanted_role and not relax_role:
        if not _role_matches(product, wanted_role):
            return False

    # Filtro por familia.
    if wanted_family:
        actual_family = _product_family(product)

        # Para consultas de frío, permitimos familias relacionadas,
        # pero el rol decide si pasan o no.
        if wanted_family == "ropa_invierno":
            if actual_family not in {
                "ropa_invierno",
                "accesorio_invierno",
                "ropa_casual",
                "ropa_deportiva",
            }:
                return False
        elif actual_family != wanted_family:
            return False

    if state.get("color"):
        colores = [_normalize_text(c) for c in product.get("colores", [])]
        if state["color"] not in colores:
            return False

    if state.get("talla") is not None:
        if not _size_matches(product, state["talla"]):
            return False

    if state.get("precio_max") is not None:
        if product.get("precio", 0) > state["precio_max"]:
            return False

    if state.get("marca"):
        marca_prod = _normalize_text(product.get("marca", ""))
        if state["marca"] not in marca_prod:
            return False

    return True


def _apply_hard_filters(
    ranked: list[tuple[dict, float]],
    state: dict,
) -> list[tuple[dict, float]]:
    """
    Aplica filtros duros sobre una lista de productos rankeados.

    Si no existen resultados y la consulta contiene un rol de frío,
    realiza un fallback controlado relajando solo el rol y manteniendo
    las demás restricciones.

    Args:
        ranked: Lista de tuplas (producto, score) ordenadas por relevancia.
        state: Estado estructurado de la consulta.

    Returns:
        Lista filtrada de tuplas (producto, score).
    """
    filtered = [(p, s) for p, s in ranked if _passes_hard_filters(p, state)]

    if filtered:
        return filtered

    # Fallback controlado: si no existe nada con el rol exacto,
    # relajamos solo el rol, pero mantenemos familia/color/talla/marca.
    if state.get("product_role") in {
        "frio_principal",
        "frio_liviano",
        "frio_accesorio",
    }:
        relaxed = [
            (p, s)
            for p, s in ranked
            if _passes_hard_filters(p, state, relax_role=True)
        ]
        return relaxed

    return filtered


def _winter_relevance_score(state: dict, product: dict) -> float:
    """
    Calcula un ajuste específico para consultas de frío o lluvia.

    Distingue entre prendas principales, capas livianas y accesorios sin
    hardcodear productos concretos. También considera señales textuales de
    abrigo, lluvia y liviandad.

    Args:
        state: Estado estructurado de la consulta.
        product: Diccionario con la información del producto.

    Returns:
        Ajuste numérico de relevancia para contextos de frío o lluvia.
    """
    use_cases = set(state.get("use_cases", []))
    if "frio" not in use_cases and "lluvia" not in use_cases:
        return 0.0

    text = _product_full_text(product)
    kind = _product_kind(product)
    role = _product_role(product)

    score = 0.0

    if role == "frio_principal":
        score += 0.28

    if role == "frio_liviano":
        score -= 0.12

        # En follow-ups tipo "abrigue menos", preferimos prendas de uso diario
        # antes que una primera capa técnica.
        if state.get("product_role") == "frio_liviano":
            kind = _product_kind(product)

            if kind in {"poleron", "sweater", "polar"}:
                score += 0.18

            if kind == "primera_capa":
                score -= 0.10

    if role == "frio_accesorio":
        score -= 0.35

    # Señales fuertes de abrigo real.
    if _contains_any(
        text,
        [
            "abrigada",
            "abrigado",
            "abrigo",
            "bajas temperaturas",
            "frio",
            "termico",
            "invierno",
        ],
    ):
        score += 0.22

    # Parka suele ser la recomendación principal más natural para frío/lluvia.
    if kind == "parka":
        score += 0.18

    # Chaqueta/abrigo también son recomendables como prenda principal.
    if kind in {"chaqueta", "abrigo", "trench"}:
        score += 0.10

    # Impermeable sirve para lluvia, pero no necesariamente abriga mucho.
    # Si el usuario pidió frío, no lo premiamos tanto solo por ser barato.
    if kind == "impermeable" and "frio" in use_cases:
        score -= 0.08

    # Si el producto dice "liviano", no es malo, pero para frío general
    # debe quedar detrás de una prenda claramente abrigada.
    if "liviano" in text and "frio" in use_cases:
        score -= 0.06

    return score


def search_products_with_state(
    query: str,
    previous_state: Optional[dict] = None,
    top_k: int = MAX_PRODUCTOS_RESULTADO,
) -> tuple[list, dict]:
    """
    Realiza una búsqueda híbrida de productos manteniendo estado conversacional.

    Combina similitud semántica, coincidencia por palabras clave, estructura
    de intención, casos de uso, relevancia de invierno, precio y stock.
    Luego aplica filtros duros, umbrales de relevancia y ordenamientos
    especiales por preferencia de precio.

    Args:
        query: Consulta del usuario en lenguaje natural.
        previous_state: Estado estructurado de la consulta anterior,
            o None si no existe contexto previo.
        top_k: Número máximo de productos a retornar.

    Returns:
        Tupla con la lista de productos relevantes y el estado estructurado
        resultante de la consulta.
    """
    products, embeddings_matrix = load_embeddings()
    state = parse_query(query, products, previous_state)

    query_vec = embed_query(query).reshape(1, -1)
    semantic_scores = cosine_similarity(query_vec, embeddings_matrix)[0]
    keyword_scores = np.array([_keyword_score(query, p) for p in products])

    adjusted_scores = []
    for i, product in enumerate(products):
        base = 0.55 * float(semantic_scores[i]) + 0.35 * float(keyword_scores[i])
        boost = (
            _structure_score(state, product)
            + _use_case_score(state, product)
            + _winter_relevance_score(state, product)
            + _price_score(products, product, state.get("price_preference"))
            + _stock_score(product)
        )
        adjusted_scores.append(base + boost)

    ranked = sorted(
        [(products[i], float(adjusted_scores[i])) for i in range(len(products))],
        key=lambda item: item[1],
        reverse=True,
    )

    filtered_ranked = _apply_hard_filters(ranked, state)

    if not filtered_ranked and any(state.get(k) for k in [
        "product_kind",
        "product_family",
        "color",
        "talla",
        "precio_max",
        "marca",
        "product_role",
    ]):
        return [], state

    final_ranked = filtered_ranked if filtered_ranked else ranked

    threshold = 0.08 if filtered_ranked else 0.14
    relevant = [(p, s) for p, s in final_ranked if s >= threshold]

    if not relevant:
        return [], state

    if state.get("price_preference") == "cheapest":
        relevant = sorted(
            relevant,
            key=lambda item: (item[0].get("precio", 0), -item[1])
        )
    elif state.get("price_preference") == "premium":
        relevant = sorted(
            relevant,
            key=lambda item: (item[0].get("precio", 0), item[1]),
            reverse=True
        )

    return [p for p, _ in relevant[:top_k]], state


def search_products(query: str, top_k: int = MAX_PRODUCTOS_RESULTADO) -> list:
    """
    Realiza una búsqueda de productos sin estado conversacional previo.

    Actúa como wrapper simplificado de search_products_with_state.

    Args:
        query: Consulta del usuario en lenguaje natural.
        top_k: Número máximo de productos a retornar.

    Returns:
        Lista de productos ordenados por relevancia.
    """
    products, _state = search_products_with_state(
        query,
        previous_state=None,
        top_k=top_k,
    )
    return products


_POLICY_INTENT_TERMS = [
    "cambio", "cambiar", "cambiarlo", "cambiarla", "cambiarlos", "cambiarlas",
    "devolucion", "devolver", "reembolso", "dinero", "garantia", "falla",
    "boleta", "comprobante", "no me queda", "me quedo chico", "me quedo grande",
    "pago", "pagos", "tarjeta", "debito", "credito", "transferencia",
    "despacho", "envio", "entrega", "seguimiento", "pedido", "retiro", "retirar",
    "soporte", "reclamo", "privacidad", "seguridad", "inventar", "numero de seguimiento",
]


def is_policy_query(query: str) -> bool:
    """
    Determina si una consulta corresponde a políticas de la tienda.

    Detecta términos y patrones relacionados con cambios, devoluciones,
    pagos, despachos, seguimiento, soporte y otros procesos de atención.

    Args:
        query: Consulta del usuario en lenguaje natural.

    Returns:
        True si la consulta parece referirse a políticas;
        False en caso contrario.
    """
    q = _normalize_text(query)

    if any(term in q for term in _POLICY_INTENT_TERMS):
        return True

    if re.search(r"\bpuedo\b.*\b(cambiar|devolver|pedir|pagar|retirar)\b", q):
        return True

    if re.search(r"\bcomo\b.*\b(seguimiento|pedido|despacho|envio|pago)\b", q):
        return True

    return False


def _policy_text(policy: dict) -> str:
    """
    Construye el texto normalizado de una política.

    Combina los campos de tema y contenido para búsquedas semánticas
    y léxicas.

    Args:
        policy: Diccionario con la información de una política.

    Returns:
        Texto normalizado de la política.
    """
    return _normalize_text(
        f"{policy.get('tema', '')} {policy.get('contenido', '')}"
    )


def _load_policy_embeddings() -> np.ndarray:
    """
    Carga o calcula los embeddings de las políticas usando caché.

    Los vectores se generan solo la primera vez y luego se reutilizan
    durante la ejecución del módulo.

    Returns:
        Matriz NumPy con los embeddings de todas las políticas.
    """
    global _policy_embeddings

    policies = _load_policies()

    if _policy_embeddings is None:
        vectors = [embed_query(_policy_text(policy)) for policy in policies]
        _policy_embeddings = np.array(vectors)

    return _policy_embeddings


def _policy_keyword_score(query: str, policy: dict) -> float:
    """
    Calcula un score léxico entre una consulta y una política.

    Elimina stopwords y mide la proporción de tokens relevantes de la
    consulta presentes en el texto de la política.

    Args:
        query: Consulta del usuario en lenguaje natural.
        policy: Diccionario con la información de una política.

    Returns:
        Score de coincidencia por palabras clave entre 0 y 1.
    """
    words = _tokens(query) - _STOPWORDS
    if not words:
        return 0.0

    policy_words = _tokens(_policy_text(policy))
    return len(words & policy_words) / len(words)


def search_policies(query: str) -> Optional[str]:
    """
    Busca la política de la tienda más relevante para una consulta.

    Combina similitud semántica y coincidencia por palabras clave. Aplica
    un umbral mínimo salvo que la consulta tenga intención explícita de
    política.

    Args:
        query: Consulta del usuario en lenguaje natural.

    Returns:
        Texto con el tema y contenido de la política más relevante,
        o None si no existe una coincidencia suficiente.
    """
    policies = _load_policies()
    if not policies:
        return None

    query_vec = embed_query(query).reshape(1, -1)
    policy_embeddings = _load_policy_embeddings()
    semantic_scores = cosine_similarity(query_vec, policy_embeddings)[0]
    keyword_scores = np.array([
        _policy_keyword_score(query, p)
        for p in policies
    ])

    combined_scores = 0.55 * semantic_scores + 0.45 * keyword_scores

    best_idx = int(np.argmax(combined_scores))
    best_score = float(combined_scores[best_idx])

    if best_score < 0.08 and not is_policy_query(query):
        return None

    best_policy = policies[best_idx]
    tema = best_policy.get("tema", "política")
    contenido = best_policy.get("contenido", "")

    return f"Tema: {tema}\n{contenido}"