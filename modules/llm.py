"""
modules/llm.py
==============
Capa de generación de respuestas con Google Gemini.

Responsabilidades:
- Construir el prompt del sistema con instrucciones del agente.
- Formatear los productos recuperados y el historial conversacional.
- Llamar a la API de Gemini y retornar la respuesta en texto.

IMPORTANTE: El LLM NUNCA busca productos ni inventa información.
Solo redacta respuestas naturales basándose en lo que recibe.
Los productos vienen previamente filtrados por search.py.
"""

import re
import time

import google.generativeai as genai

from config.settings import GEMINI_API_KEY, GEMINI_MODEL


# Configurar la API de Gemini al importar este módulo
genai.configure(api_key=GEMINI_API_KEY)


def _format_products(products: list) -> str:
    """
    Formatea la lista de productos como texto estructurado para el prompt.
    Incluye numeración para que el LLM pueda referirse a ellos
    ("la primera opción").

    Args:
        products: Lista de diccionarios de productos recuperados.

    Returns:
        Texto estructurado con la información de los productos.
    """
    if not products:
        return "No se encontraron productos relevantes."

    lines = []

    for i, p in enumerate(products, start=1):
        colores = ", ".join(str(c) for c in p.get("colores", []))
        tallas = ", ".join(str(t) for t in p.get("tallas", []))
        precio = f"${p.get('precio', 0):,}".replace(",", ".")
        stock = p.get("stock", 0)

        stock_text = (
            "sin stock actualmente"
            if stock <= 0
            else f"{stock} unidades disponibles"
        )

        lines.append(
            f"Producto interno {i}\n"
            f"Nombre: {p.get('nombre', '')}\n"
            f"Marca: {p.get('marca', '')}\n"
            f"Categoría: {p.get('categoria', '')}\n"
            f"Precio: {precio}\n"
            f"Colores: {colores}\n"
            f"Tallas: {tallas}\n"
            f"Stock: {stock_text}\n"
            f"Descripción: {p.get('descripcion', '')}\n"
            f"Tags: {', '.join(str(t) for t in p.get('tags', []))}"
        )

    return "\n\n".join(lines)


def _format_history(history: list) -> str:
    """
    Formatea el historial de conversación para el prompt.

    Args:
        history: Lista de mensajes del historial conversacional.

    Returns:
        Texto con los mensajes formateados por rol.
    """
    if not history:
        return ""

    lines = []

    for msg in history:
        role = "Usuario" if msg["role"] == "user" else "Agente"
        lines.append(f"{role}: {msg['content']}")

    return "\n".join(lines)


SYSTEM_PROMPT = """Eres un asistente virtual de una tienda de retail llamada ACME.
Tu función es ayudar a los clientes a encontrar productos y responder preguntas sobre políticas de la tienda.

REGLAS ESTRICTAS:
1. NUNCA inventes productos, precios, tallas, colores, stock ni información que no esté en los datos proporcionados.
2. NUNCA inventes códigos de seguimiento, números de pedido ni información de transacciones.
3. Si no tienes información suficiente, dilo claramente y sugiere contactar soporte.
4. Usa el orden de los productos internos para entender referencias como "la segunda", "esa negra" o "el más barato".
5. Responde siempre en español, de manera amable y natural.
6. Responde en prosa breve. NO hagas listados numerados, NO uses viñetas, NO copies el bloque de productos y NO muestres datos en formato JSON.
7. Si hay varios productos, menciona solo los más relevantes en una frase natural. El sistema agregará automáticamente el listado numerado visible al final.
8. Si el cliente pregunta por políticas, usa solo la información de políticas proporcionada.
9. Si un producto está sin stock, no lo recomiendes como primera opción. Puedes mencionarlo solo si el cliente pregunta específicamente por él, aclarando que no está disponible.
10. No repitas al final la lista completa de productos. Esa lista la genera el código después de tu respuesta.
11. No saludes con "Hola" en cada turno. Saluda solo al inicio de la conversación; en respuestas de seguimiento responde directamente.
12. Cuando recibas productos ordenados por relevancia interna, tu recomendación principal debe ser el Producto interno 1. Puedes mencionar otros productos como alternativas,
pero no presentes un producto posterior como la mejor opción principal salvo que el usuario pida explícitamente "lo más barato".
"""


def _remove_repeated_greeting(text: str, history: list) -> str:
    """
    Evita que el bot salude en cada turno de una misma conversación.

    Permite el saludo solo si no hay historial previo.
    Si ya existe historial, elimina saludos iniciales tipo:
    "Hola", "¡Hola!", "Hola, claro", etc.

    Args:
        text: Texto generado por el modelo.
        history: Historial de conversación reciente del usuario.

    Returns:
        Texto sin saludo inicial repetido cuando ya existe historial.
    """
    if not history:
        return text

    patterns = [
        r"^¡?hola!?\s*,?\s*",
        r"^buenas\s*,?\s*",
        r"^buenos dias\s*,?\s*",
        r"^buenas tardes\s*,?\s*",
        r"^buenas noches\s*,?\s*",
    ]

    cleaned = text.strip()

    for pattern in patterns:
        cleaned = re.sub(
            pattern,
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()

    # Si después de sacar "Hola," queda partiendo con minúscula,
    # arreglamos la primera letra.
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]

    return cleaned


def generate_response(
    user_message: str,
    products: list | None,
    history: list,
    policy_info: str = None,
) -> str:
    """
    Genera una respuesta natural usando Gemini.

    Args:
        user_message: Mensaje actual del usuario.
        products: Lista de productos recuperados por search.py
            (puede estar vacía o ser None).
        history: Historial de conversación reciente del usuario.
        policy_info: Texto de política relevante encontrado
            (puede ser None).

    Returns:
        Respuesta en lenguaje natural para enviar al usuario.
    """
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT,
    )

    # Construir el contexto del prompt
    context_parts = []

    if history:
        context_parts.append(
            f"=== HISTORIAL RECIENTE ===\n{_format_history(history)}"
        )

    if policy_info:
        context_parts.append(
            f"=== INFORMACIÓN DE POLÍTICAS DE LA TIENDA ===\n{policy_info}"
        )
    else:
        if products:
            context_parts.append(
                "=== PRODUCTOS DISPONIBLES, EN ORDEN DE RELEVANCIA INTERNA ===\n"
                "IMPORTANTE: El Producto interno 1 es la recomendación principal "
                "calculada por el buscador. Debes recomendar primero ese producto. "
                "Los productos siguientes son alternativas secundarias.\n\n"
                f"{_format_products(products)}"
            )
        else:
            context_parts.append(
                "=== PRODUCTOS ===\n"
                "No se encontraron productos relevantes para esta consulta."
            )

    context_parts.append(
        f"=== MENSAJE DEL USUARIO ===\n{user_message}"
    )

    context_parts.append(
        "Responde al usuario basándote ÚNICAMENTE en la información anterior. "
        "No inventes nada que no esté en los datos proporcionados."
    )

    full_prompt = "\n\n".join(context_parts)

    try:
        response = model.generate_content(
            full_prompt,
            generation_config={
                "temperature": 0.2,
                "top_p": 0.8,
            },
        )

        response_text = response.text.strip()
        return _remove_repeated_greeting(response_text, history)

    except Exception as e:
        error_str = str(e)

        # Si es un error de cuota (429), esperar el tiempo sugerido
        # y reintentar una vez.
        if "429" in error_str or "quota" in error_str.lower():
            wait_time = _extract_retry_delay(error_str)
            print(f"Cuota excedida. Reintentando en {wait_time}s...")
            time.sleep(wait_time)

            try:
                response = model.generate_content(
                    full_prompt,
                    generation_config={
                        "temperature": 0.2,
                        "top_p": 0.8,
                    },
                )

                response_text = response.text.strip()
                return _remove_repeated_greeting(response_text, history)

            except Exception as retry_error:
                print(f"Error en Gemini tras reintento: {retry_error}")
                return (
                    "Estoy recibiendo muchas consultas en este momento. "
                    "Por favor espera unos segundos y vuelve a preguntar."
                )

        print(f"Error en Gemini: {e}")
        return (
            "Lo siento, tuve un problema técnico al procesar tu consulta. "
            "Por favor intenta de nuevo en unos segundos."
        )


def _extract_retry_delay(error_str: str, default: int = 15) -> int:
    """
    Extrae el tiempo de espera sugerido por la API desde el mensaje de error.
    Si no lo encuentra, retorna un valor por defecto.

    Args:
        error_str: Mensaje de error retornado por la API.
        default: Tiempo de espera por defecto en segundos.

    Returns:
        Tiempo de espera en segundos antes de reintentar.
    """
    match = re.search(
        r"retry_delay\s*\{\s*seconds:\s*(\d+)",
        error_str,
    )
    if match:
        return int(match.group(1)) + 1  # +1 segundo de margen

    match = re.search(r"retry in ([\d.]+)s", error_str)
    if match:
        return int(float(match.group(1))) + 1

    return default