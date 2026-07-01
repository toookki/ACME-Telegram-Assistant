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

import google.generativeai as genai
from config.settings import GEMINI_API_KEY

# Configurar la API de Gemini al importar este módulo
genai.configure(api_key=GEMINI_API_KEY)

# Modelo a usar
GEMINI_MODEL = "gemini-2.5-flash-lite"  # Rápido y gratuito. Alternativa: "gemini-1.5-pro"


def _format_products(products: list) -> str:
    """
    Formatea la lista de productos como texto estructurado para el prompt.
    Incluye numeración para que el LLM pueda referirse a ellos ("la primera opción").
    """
    if not products:
        return "No se encontraron productos relevantes."

    lines = []
    for i, p in enumerate(products, start=1):
        colores = ", ".join(p.get("colores", []))
        tallas = ", ".join(str(t) for t in p.get("tallas", []))
        precio = f"${p.get('precio', 0):,}".replace(",", ".")
        lines.append(
            f"{i}. {p['nombre']} ({p.get('marca', '')}) | "
            f"Precio: {precio} | "
            f"Colores: {colores} | "
            f"Tallas: {tallas} | "
            f"Stock: {p.get('stock', 0)} | "
            f"Descripción: {p.get('descripcion', '')} | "
            f"Categoría: {p.get('categoria', '')}"
        )
    return "\n".join(lines)


def _format_history(history: list) -> str:
    """Formatea el historial de conversación para el prompt."""
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
1. NUNCA inventes productos, precios, tallas, colores ni información que no esté en los datos proporcionados.
2. NUNCA inventes códigos de seguimiento, números de pedido ni información de transacciones.
3. Si no tienes información suficiente, dilo claramente y sugiere contactar soporte.
4. Usa el listado numerado de productos para resolver referencias como "la segunda", "esa negra", "el más barato".
5. Responde siempre en español, de manera amable y natural.
6. Sé conciso pero informativo. No hagas listas innecesarias cuando puedes responder en prosa.
7. Si el cliente pregunta por políticas, usa solo la información de políticas proporcionada.
"""


def generate_response(
    user_message: str,
    products: list,
    history: list,
    policy_info: str = None,
) -> str:
    """
    Genera una respuesta natural usando Gemini.

    Args:
        user_message: El mensaje actual del usuario.
        products: Lista de productos recuperados por search.py (puede estar vacía).
        history: Historial de conversación reciente del usuario.
        policy_info: Texto de política relevante encontrado (puede ser None).

    Returns:
        str: Respuesta en lenguaje natural para enviar al usuario.
    """
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT,
    )

    # Construir el contexto del prompt
    context_parts = []

    if history:
        context_parts.append(f"=== HISTORIAL RECIENTE ===\n{_format_history(history)}")

    if products:
        context_parts.append(
            f"=== PRODUCTOS DISPONIBLES (en orden de relevancia) ===\n{_format_products(products)}"
        )
    else:
        context_parts.append("=== PRODUCTOS ===\nNo se encontraron productos relevantes para esta consulta.")

    if policy_info:
        context_parts.append(f"=== INFORMACIÓN DE POLÍTICAS DE LA TIENDA ===\n{policy_info}")

    context_parts.append(f"=== MENSAJE DEL USUARIO ===\n{user_message}")

    context_parts.append(
        "Responde al usuario basándote ÚNICAMENTE en la información anterior. "
        "No inventes nada que no esté en los datos proporcionados."
    )

    full_prompt = "\n\n".join(context_parts)

    try:
        response = model.generate_content(full_prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error en Gemini: {e}")
        return (
            "Lo siento, tuve un problema técnico al procesar tu consulta. "
            "Por favor intenta de nuevo en unos segundos."
        )
