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
    Incluye numeración para que el LLM pueda referirse a ellos ("la primera opción").
    """
    if not products:
        return "No se encontraron productos relevantes."

    lines = []
    for i, p in enumerate(products, start=1):
        colores = ", ".join(p.get("colores", []))
        tallas = ", ".join(str(t) for t in p.get("tallas", []))
        precio = f"${p.get('precio', 0):,}".replace(",", ".")
        stock = p.get("stock", 0)
        stock_label = f"{stock} (SIN STOCK - No disponible actualmente)" if stock <= 0 else f"{stock} unidades disponibles"
        lines.append(
            f"{i}. {p['nombre']} ({p.get('marca', '')}) | "
            f"Precio: {precio} | "
            f"Colores: {colores} | "
            f"Tallas: {tallas} | "
            f"Stock: {stock_label} | "
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
8. Si un producto esta marcado como "SIN STOCK", NO lo recomiendes como primera opción. Puedes mencionarlo solo si 
el cliente pregunta especificamente por él, aclarando que no está disponible por el momento y sugiriendo una alternativa 
con stock si existe entre los productos entregados.
"""


def generate_response(user_message: str, products: list, history: list, policy_info: str = None) -> str:
    """
    Genera una respuesta natural usando Gemini.

    Args:
        user_message: El mensaje actual del usuario.
        products: Lista de productos recuperados por search.py (puede estar vacía).
        history: Historial de conversación reciente del usuario.
        policy_info: Texto de política relevante encontrado (puede ser None).

    Retorna:
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
        error_str = str(e)
        # Si es un error de cuota (429), esperar el tiempo sugerido y reintentar una vez
        if "429" in error_str or "quota" in error_str.lower():
            wait_time = _extract_retry_delay(error_str)
            print(f"Cuota excedida. Reintentando en {wait_time}s...")
            time.sleep(wait_time)
            try:
                response = model.generate_content(full_prompt)
                return response.text.strip()
            except Exception as retry_error:
                print(f"Error en Gemini (Tras reintento): {retry_error}")
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
    """
    match = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", error_str)
    if match:
        return int(match.group(1)) + 1  # +1 segundo de margen
    match = re.search(r"retry in ([\d.]+)s", error_str)
    if match:
        return int(float(match.group(1))) + 1
    return default