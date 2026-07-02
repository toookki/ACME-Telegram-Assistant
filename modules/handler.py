"""
modules/handler.py
==================
Manejador de mensajes de Telegram.

Es el cerebro del flujo conversacional. Orquesta todos los módulos:
  1. Recibe el mensaje del usuario vía Telegram.
  2. Consulta la memoria conversacional del usuario.
  3. Llama al motor de búsqueda híbrida (search.py).
  4. Busca información de políticas si es relevante.
  5. Llama al LLM con toda la información recopilada.
  6. Actualiza la memoria con el turno actual.
  7. Envía la respuesta al usuario en Telegram.

También maneja los comandos /start /reset y /help.
"""

from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ParseMode

from modules import memory, search, llm, reference_resolver


# Palabras clave que sugieren una pregunta de políticas
POLICY_KEYWORDS = [
    "cambio", "cambiar", "cambiarlos", "cambiarlas", "devolucion",
    "devolución", "garantia", "garantía", "pago", "tarjeta", 
    "transferencia", "despacho", "envio", "envío", "entrega", 
    "seguimiento", "pedido", "retirar", "retiro", "stock", "disponible", 
    "promo", "promoción", "promocion", "descuento", "soporte", "reclamo", 
    "seguridad", "privacidad"
]


def _is_policy_query(text: str) -> bool:
    """Detecta si la consulta es sobre políticas de la tienda."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in POLICY_KEYWORDS)


def _format_product_list(products: list) -> str:
    """
    Genera un listado numerado simple (número, nombre, precio) en el
    MISMO orden del arreglo `products`.

    Se agrega al final de la respuesta del LLM cuando hay más de un
    producto. Es clave que este listado lo genere el código (no el
    LLM): así el número que el usuario ve en pantalla ("2.") queda
    garantizado que coincide con products[1], que es exactamente lo
    que se guarda en last_products. Si dejáramos que el LLM narrara
    el orden en prosa libre, no hay garantía de que lo haga en el
    mismo orden del arreglo interno, y "la segunda" volvería a fallar.
    """
    lines = []
    for i, p in enumerate(products, start=1):
        precio = f"${p.get('precio', 0):,}".replace(",", ".")
        lines.append(f"{i}. {p.get('nombre', '(sin nombre)')} — {precio}")
    return "\n".join(lines)


async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handler para el comando /start.
    Saluda al usuario y limpia su memoria si existía.
    """
    user_id = update.effective_user.id
    name = update.effective_user.first_name or "cliente"
    memory.clear_user(user_id)

    await update.message.reply_text(
        f"¡Hola <b>{name}</b>! Soy el asistente virtual de ACME.\n\n"
        "Puedo ayudarte a encontrar productos, responder preguntas sobre "
        "cambios, devoluciones, despacho y más.\n\n"
        "Escribe /help si quieres ver ejemplos de lo que puedo hacer.\n\n"
        "¿En qué te puedo ayudar hoy?", parse_mode=ParseMode.HTML
    )


async def handle_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handler para el comando /reset.
    Limpia la memoria conversacional del usuario.
    """
    user_id = update.effective_user.id
    memory.clear_user(user_id)
    await update.message.reply_text(
        "Conversación reiniciada. ¿En qué te puedo ayudar?"
    )

async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handler para el comando /help.
    Explica que puede hacer el bot y da ejemplos de mensajes que entiende.
    """
    help_text = (
        "<b>Asistente Virtual ACME</b>\n\n"
        "Puedo ayudarte a:\n"
        "• Buscar productos por descripción, color, talla, marca o precio\n"
        "• Recordar el contexto de la conversación (ej: \"la segunda\", \"esa negra\")\n"
        "• Responder preguntas sobre cambios, devoluciones, pagos, despacho y garantía\n\n"
        "<b>Ejemplos de mensajes que puedes escribir:</b>\n"
        "\"Busco algo cómodo para caminar mucho\"\n"
        "\"Quiero zapatillas negras talla 42\"\n"
        "\"¿Cuál me conviene si quiero algo más barato?\"\n"
        "\"¿Puedo cambiar un producto si no me queda?\"\n\n"
        "<b>Comandos disponibles:</b>\n"
        "/start - Iniciar o reiniciar la conversación\n"
        "/reset - Borrar el historial de la conversación actual\n"
        "/help - Mostrar esta ayuda"
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.HTML)

async def handle_unsupported(update:Update, context:ContextTypes.DEFAULT_TYPE):
    """
    Handler para mensajes que no son texto.
    Sin este handler, telegram no llama a ningun handler y el bot se queda en silencio,
    lo que parece un error para el usuario.
    """
    await update.message.reply_text(
        "Por ahora solo puedo entender mensajes de texto :)\n"
        "Cuentame en palabras qué estás buscando"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handler principal para mensajes de texto.
 
    Flujo:
    1. Obtener user_id y texto del mensaje.
    2. Mostrar indicador de "escribiendo..." en Telegram.
    3. Recuperar contexto de memoria.
    4. Buscar productos relevantes (siempre).
    5. Si la consulta parece de política, buscar en políticas también.
    6. Llamar al LLM con toda la información.
    7. Actualizar memoria (agregar turno al historial + actualizar productos mostrados).
    8. Responder al usuario.
    """
    user_id = update.effective_user.id
    user_text = update.message.text.strip()
 
    # Indicador de "escribiendo..." mientras procesamos
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action="typing"
    )
 
    # 1. Obtener contexto conversacional del usuario
    ctx = memory.get_context(user_id)
    history = ctx["history"]
    last_products = ctx["last_products"]
 
    # 2. Determinar primero si la consulta es de política. Si lo es, NO
    #    se buscan ni se resuelven productos: se le pasa None al LLM
    #    para que responda solo con la información de política, sin
    #    arrastrar productos irrelevantes (y sin que se les agregue el
    #    listado numerado del punto 6.5).
    is_policy_query = _is_policy_query(user_text)
 
    if is_policy_query:
        products = None
    else:
        # Ver primero si el usuario está señalando algo que YA le
        # mostramos ("la segunda", "esa negra", "la más barata"). Si es
        # así, se usa directamente ese producto de last_products y NO se
        # vuelve a buscar en el catálogo completo.
        resolved_products = reference_resolver.resolve_reference(user_text, last_products)
        if resolved_products:
            products = resolved_products
        else:
            products = search.search_products(user_text)
 
    # 4. Búsqueda de políticas (solo si la consulta lo sugiere)
    policy_info = None
    if is_policy_query:
        policy_info = search.search_policies(user_text)
 
    # 5. Agregar el mensaje del usuario al historial ANTES de llamar al LLM
    memory.update_history(user_id, "user", user_text)
 
    # 6. Generar respuesta con el LLM
    response_text = llm.generate_response(
        user_message=user_text,
        products=products,
        history=history,        # Historial previo (sin el mensaje actual)
        policy_info=policy_info,
    )
 
    # 6.5 Si se muestra más de un producto, agregar un listado numerado
    #     explícito y determinístico (ver _format_product_list). `products`
    #     puede ser None en una consulta de política, por eso se chequea
    #     primero.
    if products and len(products) > 1:
        response_text = f"{response_text}\n\n{_format_product_list(products)}"
 
    # 7. Actualizar memoria con la respuesta del agente y los productos mostrados
    memory.update_history(user_id, "assistant", response_text)
    if products:
        memory.update_last_products(user_id, products)
 
    # 8. Enviar respuesta al usuario
    # Telegram tiene límite de 4096 caracteres por mensaje
    if len(response_text) > 4096:
        for i in range(0, len(response_text), 4096):
            await update.message.reply_text(response_text[i:i+4096])
    else:
        await update.message.reply_text(response_text)
    await update.message.reply_text(response_text)