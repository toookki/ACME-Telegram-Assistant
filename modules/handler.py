"""
modules/handler.py
==================
Manejador de mensajes de Telegram.

Es el cerebro del flujo conversacional. Orquesta todos los módulos:
  1. Recibe el mensaje del usuario vía Telegram.
  2. Consulta la memoria conversacional del usuario.
  3. Detecta consultas sobre políticas de la tienda.
  4. Resuelve referencias a productos mostrados previamente.
  5. Llama al motor de búsqueda híbrida cuando corresponde.
  6. Genera una respuesta natural mediante el LLM.
  7. Actualiza la memoria con el turno actual.
  8. Envía la respuesta al usuario en Telegram.

También maneja los comandos /start, /reset y /help, además de
mensajes no soportados.
"""

from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ParseMode

from modules import memory, search, llm, reference_resolver


def _format_product_list(products: list) -> str:
    """
    Genera un listado numerado de productos respetando el orden interno.

    Cada elemento incluye número, nombre y precio. El listado se agrega
    al final de la respuesta del LLM cuando existen varios productos.

    Es importante que este listado sea generado por código y no por el
    LLM, ya que así la posición visible para el usuario coincide exactamente
    con el índice interno almacenado en last_products. Por ejemplo, el
    producto mostrado como "2." corresponde siempre a products[1].

    Args:
        products: Lista ordenada de diccionarios de productos.

    Returns:
        Texto con los productos numerados, uno por línea.
    """
    lines = []

    for i, p in enumerate(products, start=1):
        precio = f"${p.get('precio', 0):,}".replace(",", ".")
        lines.append(
            f"{i}. {p.get('nombre', '(sin nombre)')} — {precio}"
        )

    return "\n".join(lines)


async def handle_start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    """
    Maneja el comando /start.

    Limpia la memoria conversacional existente del usuario y envía
    un mensaje inicial de bienvenida con una descripción breve de
    las capacidades del asistente.

    Args:
        update: Actualización recibida desde Telegram.
        context: Contexto asociado al handler de Telegram.

    Returns:
        None.
    """
    user_id = update.effective_user.id
    name = update.effective_user.first_name or "cliente"

    memory.clear_user(user_id)

    await update.message.reply_text(
        f"¡Hola <b>{name}</b>! Soy el asistente virtual de ACME.\n\n"
        "Puedo ayudarte a encontrar productos, responder preguntas sobre "
        "cambios, devoluciones, despacho y más.\n\n"
        "Escribe /help si quieres ver ejemplos de lo que puedo hacer.\n\n"
        "¿En qué te puedo ayudar hoy?",
        parse_mode=ParseMode.HTML,
    )


async def handle_reset(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    """
    Maneja el comando /reset.

    Elimina toda la memoria conversacional asociada al usuario y
    confirma que la conversación fue reiniciada.

    Args:
        update: Actualización recibida desde Telegram.
        context: Contexto asociado al handler de Telegram.

    Returns:
        None.
    """
    user_id = update.effective_user.id

    memory.clear_user(user_id)

    await update.message.reply_text(
        "Conversación reiniciada. ¿En qué te puedo ayudar?"
    )


async def handle_help(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    """
    Maneja el comando /help.

    Explica las principales capacidades del bot, entrega ejemplos
    de consultas válidas y muestra los comandos disponibles.

    Args:
        update: Actualización recibida desde Telegram.
        context: Contexto asociado al handler de Telegram.

    Returns:
        None.
    """
    help_text = (
        "<b>Asistente Virtual ACME</b>\n\n"
        "Puedo ayudarte a:\n"
        "• Buscar productos por descripción, color, talla, marca o precio\n"
        "• Recordar el contexto de la conversación, por ejemplo: "
        "\"la segunda\" o \"esa negra\"\n"
        "• Responder preguntas sobre cambios, devoluciones, pagos, "
        "despacho y garantía\n\n"
        "<b>Ejemplos de mensajes:</b>\n"
        "\"Busco algo cómodo para caminar mucho\"\n"
        "\"Quiero zapatillas negras talla 42\"\n"
        "\"¿Cuál me conviene si quiero algo más barato?\"\n"
        "\"¿Puedo cambiar un producto si no me queda?\"\n\n"
        "<b>Comandos:</b>\n"
        "/start - Iniciar o reiniciar\n"
        "/reset - Borrar historial\n"
        "/help - Mostrar ayuda"
    )

    await update.message.reply_text(
        help_text,
        parse_mode=ParseMode.HTML,
    )


async def handle_unsupported(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    """
    Maneja mensajes cuyo contenido no es texto.

    Evita que el bot permanezca en silencio frente a formatos que todavía
    no puede procesar, como imágenes, audios, documentos o stickers.

    Args:
        update: Actualización recibida desde Telegram.
        context: Contexto asociado al handler de Telegram.

    Returns:
        None.
    """
    await update.message.reply_text(
        "Por ahora solo puedo entender mensajes de texto :p\n"
        "Cuentame en palabras qué estás buscando"
    )


async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    """
    Maneja el flujo principal para mensajes de texto.

    Flujo:
    1. Obtiene el ID del usuario y el contenido del mensaje.
    2. Muestra el indicador de "escribiendo..." en Telegram.
    3. Recupera historial, productos previos y estado de búsqueda.
    4. Detecta si la consulta corresponde a políticas de la tienda.
    5. Intenta resolver referencias a productos mostrados previamente.
    6. Si no existe una referencia directa, ejecuta la búsqueda híbrida.
    7. Registra el mensaje del usuario en el historial.
    8. Genera una respuesta natural mediante el LLM.
    9. Agrega un listado numerado cuando existen varios productos.
    10. Actualiza historial, productos recientes y estado de búsqueda.
    11. Envía la respuesta al usuario, dividiéndola si supera el límite
        de longitud permitido por Telegram.

    Args:
        update: Actualización recibida desde Telegram.
        context: Contexto asociado al handler de Telegram.

    Returns:
        None.
    """
    user_id = update.effective_user.id
    user_text = update.message.text.strip()

    # Indicador de "escribiendo..." mientras se procesa la consulta
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action="typing"
    )

    # Recuperar memoria conversacional del usuario
    ctx = memory.get_context(user_id)
    history = ctx["history"]
    last_products = ctx["last_products"]
    previous_query_state = ctx.get("query_state", {})

    products = None
    policy_info = None
    new_query_state = None

    # Las consultas de políticas se procesan separadamente
    # de la búsqueda de productos.
    if search.is_policy_query(user_text):
        policy_info = search.search_policies(user_text)

    else:
        # Intentar resolver referencias a productos ya mostrados,
        # por ejemplo "la segunda", "esa negra" o "la más barata".
        resolved_products = reference_resolver.resolve_reference(
            user_text,
            last_products,
        )

        if resolved_products is not None:
            products = resolved_products

            if len(products) == 1:
                memory.update_focused_product(
                    user_id,
                    products[0],
                )

        else:
            # Si no existe una referencia directa, ejecutar una nueva
            # búsqueda usando el estado conversacional anterior.
            products, new_query_state = search.search_products_with_state(
                user_text,
                previous_state=previous_query_state,
            )

    # Registrar el mensaje actual del usuario
    memory.update_history(
        user_id,
        "user",
        user_text,
    )

    # Generar respuesta natural con los productos, historial
    # y política relevante disponibles.
    response_text = llm.generate_response(
        user_message=user_text,
        products=products,
        history=history,
        policy_info=policy_info,
    )

    # Cuando hay varios productos, agregar un listado numerado generado
    # por código para preservar la correspondencia con last_products.
    if products and len(products) > 1:
        response_text = (
            f"{response_text}\n\n"
            f"{_format_product_list(products)}"
        )

    # Registrar la respuesta final enviada por el asistente
    memory.update_history(
        user_id,
        "assistant",
        response_text,
    )

    # Actualizar los últimos productos mostrados al usuario
    if products:
        memory.update_last_products(
            user_id,
            products,
        )

    # Actualizar el estado de búsqueda solo si se generó uno nuevo
    if new_query_state:
        memory.update_query_state(
            user_id,
            new_query_state,
        )

    # Logs de depuración del flujo de recuperación
    print(
        "PRODUCTOS RECUPERADOS:",
        [p.get("nombre") for p in products or []],
    )
    print(
        "QUERY_STATE:",
        new_query_state or previous_query_state,
    )

    # Telegram limita los mensajes de texto a 4096 caracteres.
    # Si la respuesta supera el límite, se envía en fragmentos.
    if len(response_text) > 4096:
        for i in range(0, len(response_text), 4096):
            await update.message.reply_text(
                response_text[i:i + 4096]
            )
    else:
        await update.message.reply_text(response_text)