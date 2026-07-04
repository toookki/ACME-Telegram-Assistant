"""
modules/memory.py
=================
Memoria conversacional por usuario.

Guarda:
- history: historial reciente de mensajes
- last_products: últimos productos mostrados
- preferences: preferencias implícitas del usuario
- query_state: última intención y filtros de búsqueda
- focused_product: producto discutido de forma puntual
"""

from config.settings import MAX_HISTORIAL_TURNOS


# Almacén principal:
# {user_id: {history, last_products, preferences, query_state, focused_product}}
_memory: dict = {}


def _init_user(user_id: int):
    """
    Inicializa la memoria de un usuario si no existe.

    Args:
        user_id: ID del usuario en Telegram.
    """
    if user_id not in _memory:
        _memory[user_id] = {
            "history": [],
            "last_products": [],
            "preferences": {},
            "query_state": {},
            "focused_product": None,
        }


def get_context(user_id: int) -> dict:
    """
    Retorna el contexto completo de un usuario.

    Args:
        user_id: ID del usuario en Telegram.

    Returns:
        Diccionario con toda la memoria asociada al usuario.
    """
    _init_user(user_id)
    return _memory[user_id]


def update_history(user_id: int, role: str, content: str):
    """
    Agrega un mensaje al historial del usuario.
    Mantiene solo los últimos MAX_HISTORIAL_TURNOS turnos
    (2 mensajes por turno).

    Args:
        user_id: ID del usuario en Telegram.
        role: Rol del mensaje, normalmente "user" o "assistant".
        content: Contenido textual del mensaje.
    """
    _init_user(user_id)
    _memory[user_id]["history"].append({
        "role": role,
        "content": content,
    })

    # Limitar tamaño del historial
    max_msgs = MAX_HISTORIAL_TURNOS * 2
    if len(_memory[user_id]["history"]) > max_msgs:
        _memory[user_id]["history"] = (
            _memory[user_id]["history"][-max_msgs:]
        )


def update_last_products(user_id: int, products: list):
    """
    Actualiza los últimos productos mostrados al usuario.
    Se usan para resolver referencias como "la segunda" o "esa negra".

    Si se muestra un único producto, también se establece como
    producto enfocado.

    Args:
        user_id: ID del usuario en Telegram.
        products: Lista de diccionarios de productos mostrados.
    """
    _init_user(user_id)
    _memory[user_id]["last_products"] = products

    if len(products) == 1:
        _memory[user_id]["focused_product"] = products[0]


def get_last_products(user_id: int) -> list:
    """
    Retorna los últimos productos mostrados al usuario.

    Args:
        user_id: ID del usuario en Telegram.

    Returns:
        Lista de diccionarios de productos mostrados recientemente.
    """
    _init_user(user_id)
    return _memory[user_id]["last_products"]


def update_focused_product(user_id: int, product: dict | None):
    """
    Actualiza el producto discutido de forma puntual.

    Args:
        user_id: ID del usuario en Telegram.
        product: Diccionario del producto enfocado o None para eliminar el foco.
    """
    _init_user(user_id)
    _memory[user_id]["focused_product"] = product


def get_focused_product(user_id: int):
    """
    Retorna el producto discutido de forma puntual.

    Args:
        user_id: ID del usuario en Telegram.

    Returns:
        Diccionario del producto enfocado o None si no existe uno.
    """
    _init_user(user_id)
    return _memory[user_id].get("focused_product")


def update_query_state(user_id: int, query_state: dict):
    """
    Actualiza el estado de la última búsqueda del usuario.

    Args:
        user_id: ID del usuario en Telegram.
        query_state: Diccionario con la intención y los filtros
            asociados a la búsqueda actual.
    """
    _init_user(user_id)
    _memory[user_id]["query_state"] = query_state or {}


def get_query_state(user_id: int) -> dict:
    """
    Retorna el estado de la última búsqueda del usuario.

    Args:
        user_id: ID del usuario en Telegram.

    Returns:
        Diccionario con la intención y los filtros de la búsqueda actual.
    """
    _init_user(user_id)
    return _memory[user_id].get("query_state", {})


def update_preferences(user_id: int, preferences: dict):
    """
    Actualiza las preferencias implícitas del usuario.
    Las nuevas preferencias se combinan con las existentes.

    Args:
        user_id: ID del usuario en Telegram.
        preferences: Diccionario de preferencias nuevas,
            por ejemplo {"color": "negro", "precio_max": 80000}.
    """
    _init_user(user_id)
    _memory[user_id]["preferences"].update(preferences)


def clear_user(user_id: int):
    """
    Elimina toda la memoria asociada a un usuario.
    Es útil, por ejemplo, al ejecutar el comando /start.

    Args:
        user_id: ID del usuario en Telegram.
    """
    if user_id in _memory:
        del _memory[user_id]