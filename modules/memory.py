"""
modules/memory.py
=================
Memoria conversacional por usuario.

Almacena en un diccionario en RAM (se pierde al reiniciar el bot).

Por cada user_id guarda:
  - history: lista de mensajes recientes [{role, content}]
  - last_products: últimos productos mostrados al usuario
  - preferences: preferencias implícitas detectadas (dict libre)

Esto permite resolver referencias contextuales como:
  "la segunda" corresponde a products[1]
  "esa negra" corresponde a filtrar last_products por color negro
  "algo más barato" corresponde a filtrar last_products por precio menor
"""

from config.settings import MAX_HISTORIAL_TURNOS

# Almacén principal: {user_id: {history, last_products, preferences}}
_memory: dict = {}


def _init_user(user_id: int):
    """Inicializa la memoria de un usuario si no existe."""
    if user_id not in _memory:
        _memory[user_id] = {
            "history": [],
            "last_products": [],
            "preferences": {},
        }


def get_context(user_id: int) -> dict:
    """
    Retorna el contexto completo de un usuario.

    Returns:
        dict con:
          - history: lista de {role, content}
          - last_products: lista de productos (dicts)
          - preferences: dict de preferencias implícitas
    """
    _init_user(user_id)
    return _memory[user_id]


def update_history(user_id: int, role: str, content: str):
    """
    Agrega un mensaje al historial del usuario.
    Mantiene solo los últimos MAX_HISTORIAL_TURNOS turnos (2 msgs por turno).

    Args:
        user_id: ID del usuario en Telegram.
        role: "user" o "assistant".
        content: Texto del mensaje.
    """
    _init_user(user_id)
    _memory[user_id]["history"].append({"role": role, "content": content})
    # Limitar tamaño del historial
    max_msgs = MAX_HISTORIAL_TURNOS * 2
    if len(_memory[user_id]["history"]) > max_msgs:
        _memory[user_id]["history"] = _memory[user_id]["history"][-max_msgs:]


def update_last_products(user_id: int, products: list):
    """
    Actualiza los últimos productos mostrados al usuario.
    Se usan para resolver referencias como "la segunda", "esa negra".

    Args:
        user_id: ID del usuario en Telegram.
        products: Lista de dicts de productos.
    """
    _init_user(user_id)
    _memory[user_id]["last_products"] = products


def update_preferences(user_id: int, preferences: dict):
    """
    Actualiza las preferencias implícitas del usuario.
    Se hace merge con las preferencias existentes.

    Args:
        user_id: ID del usuario.
        preferences: dict de preferencias nuevas (ej: {"color": "negro", "precio_max": 80000}).
    """
    _init_user(user_id)
    _memory[user_id]["preferences"].update(preferences)


def clear_user(user_id: int):
    """Limpia toda la memoria de un usuario (útil para el comando /start)."""
    if user_id in _memory:
        del _memory[user_id]


def get_last_products(user_id: int) -> list:
    """Shortcut para obtener los últimos productos mostrados."""
    _init_user(user_id)
    return _memory[user_id]["last_products"]
