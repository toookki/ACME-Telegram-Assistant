"""
main.py
=======
Punto de entrada del bot de Telegram ACME.

Responsabilidades:
- Pre-cargar los embeddings al inicio (para que el primer usuario no espere).
- Registrar los handlers de Telegram.
- Iniciar el bot en modo polling.

No contiene lógica de negocio. Solo orquesta.
"""

import logging
from telegram.ext import Application, CommandHandler, MessageHandler, filters

from config.settings import TELEGRAM_TOKEN
from modules.embeddings import load_embeddings
from modules.handler import handle_start, handle_reset, handle_message

# Configurar logging para ver errores y eventos en consola
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    """Función principal: inicializa y corre el bot."""

    print("Iniciando ACME Bot...")

    # Pre-cargar embeddings al inicio para que el bot responda rápido desde el primer mensaje.
    # Esto puede tardar ~1 minuto la primera vez (descarga el modelo y calcula embeddings).
    # Las veces siguientes carga desde cache en pocos segundos.
    print("Pre-cargando embeddings de productos...")
    load_embeddings()
    print("Embeddings listos. Bot listo para recibir mensajes.")

    # Crear la aplicación de Telegram con el token
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # Registrar handlers de comandos
    app.add_handler(CommandHandler("start", handle_start))
    app.add_handler(CommandHandler("reset", handle_reset))

    # Registrar handler de mensajes de texto (captura todo texto que no sea comando)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot corriendo. Presiona Ctrl+C para detener.")
    print("Bot disponible en Telegram como @acme_die_bot")

    # Iniciar el bot en modo polling (el bot pregunta a Telegram si hay mensajes nuevos)
    app.run_polling(allowed_updates=["message"])


if __name__ == "__main__":
    main()
