"""
main.py
=======
Punto de entrada del bot de Telegram ACME.

Responsabilidades:
- Pre-cargar los embeddings al inicio para evitar latencia
  en la primera consulta.
- Crear y configurar la aplicación de Telegram.
- Registrar los handlers de comandos y mensajes.
- Iniciar el bot en modo polling.

No contiene lógica de negocio. Solo coordina la inicialización
y ejecución general de la aplicación.
"""

import logging

from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
)

from config.settings import TELEGRAM_TOKEN
from modules.embeddings import load_embeddings
from modules.handler import (
    handle_start,
    handle_reset,
    handle_message,
    handle_help,
    handle_unsupported,
)


# Configurar logging para visualizar errores y eventos en consola.
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


def main():
    """
    Inicializa y ejecuta el bot de Telegram ACME.

    Flujo:
    1. Pre-carga los embeddings de productos.
    2. Crea la aplicación de Telegram usando el token configurado.
    3. Registra los handlers de comandos.
    4. Registra los handlers de mensajes de texto y formatos no soportados.
    5. Inicia el bot en modo polling.

    Returns:
        None.
    """
    print("Iniciando ACME Bot...")

    # Pre-cargar embeddings al inicio para que el bot responda
    # rápidamente desde el primer mensaje.
    #
    # La primera ejecución puede tardar más si es necesario descargar
    # el modelo y generar los embeddings. En ejecuciones posteriores,
    # normalmente se reutiliza la caché almacenada en disco.
    print("Pre-cargando embeddings de productos...")
    load_embeddings()
    print("Embeddings listos. Bot listo para recibir mensajes.")

    # Crear la aplicación de Telegram usando el token configurado.
    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .build()
    )

    # Registrar handlers de comandos.
    app.add_handler(
        CommandHandler("start", handle_start)
    )
    app.add_handler(
        CommandHandler("reset", handle_reset)
    )
    app.add_handler(
        CommandHandler("help", handle_help)
    )

    # Registrar handler para mensajes de texto que no sean comandos.
    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_message,
        )
    )

    # Registrar handler para mensajes no soportados,
    # como fotos, stickers, audios o documentos.
    app.add_handler(
        MessageHandler(
            ~filters.TEXT & ~filters.COMMAND,
            handle_unsupported,
        )
    )

    print("Bot corriendo. Presiona Ctrl+C para detener.")
    print("Bot disponible en Telegram como @acme_die_bot")

    # Iniciar el bot en modo polling.
    # El cliente consulta periódicamente a Telegram por nuevos mensajes.
    app.run_polling(
        allowed_updates=["message"]
    )


if __name__ == "__main__":
    main()