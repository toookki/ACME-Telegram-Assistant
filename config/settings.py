"""
config/settings.py
==================
Carga variables de entorno desde el archivo .env y las expone
como constantes globales para el resto del proyecto.

Uso:
    from config.settings import TELEGRAM_TOKEN, GEMINI_API_KEY
"""

import os
from dotenv import load_dotenv

# Carga el archivo .env automáticamente al importar este módulo
load_dotenv()

# Token del bot de Telegram
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

# API Key de Google Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Rutas a los archivos de datos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

PRODUCTOS_PATH = os.path.join(DATA_DIR, "productos.json")
POLITICAS_PATH = os.path.join(DATA_DIR, "politicas.json")
EMBEDDINGS_CACHE_PATH = os.path.join(DATA_DIR, "embeddings_cache.pkl")

# Parámetros del sistema
MAX_PRODUCTOS_RESULTADO = 5     # Máximo de productos a entregar al LLM
MAX_HISTORIAL_TURNOS = 6        # Turnos recientes a mantener en memoria por usuario
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # Soporte multilingüe (español)

# Validación básica al importar
if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN no está definido en el .env")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY no está definido en el .env")
