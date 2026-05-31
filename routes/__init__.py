"""Kumpulan blueprint route. Diimpor app.py via: from routes import ALL_BLUEPRINTS"""

from .main import main_bp
from .chatbot import chatbot_bp
from .word_vector import word_vector_bp
from .weather import weather_bp

# Daftar semua blueprint untuk didaftarkan ke aplikasi Flask
ALL_BLUEPRINTS = [
    main_bp,
    chatbot_bp,
    word_vector_bp,
    weather_bp,
]
