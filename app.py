import os
import sys

# Paksa mode UTF-8 di Windows (hindari UnicodeDecodeError saat impor library ML).
# Harus dijalankan SEBELUM mengimpor routes/services yang memuat transformers, dll.
if sys.platform == "win32" and not sys.flags.utf8_mode:
    os.environ["PYTHONUTF8"] = "1"
    os.execv(sys.executable, [sys.executable, "-X", "utf8", *sys.argv])

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from flask import Flask

from routes import ALL_BLUEPRINTS

app = Flask(__name__)

# Daftarkan semua blueprint (routing dipisah rapi per fitur di folder routes/)
for blueprint in ALL_BLUEPRINTS:
    app.register_blueprint(blueprint)


if __name__ == "__main__":
    # use_reloader=False agar model tidak dimuat dua kali (boros RAM) di mode debug.
    app.run(debug=True, use_reloader=False)
