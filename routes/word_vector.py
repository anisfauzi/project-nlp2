"""Route word vector: tampilkan vektor kata & cari kata yang mirip."""

from flask import Blueprint, request, render_template, jsonify

from services.word_vector import first_token, get_vector, most_similar

word_vector_bp = Blueprint("word_vector", __name__)


@word_vector_bp.route("/word_vector")
def word_vector():
    return render_template("word_vector.html")


@word_vector_bp.route("/word_vector/lookup", methods=["POST"])
def word_vector_lookup():
    data = request.get_json(silent=True) or {}
    word = first_token(data.get("word"))
    if not word:
        return jsonify({"found": False, "message": "Silakan ketik sebuah kata."})
    try:
        vector = get_vector(word)
    except FileNotFoundError as e:
        return jsonify({"found": False, "message": str(e)})

    if vector is None:
        return jsonify({"found": False, "message": f"Kata '{word}' tidak ada di kosakata model."})

    return jsonify({"found": True, "word": word, "dim": len(vector), "vector": vector})


@word_vector_bp.route("/word_vector_search")
def word_vector_search():
    return render_template("word_vector_search.html")


@word_vector_bp.route("/word_vector_search/similar", methods=["POST"])
def word_vector_similar():
    data = request.get_json(silent=True) or {}
    word = first_token(data.get("word"))
    if not word:
        return jsonify({"found": False, "message": "Silakan ketik sebuah kata."})
    try:
        results = most_similar(word, topn=10)
    except FileNotFoundError as e:
        return jsonify({"found": False, "message": str(e)})

    if results is None:
        return jsonify({"found": False, "message": f"Kata '{word}' tidak ada di kosakata model."})

    return jsonify({"found": True, "word": word, "results": results})
