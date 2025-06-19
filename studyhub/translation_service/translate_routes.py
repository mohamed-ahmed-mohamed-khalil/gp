from flask import Blueprint, request, jsonify
from controllers.translate_controller import translate_text

translate_bp = Blueprint("translate", __name__)  # ✅ Correct Blueprint name

@translate_bp.route("/", methods=["POST"])  # ✅ Now accessible at /translate
def translate():
    # Handle JSON parsing errors
    try:
        data = request.get_json()
        if data is None:
            raise ValueError("Invalid JSON")
    except Exception:
        return jsonify({"error": "Invalid JSON format"}), 400

    # Extract data safely
    text = data.get("text", "").strip()
    src_lang = data.get("src_lang", "").strip()
    tgt_lang = data.get("tgt_lang", "").strip()

    # Validate input fields
    if not text or not src_lang or not tgt_lang:
        return jsonify({"error": "Missing required fields"}), 400

    # Perform translation with error handling
    try:
        translation = translate_text(text, src_lang, tgt_lang)
        return jsonify({"translation": translation})
    except Exception as e:
        return jsonify({"error": f"Translation failed: {str(e)}"}), 500
