from flask import Blueprint,request,jsonify

from controllers.text_summerization_controler import text_summarization

summrization_pb=Blueprint('text',__name__)
@summrization_pb.route("/summeariz", methods=["POST"])
def summarize_text():
    try:
        data = request.get_json()
        text = data.get("text", "")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        summary = text_summarization(text)  # Call function from summarizer.py
        return jsonify({"summary": summary})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
