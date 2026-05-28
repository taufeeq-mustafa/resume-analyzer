import os
import tempfile

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

from resume_analyzer import analyze_resume, parse_resume

load_dotenv()

app = Flask(__name__)
CORS(app)


@app.route("/analyze", methods=["POST"])
def analyze():
    if "resume" not in request.files:
        return jsonify({"error": "No resume file uploaded"}), 400

    file = request.files["resume"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    job_desc = request.form.get("job_description", "").strip() or None

    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        resume_text = parse_resume(tmp_path)
        analysis = analyze_resume(resume_text, job_desc)
        return jsonify(analysis.model_dump())
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
