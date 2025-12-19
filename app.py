from flask import Flask, request, render_template_string
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import json
import uuid

# Import OCR class
from OCR_pass import PassportOCRExtractor

# -------------------------------
# Load env
# -------------------------------
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# -------------------------------
# Flask app
# -------------------------------
app = Flask(__name__)

# -------------------------------
# Init OCR + Gemini once
# -------------------------------
ocr_extractor = PassportOCRExtractor()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    max_output_tokens=512
)

# -------------------------------
# Simple HTML UI (inline, no templates folder)
# -------------------------------
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Passport OCR + Gemini</title>
</head>
<body style="font-family: Arial; margin: 40px;">
    <h2>Upload Passport Image</h2>

    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="file" required />
        <br><br>
        <button type="submit">Upload & Extract</button>
    </form>

    {% if result %}
        <hr>
        <h3>Extracted OCR Text</h3>
        <pre>{{ result.extracted_text }}</pre>

        <h3>Structured Passport Data</h3>
        <pre>{{ result.passport_data }}</pre>

        <h3>Accuracy</h3>
        <pre>{{ result.metrics }}</pre>
    {% endif %}
</body>
</html>
"""

# -------------------------------
# UI + Upload handler
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def upload_and_extract():

    if request.method == "POST":
        file = request.files.get("file")

        if not file:
            return "No file uploaded", 400

        temp_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
        file.save(temp_path)

        try:
            # OCR
            passport_text = ocr_extractor.extract_text(temp_path)

            # Gemini prompt (UNCHANGED LOGIC)
            prompt = f"""
You are a specialized Indian Passport Data Extraction agent.

Return ONLY valid JSON.
Use MRZ as ground truth.
Use null for missing fields.
Format dates as YYYY-MM-DD.

JSON SCHEMA:
{{
  "country": "INDIA",
  "passport_type": "P",
  "nationality": "INDIAN",
  "passport_number": "Verify against MRZ",
  "surname": "From VIZ",
  "given_names": "From VIZ",
  "date_of_birth": "YYYY-MM-DD",
  "sex": "M/F",
  "place_of_birth": null,
  "place_of_issue": null,
  "date_of_issue": "YYYY-MM-DD",
  "date_of_expiry": "YYYY-MM-DD",
  "father_name": null,
  "mother_name": null,
  "spouse_name": null,
  "address": null,
  "pin_code": null,
  "file_number": null,
  "holder_signature_present": true/false
}}

OCR TEXT:
\"\"\"{passport_text}\"\"\"
"""

            result = model.invoke(prompt)
            raw_output = result.content.strip()

            if raw_output.startswith("```"):
                raw_output = raw_output.replace("```json", "").replace("```", "").strip()

            passport_data = json.loads(raw_output)

            # Accuracy
            total_fields = len(passport_data)
            null_fields = sum(1 for v in passport_data.values() if v is None)
            extracted_fields = total_fields - null_fields
            accuracy = round((extracted_fields / total_fields) * 100, 2)

            return render_template_string(
                HTML_PAGE,
                result={
                    "extracted_text": passport_text,
                    "passport_data": json.dumps(passport_data, indent=2),
                    "metrics": {
                        "total_fields": total_fields,
                        "null_fields": null_fields,
                        "extracted_fields": extracted_fields,
                        "accuracy_percent": accuracy
                    }
                }
            )

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return render_template_string(HTML_PAGE)

# -------------------------------
# Run app
# -------------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
