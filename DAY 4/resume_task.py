import json
import PyPDF2
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

MODEL = "gemini-2.5-flash"
TEMPERATURE = 0.0

model = genai.GenerativeModel(MODEL)


def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text.strip()


def build_prompt(resume_text):
    return f"""
You are an AI Resume Analyzer.

TASK:
Extract structured data from the resume.

STRICT RULES:
1. Use ONLY information present in the resume
2. DO NOT assume anything
3. If Python is NOT mentioned → python_fit = "NO"
4. Output ONLY JSON
5. No explanation, no extra text
6. No guessing

OUTPUT FORMAT:
{{
  "name": "string",
  "skills": ["list"],
  "experience_years": number,
  "python_fit": "YES" or "NO"
}}

FEW-SHOT EXAMPLES:

INPUT:
Name: Amit
Experience: 3 years in Python and Django
Projects: Built APIs
Education: B.Tech

OUTPUT:
{{
  "name": "Amit",
  "skills": ["Python", "Django", "APIs"],
  "experience_years": 3,
  "python_fit": "YES"
}}

INPUT:
Name: Neha
Experience: 2 years in Java backend
Projects: Banking app
Education: B.Tech

OUTPUT:
{{
  "name": "Neha",
  "skills": ["Java", "Backend"],
  "experience_years": 2,
  "python_fit": "NO"
}}

INPUT:
Name: Rahul
Experience: 1 year in SQL and Data Analysis
Projects: Dashboard project
Education: BSc

OUTPUT:
{{
  "name": "Rahul",
  "skills": ["SQL", "Data Analysis"],
  "experience_years": 1,
  "python_fit": "NO"
}}

INPUT:
{resume_text}

OUTPUT:
"""


def analyze_resume(resume_text):
    prompt = build_prompt(resume_text)

    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(temperature=TEMPERATURE)
    )

    output = response.text.strip()

    try:
        return json.loads(output)
    except:
        return {
            "error": "Invalid JSON output",
            "raw_output": output
        }


if __name__ == "__main__":
    pdf_path = "sample_resume.pdf"

    resume_text = extract_text_from_pdf(pdf_path)

    print("\n--- Extracted Resume Text ---\n")
    print(resume_text)

    result = analyze_resume(resume_text)

    print("\n--- FINAL OUTPUT ---\n")
    print(json.dumps(result, indent=2))