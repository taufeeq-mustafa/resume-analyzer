"""
Resume Analyzer
---------------
Stage 1: Parse a resume (PDF/DOCX/image) into plain text with LiteParse.
Stage 2: Send that text to Gemini 2.5 Flash and get back structured feedback.

Setup:
    pip install liteparse google-genai pydantic
   

Run:
    python resume_analyzer.py path/to/resume.pdf
    python resume_analyzer.py path/to/resume.pdf --job job_description.txt
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from liteparse import LiteParse

load_dotenv()
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


# ---------- Stage 1: parsing ----------

def parse_resume(path: str) -> str:
    """Extract plain text from a resume file. OCR is on by default,
    so image-based / scanned PDFs work too."""
    parser = LiteParse(quiet=True)
    result = parser.parse(path)
    text = result.text.strip()
    if not text:
        raise ValueError("No text extracted — the file may be empty or unreadable.")
    return text


# ---------- Structured shape we want back from the model ----------

class Analysis(BaseModel):
    overall_score: int = Field(description="Score from 0-100 for overall resume quality")
    summary: str = Field(description="2-3 sentence high-level assessment")
    strengths: list[str] = Field(description="Specific things the resume does well")
    weaknesses: list[str] = Field(description="Specific problems or gaps")
    suggestions: list[str] = Field(description="Concrete, actionable improvements")
    missing_keywords: list[str] = Field(
        description="Important skills/keywords absent but expected for the role"
    )


# ---------- Stage 2: analysis ----------

def analyze_resume(resume_text: str, job_description: str | None = None) -> Analysis:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    prompt = """You are a senior technical recruiter and resume coach with 15+ years of experience \
hiring for top-tier tech companies. You have reviewed thousands of resumes across software engineering, \
data science, AI/ML, product, and design roles.

Your job is to give the candidate brutally honest, highly specific, and immediately actionable feedback. \
You are not here to flatter — you are here to help them land the job.

RULES YOU MUST FOLLOW:
1. Never give generic advice like "tailor your resume" or "add more details." Every point must reference \
actual content, exact wording, or a specific section from the resume.
2. For weaknesses, explain WHY it is a problem from a recruiter's perspective (e.g. "This makes it \
unclear whether you led the project or just contributed to it").
3. For suggestions, be prescriptive — tell the candidate exactly what to write or change, not just what \
to do.
4. The overall_score must be calibrated strictly: 90+ is rare and reserved for near-perfect resumes, \
70–89 is solid with minor issues, 50–69 has notable gaps, below 50 has serious problems.
5. missing_keywords should only list skills/technologies that are genuinely expected for the candidate's \
apparent target role — do not pad this list.
6. If the resume has formatting issues, date inconsistencies, unexplained gaps, or red flags that a \
recruiter would notice, call them out explicitly in weaknesses.
"""

    if job_description:
        prompt += f"""
You are evaluating this resume for a SPECIFIC ROLE. Your analysis must be anchored to the job description below.
- Score the resume based on fit for THIS role, not in general.
- In weaknesses, highlight every requirement from the job description that is missing or weakly demonstrated.
- In missing_keywords, only include terms that appear in the job description but are absent from the resume.
- In suggestions, prioritize changes that directly improve fit for this role.

--- JOB DESCRIPTION ---
{job_description}

"""

    prompt += f"--- RESUME ---\n{resume_text}"

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Analysis,
        ),
    )
    # The SDK parses the JSON into your Pydantic model automatically.
    return response.parsed


# ---------- Pretty-print the result ----------

def print_report(a: Analysis) -> None:
    print(f"\n{'=' * 50}")
    print(f"  OVERALL SCORE: {a.overall_score}/100")
    print(f"{'=' * 50}\n")
    print(f"{a.summary}\n")

    def section(title, items):
        print(f"{title}:")
        for item in items:
            print(f"  - {item}")
        print()

    section("STRENGTHS", a.strengths)
    section("WEAKNESSES", a.weaknesses)
    section("SUGGESTIONS", a.suggestions)
    if a.missing_keywords:
        section("MISSING KEYWORDS", a.missing_keywords)


# ---------- Entry point ----------

def main():
    ap = argparse.ArgumentParser(description="Analyze a resume with Gemini 2.5 Flash")
    ap.add_argument("resume", help="Path to the resume file (PDF, DOCX, image...)")
    ap.add_argument("--job", help="Optional path to a job description text file")
    ap.add_argument("--parse-only", action="store_true", help="Print extracted text and exit, skip AI analysis")
    args = ap.parse_args()

    print("Parsing resume...")
    resume_text = parse_resume(args.resume)
    print(f"Extracted {len(resume_text)} characters.\n")

    if args.parse_only:
        print(resume_text)
        return

    if args.job:
        with open(args.job, encoding="utf-8") as f:
            job_desc = f.read()
    else:
        job_desc = None

    print("Sending to Gemini...")
    analysis = analyze_resume(resume_text, job_desc)
    print_report(analysis)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)