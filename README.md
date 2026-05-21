# AI Resume Checker

A full-stack AI-powered resume analysis web application built with Python and Streamlit.

---

## Features

| Feature | Details |
|---------|---------|
| Resume parsing | PDF, DOCX, PNG, JPG |
| ATS score | 0–100 with category breakdown |
| Skills detection | 9 categories, 150+ skills |
| Job-role prediction | 13 roles via keyword matching |
| Profile image check | Blur, resolution, face detection |
| Grammar check | language_tool_python or built-in fallback |
| Export | PDF report + CSV data |

---

## Project Structure

```
AI_Resume/
├── app.py              ← Streamlit frontend (all pages & charts)
├── resume_parser.py    ← Text extraction from PDF / DOCX / image
├── image_checker.py    ← Profile photo detection & quality analysis
├── ats_analyzer.py     ← ATS scoring, NLP, job-role prediction
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```

---

## Quick Setup (Windows)

### 1 — Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 2 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3 — Download the spaCy language model

```bash
python -m spacy download en_core_web_sm
```

### 4 — (Optional) Install Tesseract OCR for image resumes

Download the installer from: https://github.com/UB-Mannheim/tesseract/wiki

After installing, add Tesseract to PATH, or set the path in your code:
```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

### 5 — (Optional) Grammar checking

`language_tool_python` requires **Java 8+**.  
Download Java: https://adoptium.net/

Then uncomment the line in `requirements.txt`:
```
language-tool-python>=2.7.1
```
and re-run `pip install -r requirements.txt`.

### 6 — Run the app

```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

---

## Troubleshooting

| Problem | Solution |
|---------|---------|
| `ModuleNotFoundError: spacy` | `pip install spacy && python -m spacy download en_core_web_sm` |
| `ModuleNotFoundError: cv2` | `pip install opencv-python-headless` |
| PDF shows no text | Install pdfplumber: `pip install pdfplumber` |
| OCR not working | Install Tesseract and add to PATH |
| Grammar tool error | Install Java 8+ or leave the tool disabled |
| `fpdf2` missing | `pip install fpdf2` (needed for PDF export) |

---

## ATS Score Breakdown

| Category | Max Points | Description |
|----------|-----------|-------------|
| Contact Information | 15 | Email, phone, LinkedIn, name |
| Resume Sections | 20 | Skills, education, experience, summary… |
| Skills | 15 | Number of recognised skills |
| Action Verbs | 15 | Strong verbs in experience bullets |
| Quantified Results | 10 | Numbers, %, $ amounts |
| Resume Length | 10 | 400–800 words = full marks |
| Formatting | 10 | Headers, consistent dates |
| Grammar | 5 | Deducted for detected issues |

---

## Screenshots

Upload your resume → get an interactive dashboard with:
- Gauge chart for ATS score
- Pie chart for skill distribution
- Bar chart for job-role matches
- Keyword frequency chart
- Downloadable PDF and CSV report
