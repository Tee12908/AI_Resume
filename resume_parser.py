"""
resume_parser.py
Handles text extraction from PDF, DOCX, and image resume formats.
Also extracts structured information: contact details, sections, education, experience.
"""

import re
import io
from typing import Dict, List, Optional

# ── PDF libraries ──────────────────────────────────────────────────────────────
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

# ── DOCX library ───────────────────────────────────────────────────────────────
try:
    from docx import Document
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

# ── OCR libraries ──────────────────────────────────────────────────────────────
try:
    import pytesseract
    from PIL import Image
    HAS_OCR = True
except ImportError:
    HAS_OCR = False


class ResumeParser:
    """
    Extracts and structures resume content from PDF, DOCX, and image files.
    Uses the best available library for each format with graceful fallbacks.
    """

    def extract_text(self, file_bytes: bytes, file_type: str) -> Dict:
        """
        Dispatch extraction to the correct method based on file extension.

        Returns a dict:
            text             – raw extracted text
            word_count       – number of words
            page_count       – number of pages (1 for images)
            extraction_method – which library was used
            error            – error message string, or None
        """
        result = {
            "text": "",
            "word_count": 0,
            "page_count": 0,
            "extraction_method": "",
            "error": None,
        }

        ext = file_type.lower().lstrip(".")

        try:
            if ext == "pdf":
                result = self._extract_from_pdf(file_bytes)
            elif ext in ("docx", "doc"):
                result = self._extract_from_docx(file_bytes)
            elif ext in ("png", "jpg", "jpeg", "tiff", "bmp"):
                result = self._extract_from_image(file_bytes)
            else:
                result["error"] = f"Unsupported file type: {ext}"
        except Exception as exc:
            result["error"] = str(exc)

        if result.get("text"):
            result["word_count"] = len(result["text"].split())

        return result

    # ── Private extraction methods ──────────────────────────────────────────────

    def _extract_from_pdf(self, file_bytes: bytes) -> Dict:
        """Try pdfplumber first; fall back to PyPDF2."""
        result = {
            "text": "",
            "word_count": 0,
            "page_count": 0,
            "extraction_method": "",
            "error": None,
        }

        if HAS_PDFPLUMBER:
            try:
                with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                    result["page_count"] = len(pdf.pages)
                    parts = []
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            parts.append(page_text)
                    result["text"] = "\n".join(parts)
                    result["extraction_method"] = "pdfplumber"
                    return result
            except Exception:
                pass  # fall through to PyPDF2

        if HAS_PYPDF2:
            try:
                reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
                result["page_count"] = len(reader.pages)
                parts = []
                for page in reader.pages:
                    parts.append(page.extract_text() or "")
                result["text"] = "\n".join(parts)
                result["extraction_method"] = "PyPDF2"
            except Exception as exc:
                result["error"] = f"PDF extraction failed: {exc}"
        else:
            result["error"] = "No PDF library found. Install pdfplumber or PyPDF2."

        return result

    def _extract_from_docx(self, file_bytes: bytes) -> Dict:
        """Extract text from a DOCX file including table cells."""
        result = {
            "text": "",
            "word_count": 0,
            "page_count": 1,
            "extraction_method": "python-docx",
            "error": None,
        }

        if not HAS_DOCX:
            result["error"] = "python-docx not installed. Run: pip install python-docx"
            return result

        try:
            doc = Document(io.BytesIO(file_bytes))
            parts = [p.text for p in doc.paragraphs if p.text.strip()]
            # Include table cells too
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        if cell.text.strip():
                            parts.append(cell.text)
            result["text"] = "\n".join(parts)
        except Exception as exc:
            result["error"] = f"DOCX extraction failed: {exc}"

        return result

    def _extract_from_image(self, file_bytes: bytes) -> Dict:
        """Extract text from an image using Tesseract OCR."""
        result = {
            "text": "",
            "word_count": 0,
            "page_count": 1,
            "extraction_method": "pytesseract",
            "error": None,
        }

        if not HAS_OCR:
            result["error"] = "pytesseract not installed. Run: pip install pytesseract Pillow"
            return result

        try:
            image = Image.open(io.BytesIO(file_bytes))
            result["text"] = pytesseract.image_to_string(image, config="--oem 3 --psm 6")
        except Exception as exc:
            result["error"] = f"OCR extraction failed: {exc}"

        return result

    # ── Structured extraction methods ───────────────────────────────────────────

    def extract_contact_info(self, text: str) -> Dict:
        """
        Parse contact details from raw resume text.
        Returns: email, phone, linkedin, github, website, name.
        """
        contact: Dict[str, Optional[str]] = {
            "email": None,
            "phone": None,
            "linkedin": None,
            "github": None,
            "website": None,
            "name": None,
        }

        # Email
        emails = re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text)
        if emails:
            contact["email"] = emails[0]

        # Phone – handles international/US formats
        phones = re.findall(
            r"(\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}", text
        )
        if phones:
            raw = phones[0]
            contact["phone"] = "".join(raw).strip() if isinstance(raw, tuple) else raw

        # LinkedIn
        li_match = re.search(
            r"linkedin\.com/in/([A-Za-z0-9\-_]+)", text, re.IGNORECASE
        )
        if li_match:
            contact["linkedin"] = f"linkedin.com/in/{li_match.group(1)}"

        # GitHub
        gh_match = re.search(r"github\.com/([A-Za-z0-9\-_]+)", text, re.IGNORECASE)
        if gh_match:
            contact["github"] = f"github.com/{gh_match.group(1)}"

        # Personal website (exclude common email/social domains)
        excluded = {"linkedin.com", "github.com", "gmail.com", "yahoo.com", "outlook.com"}
        web_matches = re.findall(
            r"(?:https?://)?(?:www\.)?([A-Za-z0-9\-]+\.[a-z]{2,}(?:/[^\s]*)?)", text
        )
        for site in web_matches:
            if not any(ex in site.lower() for ex in excluded):
                contact["website"] = site
                break

        # Name heuristic – first 1-4 word line near the top
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        for line in lines[:8]:
            words = line.split()
            if 2 <= len(words) <= 4 and all(w.replace(".", "").isalpha() for w in words):
                contact["name"] = line
                break

        return contact

    def extract_sections(self, text: str) -> Dict[str, bool]:
        """
        Detect which standard resume sections are present.
        Returns a dict mapping section slug → bool.
        """
        section_keywords: Dict[str, List[str]] = {
            "contact": ["contact", "contact information", "personal information"],
            "summary": ["summary", "professional summary", "career summary", "profile", "about me"],
            "objective": ["objective", "career objective", "job objective"],
            "skills": ["skills", "technical skills", "core competencies", "key skills", "expertise"],
            "education": ["education", "academic background", "qualifications", "degree"],
            "experience": [
                "experience", "work experience", "professional experience",
                "employment history", "work history", "career history",
            ],
            "projects": ["projects", "personal projects", "portfolio", "key projects"],
            "certifications": ["certifications", "certificates", "credentials", "licenses"],
            "awards": ["awards", "achievements", "honors", "recognition"],
            "languages": ["languages", "language skills"],
            "interests": ["interests", "hobbies", "activities"],
            "references": ["references", "referees"],
            "volunteer": ["volunteer", "volunteering", "community service"],
            "publications": ["publications", "research", "papers"],
        }

        text_lower = text.lower()
        sections: Dict[str, bool] = {key: False for key in section_keywords}

        for section, keywords in section_keywords.items():
            if any(kw in text_lower for kw in keywords):
                sections[section] = True

        # Contact is also true when an email/phone is present
        if re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text):
            sections["contact"] = True

        return sections

    def extract_education(self, text: str) -> List[Dict]:
        """Return institutions and degree keywords found in the text."""
        degree_patterns = [
            r"(Bachelor|B\.S\.|B\.A\.|B\.Tech|B\.E\.|BSc|BA|BS|BTech|BCA|BBA)",
            r"(Master|M\.S\.|M\.A\.|M\.Tech|MBA|MSc|MA|MS|MTech|MCA|MPhil)",
            r"(Ph\.?D|Doctorate|Doctor)",
            r"(Associate|Diploma|Certificate|High School)",
        ]
        degrees_found: List[str] = []
        for pattern in degree_patterns:
            degrees_found.extend(re.findall(pattern, text, re.IGNORECASE))

        education: List[Dict] = []
        edu_keywords = ["university", "college", "institute", "school", "academy"]
        for line in text.split("\n"):
            if any(kw in line.lower() for kw in edu_keywords):
                education.append({"institution": line.strip()})

        return education

    def extract_experience(self, text: str) -> List[Dict]:
        """Return bullet-level experience lines that start with action verbs."""
        action_verbs = [
            "managed", "developed", "created", "led", "implemented",
            "designed", "built", "launched", "increased", "reduced",
            "improved", "achieved", "collaborated", "analyzed", "maintained",
        ]
        experiences: List[Dict] = []
        for line in text.split("\n"):
            line_lower = line.lower().strip()
            if any(verb in line_lower for verb in action_verbs):
                experiences.append({"description": line.strip()})
        return experiences
