"""
ats_analyzer.py
Core AI analysis engine.

Responsibilities:
  - ATS score calculation (0-100) with category breakdown
  - Skill detection across 9 skill categories
  - Job-role prediction via keyword matching
  - Grammar checking (language_tool_python if available; simple fallback)
  - Action-verb counting
  - Quantified-achievement detection
  - Strengths / weaknesses / improvement recommendations
  - Keyword-density analysis
"""

import re
from collections import Counter
from typing import Dict, List, Optional

# ── Optional NLP backend ───────────────────────────────────────────────────────
try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    HAS_SPACY = True
except Exception:
    HAS_SPACY = False
    _nlp = None

# ── Optional grammar checker (requires Java) ───────────────────────────────────
try:
    import language_tool_python
    _grammar_tool = language_tool_python.LanguageTool("en-US")
    HAS_GRAMMAR = True
except Exception:
    HAS_GRAMMAR = False
    _grammar_tool = None


class ATSAnalyzer:
    """
    Full-spectrum ATS analysis engine.
    Call .analyze(text, sections, contact_info) to get all results at once.
    """

    # ── Skill taxonomy ─────────────────────────────────────────────────────────
    SKILLS_DB: Dict[str, List[str]] = {
        "programming_languages": [
            "python", "java", "javascript", "typescript", "c++", "c#", "ruby", "go",
            "rust", "php", "swift", "kotlin", "scala", "r", "matlab", "perl", "bash",
            "html", "css", "sql", "dart", "lua", "haskell",
        ],
        "frameworks": [
            "react", "angular", "vue", "django", "flask", "fastapi", "spring",
            "node.js", "express", "next.js", "nuxt", "gatsby", "laravel", "rails",
            "asp.net", "tensorflow", "pytorch", "keras", "scikit-learn", "pandas",
            "numpy", "scipy", "bootstrap", "tailwind", "jquery", "redux", "graphql",
        ],
        "databases": [
            "mysql", "postgresql", "mongodb", "redis", "sqlite", "oracle",
            "dynamodb", "cassandra", "elasticsearch", "firebase", "neo4j",
            "mariadb", "couchdb", "sql server",
        ],
        "cloud_devops": [
            "aws", "azure", "gcp", "google cloud", "docker", "kubernetes",
            "terraform", "ansible", "jenkins", "ci/cd", "github actions",
            "gitlab ci", "linux", "nginx", "heroku", "vercel",
        ],
        "tools": [
            "git", "github", "gitlab", "jira", "confluence", "slack", "trello",
            "postman", "figma", "excel", "power bi", "tableau", "jupyter",
            "vs code", "intellij",
        ],
        "soft_skills": [
            "leadership", "communication", "teamwork", "problem solving",
            "critical thinking", "project management", "agile", "scrum",
            "time management", "adaptability", "creativity", "analytical",
            "collaboration", "mentoring",
        ],
        "data_science": [
            "machine learning", "deep learning", "neural networks", "nlp",
            "computer vision", "data analysis", "data visualization", "statistics",
            "a/b testing", "etl", "feature engineering", "model deployment",
            "mlops", "big data", "spark", "hadoop",
        ],
        "security": [
            "cybersecurity", "penetration testing", "ethical hacking", "soc",
            "siem", "vulnerability assessment", "network security", "firewall",
            "ssl/tls", "owasp", "encryption",
        ],
        "business": [
            "business analysis", "stakeholder management", "budget management",
            "strategic planning", "risk management", "product management",
            "digital marketing", "seo", "google analytics", "crm", "salesforce", "sap",
        ],
    }

    # ── Job-role profiles ──────────────────────────────────────────────────────
    JOB_ROLES: Dict[str, List[str]] = {
        "Software Engineer": [
            "software", "developer", "engineer", "programming", "python",
            "java", "javascript", "c++", "react", "django", "api", "backend", "frontend",
        ],
        "Data Scientist / ML Engineer": [
            "machine learning", "data science", "deep learning", "neural",
            "pandas", "scikit-learn", "tensorflow", "pytorch", "statistics", "model", "nlp", "ai",
        ],
        "Data Analyst": [
            "data analysis", "excel", "tableau", "power bi", "sql",
            "visualization", "reporting", "dashboard", "metrics", "kpi", "analytics",
        ],
        "DevOps / Cloud Engineer": [
            "devops", "docker", "kubernetes", "aws", "azure", "gcp",
            "ci/cd", "terraform", "ansible", "linux", "infrastructure", "deployment",
        ],
        "Frontend Developer": [
            "html", "css", "javascript", "react", "angular", "vue", "ui",
            "ux", "responsive", "bootstrap", "tailwind", "typescript",
        ],
        "Backend Developer": [
            "api", "backend", "server", "database", "rest", "microservices",
            "django", "flask", "node", "spring", "postgresql", "mysql", "redis",
        ],
        "Full Stack Developer": [
            "full stack", "fullstack", "frontend", "backend", "react",
            "node", "angular", "vue", "django", "database", "api",
        ],
        "Cybersecurity Analyst": [
            "security", "cybersecurity", "penetration", "firewall", "vulnerability",
            "ethical hacking", "siem", "soc", "incident response", "network security",
        ],
        "Product Manager": [
            "product", "roadmap", "agile", "scrum", "stakeholder",
            "user stories", "market research", "strategy", "metrics", "requirements",
        ],
        "UI/UX Designer": [
            "design", "figma", "adobe", "wireframe", "prototype",
            "user experience", "ui", "ux", "sketch", "usability",
        ],
        "Project Manager": [
            "project management", "pmp", "agile", "scrum", "budget",
            "schedule", "risk", "stakeholder", "delivery", "planning",
        ],
        "Business Analyst": [
            "business analysis", "requirements", "stakeholder", "process improvement",
            "documentation", "sql", "reporting", "workflow", "gap analysis",
        ],
        "Mobile Developer": [
            "android", "ios", "swift", "kotlin", "flutter",
            "react native", "mobile app", "xcode", "dart",
        ],
    }

    # ── ATS-friendly action verbs ──────────────────────────────────────────────
    ACTION_VERBS: List[str] = [
        "achieved", "built", "created", "delivered", "designed", "developed",
        "enhanced", "established", "generated", "implemented", "improved",
        "increased", "launched", "led", "managed", "optimized", "oversaw",
        "produced", "reduced", "resolved", "spearheaded", "streamlined",
        "transformed", "trained", "won", "coordinated", "collaborated",
        "analyzed", "automated", "deployed", "integrated", "maintained",
        "negotiated", "mentored", "architected", "scaled", "migrated",
    ]

    # ── Main analysis entry point ──────────────────────────────────────────────

    def analyze(self, text: str, sections: Dict, contact_info: Dict) -> Dict:
        """
        Run all analyses and return a single result dict.

        Args:
            text         – full extracted resume text
            sections     – output of ResumeParser.extract_sections()
            contact_info – output of ResumeParser.extract_contact_info()
        """
        detected_skills = self._extract_skills(text)
        action_verb_info = self._count_action_verbs(text)
        quant_info = self._detect_quantified_achievements(text)
        job_role = self._predict_job_role(text)
        missing_sections = self._find_missing_sections(sections)
        present_sections = self._find_present_sections(sections)
        grammar_issues = self._check_grammar(text)
        ats_score = self._calculate_ats_score(
            text, sections, contact_info, detected_skills,
            action_verb_info, quant_info, grammar_issues
        )

        analysis = {
            "ats_score": ats_score,
            "detected_skills": detected_skills,
            "missing_sections": missing_sections,
            "present_sections": present_sections,
            "job_role": job_role,
            "grammar_issues": grammar_issues,
            "action_verb_count": action_verb_info,
            "quantified_achievements": quant_info,
            "word_count": len(text.split()),
            "character_count": len(text),
            "readability_score": self._calculate_readability(text),
            "keyword_density": self._calculate_keyword_density(text),
        }

        # Strengths / weaknesses depend on the above
        analysis["strengths"] = self._identify_strengths(text, sections, contact_info, analysis)
        analysis["weaknesses"] = self._identify_weaknesses(text, sections, contact_info, analysis)
        analysis["improvements"] = self._generate_improvements(analysis, sections)
        analysis["recommended_skills"] = self._recommend_skills(
            text, detected_skills, job_role
        )

        return analysis

    # ── ATS Score calculation ──────────────────────────────────────────────────

    def _calculate_ats_score(
        self,
        text: str,
        sections: Dict,
        contact_info: Dict,
        detected_skills: Dict,
        action_verb_info: Dict,
        quant_info: Dict,
        grammar_issues: List,
    ) -> Dict:
        """
        Score breakdown (total = 100):
            Contact info          15
            Section completeness  20
            Skills                15
            Action verbs          15
            Quantified results    10
            Resume length         10
            Formatting            10
            Grammar               5
        """
        score = 0.0
        breakdown: Dict = {}
        text_lower = text.lower()

        # ── Contact (15) ──────────────────────────────────────────────────────
        contact_score = sum([
            5 if contact_info.get("email") else 0,
            5 if contact_info.get("phone") else 0,
            3 if contact_info.get("linkedin") else 0,
            2 if contact_info.get("name") else 0,
        ])
        breakdown["Contact Info"] = {"score": contact_score, "max": 15}
        score += contact_score

        # ── Sections (20) ─────────────────────────────────────────────────────
        critical = ["skills", "education", "experience"]
        optional = ["summary", "objective", "projects", "certifications"]
        sect_score = sum(5 for s in critical if sections.get(s))
        sect_score += sum(2 for s in optional if sections.get(s))
        sect_score = min(sect_score, 20)
        breakdown["Sections"] = {"score": sect_score, "max": 20}
        score += sect_score

        # ── Skills (15) ───────────────────────────────────────────────────────
        total_skills = sum(len(v) for v in detected_skills.values())
        skill_score = min(total_skills * 1.5, 15)
        breakdown["Skills"] = {"score": round(skill_score, 1), "max": 15}
        score += skill_score

        # ── Action verbs (15) ─────────────────────────────────────────────────
        verb_count = action_verb_info.get("count", 0)
        verb_score = min(verb_count * 1.5, 15)
        breakdown["Action Verbs"] = {"score": round(verb_score, 1), "max": 15}
        score += verb_score

        # ── Quantified achievements (10) ──────────────────────────────────────
        quant_count = quant_info.get("count", 0)
        quant_score = min(quant_count * 2, 10)
        breakdown["Quantified Results"] = {"score": quant_score, "max": 10}
        score += quant_score

        # ── Resume length (10) ────────────────────────────────────────────────
        wc = len(text.split())
        if 400 <= wc <= 800:
            length_score = 10
        elif 300 <= wc < 400 or 800 < wc <= 1000:
            length_score = 7
        elif wc < 200:
            length_score = 2
        elif wc < 300:
            length_score = 4
        else:
            length_score = 5
        breakdown["Resume Length"] = {"score": length_score, "max": 10, "word_count": wc}
        score += length_score

        # ── Formatting (10) ───────────────────────────────────────────────────
        fmt_score = 0
        if re.search(r"^[A-Z][A-Z\s]{3,}$", text, re.MULTILINE):
            fmt_score += 5   # capitalised section headers
        if re.search(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b|\d{4}", text):
            fmt_score += 5   # dates present
        breakdown["Formatting"] = {"score": fmt_score, "max": 10}
        score += fmt_score

        # ── Grammar (5) ───────────────────────────────────────────────────────
        grammar_score = max(0, 5 - min(len(grammar_issues), 5))
        breakdown["Grammar"] = {"score": grammar_score, "max": 5, "issues": len(grammar_issues)}
        score += grammar_score

        total = min(round(score), 100)
        return {
            "total": total,
            "breakdown": breakdown,
            "grade": self._grade(total),
            "color": self._score_color(total),
        }

    @staticmethod
    def _grade(score: int) -> str:
        if score >= 90: return "A+"
        if score >= 80: return "A"
        if score >= 70: return "B"
        if score >= 60: return "C"
        if score >= 50: return "D"
        return "F"

    @staticmethod
    def _score_color(score: int) -> str:
        if score >= 80: return "green"
        if score >= 60: return "orange"
        return "red"

    # ── Skill extraction ───────────────────────────────────────────────────────

    def _extract_skills(self, text: str) -> Dict[str, List[str]]:
        text_lower = text.lower()
        found: Dict[str, List[str]] = {}
        for category, skills in self.SKILLS_DB.items():
            hits = []
            for skill in skills:
                if " " in skill:
                    if skill in text_lower:
                        hits.append(skill)
                else:
                    if re.search(r"\b" + re.escape(skill) + r"\b", text_lower):
                        hits.append(skill)
            if hits:
                found[category] = hits
        return found

    # ── Section helpers ────────────────────────────────────────────────────────

    def _find_missing_sections(self, sections: Dict) -> Dict:
        critical = []
        if not sections.get("contact"):
            critical.append("Contact Information")
        if not sections.get("summary") and not sections.get("objective"):
            critical.append("Professional Summary / Objective")
        if not sections.get("skills"):
            critical.append("Skills Section")
        if not sections.get("education"):
            critical.append("Education")
        if not sections.get("experience"):
            critical.append("Work Experience")

        optional = []
        if not sections.get("projects"):
            optional.append("Projects")
        if not sections.get("certifications"):
            optional.append("Certifications")

        return {"critical": critical, "optional": optional}

    def _find_present_sections(self, sections: Dict) -> List[str]:
        return [k.replace("_", " ").title() for k, v in sections.items() if v]

    # ── Job role prediction ────────────────────────────────────────────────────

    def _predict_job_role(self, text: str) -> Dict:
        text_lower = text.lower()
        scores = {}
        for role, keywords in self.JOB_ROLES.items():
            hits = [k for k in keywords if k in text_lower]
            pct = (len(hits) / len(keywords)) * 100 if keywords else 0
            scores[role] = {"score": len(hits), "match_percentage": round(pct, 1), "keywords_found": hits}

        sorted_roles = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)
        top = [{"role": r, **d} for r, d in sorted_roles[:5] if d["score"] > 0]

        primary = sorted_roles[0][0] if sorted_roles and sorted_roles[0][1]["score"] > 0 else "General Professional"
        confidence = min(sorted_roles[0][1]["match_percentage"] * 2, 100) if sorted_roles else 0

        return {"primary": primary, "top_matches": top, "confidence": round(confidence, 1)}

    # ── Grammar checking ───────────────────────────────────────────────────────

    def _check_grammar(self, text: str) -> List[Dict]:
        if HAS_GRAMMAR and _grammar_tool:
            try:
                matches = _grammar_tool.check(text[:5000])
                return [
                    {
                        "message": m.message,
                        "context": m.context,
                        "suggestions": m.replacements[:3],
                    }
                    for m in matches[:20]
                ]
            except Exception:
                pass
        return self._simple_grammar_check(text)

    @staticmethod
    def _simple_grammar_check(text: str) -> List[Dict]:
        """Lightweight grammar check without external tools."""
        issues: List[Dict] = []
        typos = {
            "teh ": "the ", "dont ": "don't ", "cant ": "can't ",
            "wont ": "won't ", "im ": "I'm ", "ive ": "I've ",
        }
        for sentence in text.split("."):
            s = sentence.strip()
            if not s:
                continue
            if "  " in s:
                issues.append({"message": "Double space found", "context": s[:60], "suggestions": []})
            sl = s.lower() + " "
            for wrong, right in typos.items():
                if wrong in sl:
                    issues.append({
                        "message": f"Possible typo: '{wrong.strip()}' → '{right.strip()}'",
                        "context": s[:60],
                        "suggestions": [right.strip()],
                    })
        return issues[:10]

    # ── Action-verb analysis ───────────────────────────────────────────────────

    def _count_action_verbs(self, text: str) -> Dict:
        text_lower = text.lower()
        found = [v for v in self.ACTION_VERBS if re.search(r"\b" + re.escape(v) + r"\b", text_lower)]
        n = len(found)
        return {
            "count": n,
            "verbs": found,
            "rating": "Excellent" if n >= 10 else "Good" if n >= 7 else "Fair" if n >= 4 else "Poor",
        }

    # ── Quantified achievements ────────────────────────────────────────────────

    def _detect_quantified_achievements(self, text: str) -> Dict:
        patterns = [
            r"\d+\s*%",
            r"\$[\d,]+",
            r"\d+\+",
            r"\d+\s*(users|customers|clients|employees|team members|engineers)",
            r"(increased|reduced|improved|grew|boosted|cut)\s+(?:by\s+)?\d+",
            r"\d+\s*(projects|applications|systems|features|products|services)",
            r"(top|ranked)\s+\d+",
        ]
        hits: List[str] = []
        text_lower = text.lower()
        for p in patterns:
            for m in re.findall(p, text_lower):
                hits.append(m if isinstance(m, str) else " ".join(m).strip())

        n = len(hits)
        return {
            "count": n,
            "examples": hits[:5],
            "rating": "Excellent" if n >= 8 else "Good" if n >= 5 else "Fair" if n >= 2 else "Poor",
        }

    # ── Readability ────────────────────────────────────────────────────────────

    def _calculate_readability(self, text: str) -> Dict:
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        words = text.split()
        avg = len(words) / len(sentences) if sentences else 0
        return {
            "avg_sentence_length": round(avg, 1),
            "total_sentences": len(sentences),
            "total_words": len(words),
            "rating": "Excellent" if avg <= 15 else "Good" if avg <= 20 else "Fair" if avg <= 25 else "Poor",
        }

    # ── Keyword density ────────────────────────────────────────────────────────

    def _calculate_keyword_density(self, text: str) -> Dict:
        STOP = {
            "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
            "her", "was", "one", "our", "out", "day", "get", "has", "him", "his",
            "how", "its", "may", "now", "old", "see", "two", "use", "way", "who",
            "have", "that", "this", "with", "they", "from", "been", "will", "more",
            "when", "than", "also", "your", "each", "both", "time", "year", "team",
            "work", "able", "into", "well", "very", "just", "most", "over", "such",
        }
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        counter = Counter(w for w in words if w not in STOP)
        top = counter.most_common(15)
        return {"top_keywords": [{"word": w, "count": c} for w, c in top]}

    # ── Strengths ──────────────────────────────────────────────────────────────

    def _identify_strengths(
        self, text: str, sections: Dict, contact_info: Dict, analysis: Dict
    ) -> List[str]:
        strengths: List[str] = []

        filled_contact = sum(1 for v in contact_info.values() if v)
        if filled_contact >= 3:
            strengths.append("Complete contact information provided")

        verbs = analysis.get("action_verb_count", {})
        if verbs.get("count", 0) >= 7:
            strengths.append(f"Strong use of action verbs ({verbs['count']} found)")

        quant = analysis.get("quantified_achievements", {})
        if quant.get("count", 0) >= 3:
            strengths.append(f"Good use of quantified achievements ({quant['count']} found)")

        total_skills = sum(len(v) for v in analysis.get("detected_skills", {}).values())
        if total_skills >= 10:
            strengths.append(f"Diverse skill set – {total_skills} skills identified")

        section_count = sum(1 for v in sections.values() if v)
        if section_count >= 5:
            strengths.append(f"Well-structured resume with {section_count} sections")

        wc = analysis.get("word_count", 0)
        if 400 <= wc <= 800:
            strengths.append("Optimal resume length (400–800 words)")

        if contact_info.get("linkedin"):
            strengths.append("LinkedIn profile included")
        if contact_info.get("github"):
            strengths.append("GitHub profile included")
        if sections.get("projects"):
            strengths.append("Projects section demonstrates practical experience")
        if sections.get("certifications"):
            strengths.append("Certifications show commitment to professional growth")

        return strengths or ["Resume content found – keep improving for best results"]

    # ── Weaknesses ─────────────────────────────────────────────────────────────

    def _identify_weaknesses(
        self, text: str, sections: Dict, contact_info: Dict, analysis: Dict
    ) -> List[str]:
        weaknesses: List[str] = []

        if not contact_info.get("email"):
            weaknesses.append("Missing email address")
        if not contact_info.get("phone"):
            weaknesses.append("Missing phone number")
        if not contact_info.get("linkedin"):
            weaknesses.append("No LinkedIn profile URL")

        for s in analysis.get("missing_sections", {}).get("critical", []):
            weaknesses.append(f"Missing critical section: {s}")

        if analysis.get("action_verb_count", {}).get("count", 0) < 4:
            weaknesses.append("Too few action verbs – add impactful verbs to each bullet")

        if analysis.get("quantified_achievements", {}).get("count", 0) < 2:
            weaknesses.append("No quantified achievements – add numbers, percentages, or dollar amounts")

        wc = analysis.get("word_count", 0)
        if wc < 300:
            weaknesses.append(f"Resume too short ({wc} words) – aim for 400–800 words")
        elif wc > 1000:
            weaknesses.append(f"Resume too long ({wc} words) – condense to 1–2 pages")

        total_skills = sum(len(v) for v in analysis.get("detected_skills", {}).values())
        if total_skills < 5:
            weaknesses.append("Few skills detected – expand the skills section explicitly")

        if len(analysis.get("grammar_issues", [])) > 5:
            weaknesses.append(
                f"Multiple grammar/spelling issues detected ({len(analysis['grammar_issues'])})"
            )

        if not sections.get("certifications"):
            weaknesses.append("No certifications listed – consider adding relevant credentials")

        return weaknesses

    # ── Improvements ───────────────────────────────────────────────────────────

    def _generate_improvements(self, analysis: Dict, sections: Dict) -> List[Dict]:
        improvements: List[Dict] = []

        total_skills = sum(len(v) for v in analysis.get("detected_skills", {}).values())
        if total_skills < 10:
            improvements.append({
                "priority": "High",
                "category": "Skills",
                "suggestion": "Expand your skills section with technical and domain-specific skills.",
            })

        if analysis.get("quantified_achievements", {}).get("count", 0) < 3:
            improvements.append({
                "priority": "High",
                "category": "Impact",
                "suggestion": (
                    "Quantify results: 'Increased sales by 30%', "
                    "'Managed team of 10', 'Cut deployment time by 40%'."
                ),
            })

        if not sections.get("summary") and not sections.get("objective"):
            improvements.append({
                "priority": "High",
                "category": "Summary",
                "suggestion": "Add a 2–3 line professional summary at the top of your resume.",
            })

        improvements.append({
            "priority": "Medium",
            "category": "ATS Keywords",
            "suggestion": "Copy exact keywords from the job description into your resume.",
        })

        if not sections.get("projects"):
            improvements.append({
                "priority": "Medium",
                "category": "Projects",
                "suggestion": "Add a Projects section to demonstrate hands-on experience.",
            })

        if analysis.get("action_verb_count", {}).get("count", 0) < 7:
            improvements.append({
                "priority": "Medium",
                "category": "Language",
                "suggestion": (
                    "Begin each bullet with a strong verb: "
                    "Developed, Implemented, Led, Achieved, Architected."
                ),
            })

        improvements.append({
            "priority": "Low",
            "category": "Formatting",
            "suggestion": "Use consistent date format (e.g., Jan 2022 – Mar 2024) throughout.",
        })

        improvements.append({
            "priority": "Low",
            "category": "Online Presence",
            "suggestion": "Include your LinkedIn and GitHub URLs if not already present.",
        })

        return improvements

    # ── Skill recommendations ──────────────────────────────────────────────────

    def _recommend_skills(
        self, text: str, detected_skills: Dict, job_role: Dict
    ) -> List[str]:
        text_lower = text.lower()
        role = job_role.get("primary", "")

        role_skill_map: Dict[str, List[str]] = {
            "Software Engineer": ["docker", "kubernetes", "aws", "rest api", "git", "agile", "testing"],
            "Data Scientist / ML Engineer": ["tensorflow", "pytorch", "mlops", "spark", "sql", "statistics"],
            "Data Analyst": ["tableau", "power bi", "excel", "sql", "python", "r", "statistics"],
            "DevOps / Cloud Engineer": ["kubernetes", "terraform", "ansible", "aws", "azure", "linux"],
            "Frontend Developer": ["react", "typescript", "css", "testing", "webpack", "accessibility"],
            "Backend Developer": ["microservices", "docker", "redis", "postgresql", "testing"],
            "Full Stack Developer": ["react", "node.js", "postgresql", "docker", "aws", "typescript"],
            "Cybersecurity Analyst": ["siem", "soc", "compliance", "cloud security", "network security"],
            "Product Manager": ["jira", "user research", "a/b testing", "data analysis", "sql"],
            "UI/UX Designer": ["figma", "user research", "prototyping", "accessibility", "usability testing"],
        }

        recs: List[str] = []
        target_skills = role_skill_map.get(role, [])
        for skill in target_skills:
            if skill not in text_lower:
                recs.append(skill)

        general = ["git", "agile", "communication", "problem solving", "documentation"]
        for skill in general:
            if skill not in text_lower and skill not in recs:
                recs.append(skill)

        return recs[:10]
