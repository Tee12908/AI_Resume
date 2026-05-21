"""
image_checker.py
Detects embedded profile photos in resumes and evaluates their quality.
Works with PDF, DOCX, and standalone image files.
Uses OpenCV for blur detection and face detection (Haar cascade).
"""

import io
from typing import Dict, List

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import numpy as np
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False


class ImageChecker:
    """
    Detects and evaluates profile photos embedded in resumes.
    Checks: presence, face detection, blur, resolution, and overall quality.
    """

    # Minimum recommended dimension for a profile photo
    MIN_PHOTO_DIM = 200
    # Laplacian variance threshold – below this is considered blurry
    BLUR_THRESHOLD = 80.0
    # Skip images smaller than this (likely decorative icons)
    MIN_RELEVANT_DIM = 60

    def __init__(self) -> None:
        self._face_cascade = None
        if HAS_OPENCV:
            try:
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                self._face_cascade = cv2.CascadeClassifier(cascade_path)
            except Exception:
                self._face_cascade = None

    # ── Public API ──────────────────────────────────────────────────────────────

    def check_resume_for_image(self, file_bytes: bytes, file_type: str) -> Dict:
        """
        Main entry point.  Returns a comprehensive image-analysis dict.

        Keys:
            has_image       – bool
            image_count     – int
            has_face        – bool
            face_count      – int
            quality         – quality sub-dict (see _analyze_image_quality)
            recommendations – list of actionable strings
        """
        result: Dict = {
            "has_image": False,
            "image_count": 0,
            "has_face": False,
            "face_count": 0,
            "quality": {},
            "recommendations": [],
        }

        ext = file_type.lower().lstrip(".")

        if ext == "pdf":
            images = self._extract_images_from_pdf(file_bytes)
        elif ext in ("docx", "doc"):
            images = self._extract_images_from_docx(file_bytes)
        elif ext in ("png", "jpg", "jpeg", "tiff", "bmp"):
            images = self._load_standalone_image(file_bytes)
        else:
            return result

        if not images:
            return result

        result["has_image"] = True
        result["image_count"] = len(images)

        # Analyse the largest image (most likely the profile photo)
        largest = max(images, key=lambda img: img.width * img.height)
        result["quality"] = self._analyze_image_quality(largest)

        if self._face_cascade is not None:
            face_info = self._detect_face(largest)
            result["has_face"] = face_info["has_face"]
            result["face_count"] = face_info["face_count"]

        result["recommendations"] = self._build_recommendations(result)
        return result

    # ── Image extraction helpers ────────────────────────────────────────────────

    def _load_standalone_image(self, file_bytes: bytes) -> List["Image.Image"]:
        if not HAS_PIL:
            return []
        try:
            return [Image.open(io.BytesIO(file_bytes)).convert("RGB")]
        except Exception:
            return []

    def _extract_images_from_pdf(self, file_bytes: bytes) -> List["Image.Image"]:
        """Extract embedded raster images from a PDF using PyPDF2."""
        images: List["Image.Image"] = []
        if not (HAS_PYPDF2 and HAS_PIL):
            return images

        try:
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            for page in reader.pages:
                try:
                    resources = page.get("/Resources", {})
                    if hasattr(resources, "get_object"):
                        resources = resources.get_object()
                    x_objects = resources.get("/XObject", {})
                    if hasattr(x_objects, "get_object"):
                        x_objects = x_objects.get_object()

                    for name in x_objects:
                        obj = x_objects[name]
                        if hasattr(obj, "get_object"):
                            obj = obj.get_object()
                        if obj.get("/Subtype") != "/Image":
                            continue

                        width = int(obj.get("/Width", 0))
                        height = int(obj.get("/Height", 0))
                        if width < self.MIN_RELEVANT_DIM or height < self.MIN_RELEVANT_DIM:
                            continue

                        img = self._pdf_obj_to_pil(obj, width, height)
                        if img:
                            images.append(img)
                except Exception:
                    continue
        except Exception:
            pass

        return images

    def _pdf_obj_to_pil(self, obj, width: int, height: int) -> "Optional[Image.Image]":
        """Convert a PyPDF2 image XObject to a PIL Image."""
        if not HAS_PIL:
            return None
        try:
            data = obj.get_data()
            filter_type = obj.get("/Filter", "")

            if filter_type == "/DCTDecode":
                return Image.open(io.BytesIO(data)).convert("RGB")

            if filter_type in ("/FlateDecode", ""):
                color_space = str(obj.get("/ColorSpace", "/DeviceRGB"))
                if "Gray" in color_space or color_space == "/DeviceGray":
                    mode, channels = "L", 1
                else:
                    mode, channels = "RGB", 3

                expected = width * height * channels
                if len(data) == expected:
                    return Image.frombytes(mode, (width, height), data)
        except Exception:
            pass
        return None

    def _extract_images_from_docx(self, file_bytes: bytes) -> List["Image.Image"]:
        """Extract all embedded images from a DOCX file."""
        images: List["Image.Image"] = []
        if not HAS_PIL:
            return images
        try:
            from docx import Document
            doc = Document(io.BytesIO(file_bytes))
            for rel in doc.part.rels.values():
                if "image" in rel.target_ref:
                    try:
                        blob = rel.target_part.blob
                        img = Image.open(io.BytesIO(blob)).convert("RGB")
                        if img.width >= self.MIN_RELEVANT_DIM and img.height >= self.MIN_RELEVANT_DIM:
                            images.append(img)
                    except Exception:
                        continue
        except Exception:
            pass
        return images

    # ── Quality analysis ────────────────────────────────────────────────────────

    def _analyze_image_quality(self, image: "Image.Image") -> Dict:
        """
        Returns quality metrics for a PIL Image:
            width, height, resolution, is_low_resolution,
            is_blurry, blur_score, is_grayscale,
            size_rating, overall_quality
        """
        quality: Dict = {
            "width": image.width,
            "height": image.height,
            "resolution": f"{image.width}x{image.height}",
            "is_low_resolution": False,
            "is_blurry": False,
            "blur_score": 0.0,
            "is_grayscale": False,
            "size_rating": "Low",
            "overall_quality": "Unknown",
        }

        # Resolution check
        if image.width < self.MIN_PHOTO_DIM or image.height < self.MIN_PHOTO_DIM:
            quality["is_low_resolution"] = True

        # Size rating
        if image.width >= 400 and image.height >= 400:
            quality["size_rating"] = "High"
        elif image.width >= self.MIN_PHOTO_DIM and image.height >= self.MIN_PHOTO_DIM:
            quality["size_rating"] = "Medium"

        # Grayscale check
        if image.mode in ("L", "LA"):
            quality["is_grayscale"] = True
        elif HAS_OPENCV:
            try:
                arr = np.array(image)
                if arr.ndim == 3:
                    quality["is_grayscale"] = (
                        np.allclose(arr[:, :, 0], arr[:, :, 1], atol=5)
                        and np.allclose(arr[:, :, 1], arr[:, :, 2], atol=5)
                    )
            except Exception:
                pass

        # Blur via Laplacian variance
        if HAS_OPENCV:
            try:
                cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
                blur_score = float(cv2.Laplacian(cv_img, cv2.CV_64F).var())
                quality["blur_score"] = round(blur_score, 2)
                quality["is_blurry"] = blur_score < self.BLUR_THRESHOLD
            except Exception:
                pass

        # Overall quality label
        if quality["is_low_resolution"] or quality["is_blurry"]:
            quality["overall_quality"] = "Poor"
        elif quality["size_rating"] == "Medium":
            quality["overall_quality"] = "Fair"
        elif quality["size_rating"] == "High":
            quality["overall_quality"] = "Good"
        else:
            quality["overall_quality"] = "Poor"

        return quality

    def _detect_face(self, image: "Image.Image") -> Dict:
        """Run OpenCV Haar cascade face detection on a PIL Image."""
        result = {"has_face": False, "face_count": 0}
        if not HAS_OPENCV or self._face_cascade is None:
            return result
        try:
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            faces = self._face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            if len(faces) > 0:
                result["has_face"] = True
                result["face_count"] = int(len(faces))
        except Exception:
            pass
        return result

    # ── Recommendations ─────────────────────────────────────────────────────────

    def _build_recommendations(self, result: Dict) -> List[str]:
        recs: List[str] = []
        quality = result.get("quality", {})

        if not result["has_image"]:
            return recs

        if result["has_face"]:
            recs.append("Profile photo detected — make sure it looks professional.")
            if quality.get("is_blurry"):
                recs.append("Photo appears blurry. Replace with a sharper, high-quality image.")
            if quality.get("is_low_resolution"):
                recs.append("Photo resolution is low. Use at least 400×400 px for best results.")
            if quality.get("is_grayscale"):
                recs.append("Using a colour photo gives a more modern, polished impression.")
            if quality.get("overall_quality") == "Good":
                recs.append("Photo quality looks great!")
            elif quality.get("overall_quality") == "Poor":
                recs.append("Overall photo quality is poor — consider a professional headshot.")
        else:
            recs.append(
                "An image was found but no face was detected. "
                "Ensure the photo is a clear front-facing headshot."
            )

        return recs
