import re

from pypdf import PdfReader


# ============================================================
# EXTRACT PDF PAGES
# ============================================================

def extract_pages_from_pdf(pdf_file):

    reader = PdfReader(pdf_file)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):

        try:

            text = page.extract_text() or ""

        except Exception:

            text = ""

        pages.append(
            {
                "page": page_number,
                "text": text.strip(),
            }
        )

    return pages


# ============================================================
# EXTRACT TEXT FROM PDF
# ============================================================

def extract_text_from_pdf(pdf_path):

    reader = PdfReader(pdf_path)

    text = ""

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):

        try:

            page_text = (
                page.extract_text()
                or ""
            )

        except Exception:

            page_text = ""

        if page_text:

            text += (
                f"\n\n"
                f"--- Page {page_number} ---"
                f"\n\n"
            )

            text += page_text

    return text


# ============================================================
# CHUNK TEXT
# ============================================================

def chunk_text(
    text,
    chunk_size=None,
    chunk_overlap=None,
):

    if not text:
        return []

    # Environment-based configuration
    if chunk_size is None:

        import os

        chunk_size = int(
            os.getenv(
                "CHUNK_SIZE",
                "1000",
            )
        )

    if chunk_overlap is None:

        import os

        chunk_overlap = int(
            os.getenv(
                "CHUNK_OVERLAP",
                "150",
            )
        )

    if chunk_overlap >= chunk_size:

        raise ValueError(
            "chunk_overlap must be smaller "
            "than chunk_size"
        )

    chunks = []

    start = 0

    step = (
        chunk_size
        - chunk_overlap
    )

    while start < len(text):

        end = start + chunk_size

        chunk = text[
            start:end
        ].strip()

        if chunk:

            chunks.append(
                chunk
            )

        start += step

    return chunks


# ============================================================
# METADATA EXTRACTION
# ============================================================

def extract_metadata(pages):

    if not pages:

        return {
            "title": "Unknown",
            "authors": "Unknown",
            "year": "Unknown",
            "doi": "Not found",
        }

    first_pages_text = "\n".join(
        page["text"]
        for page in pages[:3]
        if page.get("text")
    )

    lines = [
        line.strip()
        for line in first_pages_text.splitlines()
        if line.strip()
    ]

    # ========================================================
    # DOI
    # ========================================================

    doi_match = re.search(
        r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)",
        first_pages_text,
        re.IGNORECASE,
    )

    doi = (
        doi_match.group(1).rstrip(
            ".,;)"
        )
        if doi_match
        else "Not found"
    )

    # ========================================================
    # YEAR
    # ========================================================

    year_match = re.search(
        r"\b(19|20)\d{2}\b",
        first_pages_text,
    )

    year = (
        year_match.group(0)
        if year_match
        else "Unknown"
    )

    # ========================================================
    # TITLE
    # ========================================================

    title = "Unknown"

    ignored = {
        "abstract",
        "introduction",
        "contents",
        "keywords",
        "references",
        "acknowledgements",
        "acknowledgments",
    }

    title_candidates = []

    for line in lines[:30]:

        clean_line = line.strip()

        lower = clean_line.lower()

        if lower in ignored:
            continue

        if len(clean_line) < 15:
            continue

        if len(clean_line) > 250:
            continue

        if re.search(
            r"^(arxiv|doi|http|www\.)",
            lower,
        ):
            continue

        if re.search(
            r"\b(university|department|institute|school)\b",
            lower,
        ):
            continue

        title_candidates.append(
            clean_line
        )

    if title_candidates:

        title = title_candidates[0]

    # ========================================================
    # AUTHORS
    # ========================================================

    authors = "Unknown"

    for index, line in enumerate(
        lines[:30]
    ):

        if (
            title != "Unknown"
            and line == title
        ):

            nearby = lines[
                index + 1:index + 5
            ]

            possible_author_text = (
                " ".join(
                    nearby
                )
            )

            if possible_author_text:

                lower_author = (
                    possible_author_text.lower()
                )

                if (
                    "university"
                    not in lower_author
                    and "department"
                    not in lower_author
                ):

                    authors = (
                        possible_author_text
                    )

                    break

    return {
        "title": title,
        "authors": authors,
        "year": year,
        "doi": doi,
    }