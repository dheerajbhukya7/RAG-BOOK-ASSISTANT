import hashlib


def clean_text(text: str) -> str:
    return ' '.join(text.encode('utf-8', 'ignore').decode('utf-8').split())


def pdf_fingerprint(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
