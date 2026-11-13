import unittest
from pdf_utils import clean_text, pdf_fingerprint

class PDFUtilsTests(unittest.TestCase):
    def test_normalizes_whitespace(self):
        self.assertEqual(clean_text("  Hello\n\tworld  "), "Hello world")

    def test_preserves_unicode(self):
        self.assertEqual(clean_text(" café 你好 "), "café 你好")

    def test_empty_text(self):
        self.assertEqual(clean_text(" \n "), "")

    def test_fingerprint_is_content_based(self):
        self.assertEqual(pdf_fingerprint(b"abc"), pdf_fingerprint(b"abc"))
        self.assertNotEqual(pdf_fingerprint(b"abc"), pdf_fingerprint(b"abd"))

if __name__ == "__main__":
    unittest.main()
