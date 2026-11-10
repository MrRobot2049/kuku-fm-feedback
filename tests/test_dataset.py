import unittest

import pandas as pd

import agentic_rag as rag


class DatasetContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = pd.read_csv(rag.DATA_PATH).fillna("")

    def test_expected_review_fields_exist(self):
        self.assertTrue({"content", "score", "at", "appVersion", "thumbsUpCount", "replyContent"}
                        .issubset(self.frame.columns))

    def test_sample_is_nonempty_and_within_requested_collection_size(self):
        self.assertGreater(len(self.frame), 0)
        self.assertLessEqual(len(self.frame), 200)

    def test_review_text_and_ratings_are_usable(self):
        self.assertTrue(self.frame["content"].astype(str).str.strip().ne("").all())
        self.assertTrue(self.frame["score"].astype(int).between(1, 5).all())

    def test_duplicate_review_text_was_removed(self):
        # Exact duplicate strings are removed during collection. Differently
        # capitalized short reviews (for example "good" and "Good") are retained.
        normalized = self.frame["content"].astype(str).str.strip()
        self.assertEqual(normalized.nunique(), len(self.frame))

    def test_dates_are_parseable(self):
        parsed = pd.to_datetime(self.frame["at"], errors="coerce", utc=True)
        self.assertTrue(parsed.notna().all())


if __name__ == "__main__":
    unittest.main()
