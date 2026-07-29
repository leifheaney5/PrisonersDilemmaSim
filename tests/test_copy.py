from pathlib import Path
import unittest


class CopyStyleTests(unittest.TestCase):
    @staticmethod
    def _repository_text_files():
        root = Path(__file__).resolve().parents[1]
        checked_suffixes = {".css", ".html", ".js", ".json", ".md", ".py", ".yaml", ".yml"}
        ignored_parts = {".git", "node_modules", "playwright-report", "test-results"}
        return (
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix in checked_suffixes
            and not ignored_parts.intersection(path.parts)
        )

    def test_repository_copy_does_not_use_em_dashes(self):
        root = Path(__file__).resolve().parents[1]
        em_dash = chr(0x2014)
        offenders = []
        for path in self._repository_text_files():
            if em_dash in path.read_text(encoding="utf-8"):
                offenders.append(str(path.relative_to(root)))
        self.assertEqual(offenders, [], f"Em dashes found in: {', '.join(offenders)}")

    def test_repository_has_no_unresolved_merge_markers(self):
        root = Path(__file__).resolve().parents[1]
        marker_prefixes = ("<<<<<<< ", "=======", ">>>>>>> ")
        offenders = []
        for path in self._repository_text_files():
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if line.startswith(marker_prefixes):
                    offenders.append(f"{path.relative_to(root)}:{line_number}")
        self.assertEqual(offenders, [], f"Merge markers found in: {', '.join(offenders)}")


if __name__ == "__main__":
    unittest.main()
