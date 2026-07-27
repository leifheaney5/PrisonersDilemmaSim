from pathlib import Path
import unittest


class CopyStyleTests(unittest.TestCase):
    def test_repository_copy_does_not_use_em_dashes(self):
        root = Path(__file__).resolve().parents[1]
        checked_suffixes = {".css", ".md", ".py", ".yaml", ".yml"}
        em_dash = chr(0x2014)
        offenders = []
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in checked_suffixes and ".git" not in path.parts:
                if em_dash in path.read_text(encoding="utf-8"):
                    offenders.append(str(path.relative_to(root)))
        self.assertEqual(offenders, [], f"Em dashes found in: {', '.join(offenders)}")


if __name__ == "__main__":
    unittest.main()
