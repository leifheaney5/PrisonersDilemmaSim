import base64
import json
import unittest

from pages.artifact_io import decode_json_upload


def _upload(raw: bytes, media_type: str = "application/json") -> str:
    return f"data:{media_type};base64,{base64.b64encode(raw).decode('ascii')}"


class ArtifactUploadTests(unittest.TestCase):
    def test_valid_json_upload_round_trips(self):
        payload = {"schema_version": 1, "strategies": ["A", "B"]}
        contents = _upload(json.dumps(payload).encode("utf-8"))
        self.assertEqual(decode_json_upload(contents, max_bytes=1_000), payload)

    def test_invalid_data_urls_and_base64_are_rejected(self):
        for contents in (
            None,
            "plain text",
            "data:application/json,{}",
            "data:application/json;base64,***",
            _upload(b"{}", media_type="image/png"),
        ):
            with self.subTest(contents=contents):
                with self.assertRaises(ValueError):
                    decode_json_upload(contents, max_bytes=100)

    def test_size_limit_is_checked_before_and_after_decoding(self):
        contents = _upload(b'{"value":"too large"}')
        with self.assertRaisesRegex(ValueError, "exceeds 5 bytes"):
            decode_json_upload(contents, max_bytes=5)

    def test_non_standard_numbers_and_duplicate_keys_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-standard JSON number"):
            decode_json_upload(_upload(b'{"value":NaN}'), max_bytes=100)
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            decode_json_upload(_upload(b'{"value":1,"value":2}'), max_bytes=100)

    def test_deep_and_oversized_collections_are_rejected(self):
        deep: object = 0
        for _ in range(34):
            deep = [deep]
        with self.assertRaisesRegex(ValueError, "maximum depth"):
            decode_json_upload(_upload(json.dumps(deep).encode("utf-8")), max_bytes=1_000)

        many_items = json.dumps(list(range(20_001))).encode("utf-8")
        with self.assertRaisesRegex(ValueError, "too many items"):
            decode_json_upload(_upload(many_items), max_bytes=200_000)


if __name__ == "__main__":
    unittest.main()
