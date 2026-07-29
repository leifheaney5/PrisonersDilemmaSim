import unittest

from flask import Flask, g

from pages.error_reporting import MAX_PUBLIC_ERROR_DETAIL, public_error_message


class ErrorReportingTests(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)

    def test_validation_errors_keep_bounded_actionable_details(self):
        with self.app.test_request_context("/"):
            g._pd_request_id = "request-123"
            message = public_error_message("Import failed", ValueError("bad\nsetting"), code="test_import")
        self.assertEqual(message, "Import failed: bad setting Reference: request-123.")

        with self.app.test_request_context("/"):
            g._pd_request_id = "request-456"
            message = public_error_message(
                "Import failed",
                ValueError("x" * (MAX_PUBLIC_ERROR_DETAIL + 100)),
                code="test_import",
            )
        detail = message.split(": ", 1)[1].split(" Reference:", 1)[0]
        self.assertEqual(len(detail), MAX_PUBLIC_ERROR_DETAIL)
        self.assertTrue(detail.endswith("..."))

    def test_unexpected_errors_do_not_expose_internal_messages(self):
        with self.app.test_request_context("/"):
            g._pd_request_id = "request-secret"
            with self.assertLogs("pd", level="ERROR") as logs:
                message = public_error_message(
                    "Export failed",
                    RuntimeError("secret filesystem path /srv/private"),
                    code="test_export",
                )
        self.assertEqual(message, "Export failed. Reference: request-secret.")
        self.assertNotIn("/srv/private", message)
        self.assertIn("code=test_export", " ".join(logs.output))


if __name__ == "__main__":
    unittest.main()
