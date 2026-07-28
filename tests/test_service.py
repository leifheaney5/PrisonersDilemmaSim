import unittest
from pathlib import Path

from pages.app import APP_ENV, APP_VERSION, app, display_page, server


def _component_text(component):
    if isinstance(component, str):
        return component
    children = getattr(component, "children", None)
    if isinstance(children, (list, tuple)):
        return " ".join(_component_text(child) for child in children)
    if children is None:
        return ""
    return _component_text(children)


class ServiceEndpointTests(unittest.TestCase):
    def test_health_endpoint(self):
        response = server.test_client().get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json(),
            {"status": "ok", "app_version": APP_VERSION, "environment": APP_ENV},
        )

    def test_liveness_and_readiness_endpoints(self):
        client = server.test_client()
        live = client.get("/health/live")
        ready = client.get("/health/ready")
        self.assertEqual(live.status_code, 200)
        self.assertEqual(live.get_json(), {"status": "ok", "service": "web"})
        self.assertEqual(ready.status_code, 200)
        self.assertEqual(ready.get_json()["status"], "ready")
        self.assertTrue(all(ready.get_json()["checks"].values()))

    def test_version_endpoint_declares_artifact_schemas(self):
        response = server.test_client().get("/version")
        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["app_version"], APP_VERSION)
        self.assertEqual(payload["game_version"], 1)
        self.assertIn("evolution_result", payload["artifact_schema_versions"])
        self.assertEqual(payload["environment"], APP_ENV)
        self.assertIn("commit_sha", payload)
        self.assertIn("deployed_at", payload)
        self.assertIn(payload["rate_limit_backend"], {"process", "redis"})

    def test_security_headers_are_present(self):
        response = server.test_client().get("/")
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(response.headers["X-Frame-Options"], "SAMEORIGIN")
        self.assertIn("object-src 'none'", response.headers["Content-Security-Policy"])
        self.assertIn("camera=()", response.headers["Permissions-Policy"])
        self.assertTrue(response.headers["X-Request-ID"])

    def test_request_ids_accept_safe_values_and_replace_unsafe_values(self):
        client = server.test_client()
        safe = client.get("/", headers={"X-Request-ID": "trace_123.test-value"})
        self.assertEqual(safe.headers["X-Request-ID"], "trace_123.test-value")

        unsafe = client.get("/", headers={"X-Request-ID": "unsafe request id"})
        self.assertNotEqual(unsafe.headers["X-Request-ID"], "unsafe request id")
        self.assertRegex(unsafe.headers["X-Request-ID"], r"^[0-9a-f]{32}$")

    def test_paypal_loader_does_not_poll_pages_without_a_button(self):
        source = Path("assets/paypal-hosted-buttons.js").read_text(encoding="utf-8")
        self.assertNotIn("setTimeout(tick", source)
        self.assertIn('paypalRendered = "pending"', source)
        self.assertIn('paypalRendered = "1"', source)
        self.assertIn('paypalRendered = "error"', source)

    def test_https_response_adds_hsts(self):
        response = server.test_client().get("/", headers={"X-Forwarded-Proto": "https"})
        self.assertIn("max-age=31536000", response.headers["Strict-Transport-Security"])

    def test_oversized_dash_request_is_rejected(self):
        response = server.test_client().post(
            "/_dash-update-component",
            data=b"x" * 3_000_001,
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.get_json()["error"], "request_too_large")

    def test_unknown_route_renders_not_found_page(self):
        text = _component_text(display_page("/not-a-real-page"))
        self.assertIn("Page not found", text)
        self.assertIn("Go to Overview", text)
        response = server.test_client().get("/not-a-real-page")
        self.assertEqual(response.status_code, 404)

    def test_known_dash_deep_links_return_success(self):
        client = server.test_client()
        for route in ("/learn", "/profiles", "/experiment", "/donate"):
            with self.subTest(route=route):
                self.assertEqual(client.get(route).status_code, 200)
        redirect_response = client.get("/explore")
        self.assertEqual(redirect_response.status_code, 302)
        self.assertEqual(redirect_response.headers["Location"], "/experiment")

    def test_layout_exposes_version_footer_and_metadata(self):
        text = _component_text(app.layout)
        self.assertIn(f"Version {APP_VERSION}", text)
        self.assertIn("Privacy", text)
        self.assertIn("Security", text)
        index = server.test_client().get("/").get_data(as_text=True)
        self.assertIn("Explore the Iterated Prisoner&#x27;s Dilemma", index)
        self.assertIn("Prisoner&#x27;s Dilemma Strategy Lab", index)
        self.assertNotIn("www.paypal.com/sdk/js", index)


if __name__ == "__main__":
    unittest.main()
