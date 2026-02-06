import tempfile
import unittest
from pathlib import Path

from click.testing import CliRunner

import chat_cli


class InterceptorPipelineTests(unittest.TestCase):
    def test_legacy_interceptor_payload_supported(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "legacy.py"
            p.write_text(
                "def intercept_payload(payload):\n"
                "  payload = dict(payload)\n"
                "  payload['model'] = 'override'\n"
                "  return payload\n",
                encoding="utf-8",
            )
            pipe = chat_cli.InterceptorPipeline([p])
            ctx = chat_cli.InterceptorContext(1, chat_cli.datetime.now(chat_cli.timezone.utc), "x")
            out = pipe.run_pre_send({"model": "a", "messages": []}, ctx)
            self.assertEqual(out["model"], "override")

    def test_pre_and_post_hooks(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "hooks.py"
            p.write_text(
                "def pre_send(payload, context):\n"
                "  payload = dict(payload)\n"
                "  payload['model'] = 'x'\n"
                "  return payload\n"
                "def post_receive(response_json, context):\n"
                "  response_json['message']['content'] += '!' \n"
                "  return response_json\n",
                encoding="utf-8",
            )
            pipe = chat_cli.InterceptorPipeline([p])
            ctx = chat_cli.InterceptorContext(2, chat_cli.datetime.now(chat_cli.timezone.utc), "x")
            out = pipe.run_pre_send({"model": "a", "messages": []}, ctx)
            self.assertEqual(out["model"], "x")
            post = pipe.run_post_receive({"message": {"content": "ok"}}, ctx)
            self.assertEqual(post["message"]["content"], "ok!")


class CliTests(unittest.TestCase):
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(chat_cli.app, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Click", result.output)


if __name__ == "__main__":
    unittest.main()
