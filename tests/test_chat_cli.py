import json
import tempfile
import unittest
from pathlib import Path

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
                "  response_json['message']['content'] += '!'\n"
                "  return response_json\n",
                encoding="utf-8",
            )
            pipe = chat_cli.InterceptorPipeline([p])
            ctx = chat_cli.InterceptorContext(2, chat_cli.datetime.now(chat_cli.timezone.utc), "x")
            out = pipe.run_pre_send({"model": "a", "messages": []}, ctx)
            self.assertEqual(out["model"], "x")
            post = pipe.run_post_receive({"message": {"content": "ok"}}, ctx)
            self.assertEqual(post["message"]["content"], "ok!")


class ConfigTests(unittest.TestCase):
    def test_build_config_respects_no_transcript_and_no_stream(self):
        args = chat_cli.parse_args([
            "--endpoint",
            "http://x",
            "--model",
            "m",
            "--no-transcript",
            "--no-stream",
        ])
        cfg = chat_cli.build_config(args)
        self.assertIsNone(cfg.transcript_path)
        self.assertFalse(cfg.stream)


if __name__ == "__main__":
    unittest.main()
