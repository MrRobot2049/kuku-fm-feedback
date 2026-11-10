import csv
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import agentic_rag as rag


class QueryPlanningTests(unittest.TestCase):
    def test_offline_expansion_avoids_unrelated_audio_terms(self):
        queries = rag.offline_query_rewrites(
            "What are listeners saying about unexpected payments and refund support?"
        )
        self.assertGreaterEqual(len(queries), 2)
        self.assertTrue(any("autopay" in query for query in queries))
        self.assertFalse(any("playback" in query for query in queries))

    def test_plan_queries_parses_llm_json(self):
        with patch.object(rag, "call_llm", return_value='{"queries":["autopay charges","refund complaint"]}'):
            queries, method = rag.plan_queries("Why was I charged?")
        self.assertEqual(method, "llm")
        self.assertEqual(queries[0], "Why was I charged?")
        self.assertIn("autopay charges", queries)

    def test_topic_rewrites_cover_customer_feedback_categories(self):
        cases = {
            "My subscription charge was deducted": "autopay",
            "Support never responded to my refund request": "customer support",
            "Playback keeps buffering": "buffering",
            "Recommend a new story": "audiobooks",
            "The app crashes when I log in": "reliability",
            "Please add more Hindi language content": "Hindi",
            "Can I listen to episodes offline?": "offline",
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                rewrites = rag.offline_query_rewrites(question)
                self.assertTrue(any(expected.lower() in q.lower() for q in rewrites), rewrites)

    def test_unrelated_question_gets_generic_rewrite(self):
        rewrites = rag.offline_query_rewrites("What do people think?")
        self.assertEqual(len(rewrites), 2)
        self.assertIn("Kuku FM listener feedback", rewrites[1])

    def test_malformed_llm_json_falls_back_to_rules(self):
        with patch.object(rag, "call_llm", return_value="not json"):
            queries, method = rag.plan_queries("Why was my payment deducted?")
        self.assertEqual(method, "offline-rule-based")
        self.assertTrue(any("autopay" in query for query in queries))

    def test_empty_llm_query_list_falls_back_to_rules(self):
        with patch.object(rag, "call_llm", return_value='{"queries":[]}'):
            queries, method = rag.plan_queries("Why was my payment deducted?")
        self.assertEqual(method, "offline-rule-based")
        self.assertGreaterEqual(len(queries), 2)


class DenseRetrievalTests(unittest.TestCase):
    def test_retriever_returns_best_match_and_stable_citation(self):
        rows = [
            {"content": "Unexpected autopay charge", "score": 1, "at": "2026-01-01"},
            {"content": "Audio keeps buffering", "score": 2, "at": "2026-01-02"},
        ]
        fake_model = MagicMock()
        fake_model.encode.side_effect = [
            # Corpus embeddings: charge, audio.
            __import__("numpy").array([[1.0, 0.0], [0.0, 1.0]]),
            # Search vector points to the charge review.
            __import__("numpy").array([[1.0, 0.0]]),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "reviews.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            fake_transformers = types.SimpleNamespace(
                SentenceTransformer=MagicMock(return_value=fake_model)
            )
            with patch.object(rag, "CACHE_DIR", Path(temp_dir) / "cache"), patch.dict(
                sys.modules, {"sentence_transformers": fake_transformers}
            ):
                retriever = rag.DenseRetriever(csv_path)
                result = retriever.retrieve(["payment deducted"], top_k=1)
        self.assertEqual(result[0].citation, "R001")
        self.assertEqual(result[0].text, "Unexpected autopay charge")
        self.assertAlmostEqual(result[0].embedding_score, 1.0)

    def test_cross_encoder_drops_large_relevance_outlier(self):
        candidates = [
            rag.Candidate(0, "R001", "AutoPay charge", 1, "2026-01-01", 0.8),
            rag.Candidate(1, "R002", "Refund denied", 1, "2026-01-02", 0.7),
            rag.Candidate(2, "R003", "Audio story recommendation", 5, "2026-01-03", 0.4),
        ]
        fake_model = MagicMock()
        fake_model.predict.return_value = [1.0, -1.0, -8.0]
        fake_transformers = types.SimpleNamespace(CrossEncoder=MagicMock(return_value=fake_model))
        with patch.dict(sys.modules, {"sentence_transformers": fake_transformers}):
            ranked = rag.RelevanceReranker().rerank("unexpected payment refund", candidates)
        self.assertEqual([item.citation for item in ranked], ["R001", "R002"])
        self.assertGreater(ranked[0].rerank_score, ranked[1].rerank_score)

    def test_reranker_empty_candidates_does_not_load_model(self):
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            self.assertEqual(rag.RelevanceReranker().rerank("query", []), [])

    def test_reranker_respects_keep_count(self):
        candidates = [
            rag.Candidate(i, f"R{i + 1:03d}", f"review {i}", 1, None, 0.5)
            for i in range(4)
        ]
        fake_model = MagicMock()
        fake_model.predict.return_value = [3.0, 2.0, 1.0, 0.0]
        fake_transformers = types.SimpleNamespace(CrossEncoder=MagicMock(return_value=fake_model))
        with patch.dict(sys.modules, {"sentence_transformers": fake_transformers}):
            ranked = rag.RelevanceReranker().rerank("query", candidates, keep=2)
        self.assertEqual([c.citation for c in ranked], ["R001", "R002"])

    def test_evidence_text_includes_stable_citation_and_metadata(self):
        candidate = rag.Candidate(6, "R007", "Payment deducted", 1, "2026-10-01", .71, -2.0)
        rendered = rag.evidence_text([candidate])
        self.assertIn("[R007]", rendered)
        self.assertIn("1/5 stars", rendered)
        self.assertIn("Payment deducted", rendered)


class ProviderTests(unittest.TestCase):
    def test_groq_key_resolves_to_configured_default_model(self):
        with patch.object(rag, "LLM_PROVIDER", "groq"), patch.dict(
            os.environ, {"GROQ_API_KEY": "test-key"}, clear=True
        ):
            provider, model, key_env, base_url = rag.resolve_llm_config()
        self.assertEqual(provider, "groq")
        self.assertEqual(model, "openai/gpt-oss-20b")
        self.assertEqual(key_env, "GROQ_API_KEY")
        self.assertEqual(base_url, "https://api.groq.com/openai/v1")

    def test_gemini_provider_uses_compatible_chat_api(self):
        completion = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="rewritten query"))]
        )
        create = MagicMock(return_value=completion)

        class FakeOpenAI:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                instances.append(self)
                self.chat = types.SimpleNamespace(
                    completions=types.SimpleNamespace(create=create)
                )

        instances = []
        fake_module = types.SimpleNamespace(OpenAI=FakeOpenAI)
        with patch.dict(sys.modules, {"openai": fake_module}), patch.object(
            rag, "LLM_PROVIDER", "gemini"
        ), patch.dict(os.environ, {
            "GEMINI_API_KEY": "test-key",
            "LLM_MODEL": "gemini-3.8-flash",
        }, clear=False):
            result = rag.call_llm("system instructions", "question")
        self.assertEqual(result, "rewritten query")
        self.assertEqual(create.call_args.kwargs["model"], "gemini-3.8-flash")
        self.assertEqual(
            instances[0].kwargs["base_url"],
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    def test_explicit_session_config_uses_key_without_environment_variable(self):
        completion = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="ok"))]
        )
        instances = []

        class FakeOpenAI:
            def __init__(self, **kwargs):
                instances.append(kwargs)
                self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(
                    create=MagicMock(return_value=completion)
                ))

        with patch.dict(sys.modules, {"openai": types.SimpleNamespace(OpenAI=FakeOpenAI)}), \
             patch.dict(os.environ, {}, clear=True):
            self.assertEqual(rag.call_llm("rules", "prompt", {
                "provider": "groq", "model": "test-model", "api_key": "session-secret"
            }), "ok")
        self.assertEqual(instances[0]["api_key"], "session-secret")
        self.assertEqual(instances[0]["base_url"], "https://api.groq.com/openai/v1")

    def test_unknown_provider_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "No API key found"):
            rag.resolve_llm_config({"provider": "unknown"})

    def test_missing_provider_key_is_rejected_before_network_call(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "GROQ_API_KEY is required"):
                rag.call_llm("rules", "prompt", {"provider": "groq", "model": "m"})


class PipelineTests(unittest.TestCase):
    def test_pipeline_passes_ranked_evidence_through_answer_audit(self):
        candidate = rag.Candidate(0, "R001", "AutoPay payment review", 1, "2026-01-01", 0.9, 1.2)
        retriever = MagicMock()
        retriever.retrieve.return_value = [candidate]
        reranker = MagicMock()
        reranker.rerank.return_value = [candidate]
        with (
            patch.object(rag, "plan_queries", return_value=(['payment', 'autopay'], 'llm')),
            patch.object(rag, "DenseRetriever", return_value=retriever),
            patch.object(rag, "RelevanceReranker", return_value=reranker),
            patch.object(rag, "make_answer", return_value="Draft [R001].") as make,
            patch.object(rag, "verify_answer", return_value="Checked [R001].") as verify,
        ):
            result = rag.run("Was I charged?", top_k=4)

        self.assertEqual(result["rewrite_method"], "llm")
        self.assertEqual(result["answer"], "Checked [R001].")
        self.assertEqual(result["evidence"][0]["citation"], "R001")
        make.assert_called_once_with("Was I charged?", [candidate])
        verify.assert_called_once_with("Was I charged?", "Draft [R001].", [candidate])

    def test_offline_mode_skips_all_llm_functions(self):
        candidate = rag.Candidate(0, "R001", "Buffering during playback", 1, None, .8, 1.0)
        retriever = MagicMock()
        retriever.retrieve.return_value = [candidate]
        reranker = MagicMock()
        reranker.rerank.return_value = [candidate]
        with patch.object(rag, "plan_queries") as planner, \
             patch.object(rag, "make_answer") as answer, \
             patch.object(rag, "verify_answer") as audit:
            result = rag.run("Playback buffers", use_llm=False,
                             retriever=retriever, reranker=reranker)
        planner.assert_not_called()
        answer.assert_not_called()
        audit.assert_not_called()
        self.assertEqual(result["answer_method"], "offline evidence-only mode")
        self.assertEqual(result["evidence"][0]["citation"], "R001")

    def test_empty_question_and_invalid_top_k_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Question must not be empty"):
            rag.run("  ")
        with self.assertRaisesRegex(ValueError, "top_k must be at least 1"):
            rag.run("A valid question", top_k=0)

    def test_citation_audit_failure_preserves_draft_and_reports_limitation(self):
        candidate = rag.Candidate(0, "R001", "Refund support", 1, None, .8, 1.0)
        retriever = MagicMock()
        retriever.retrieve.return_value = [candidate]
        reranker = MagicMock()
        reranker.rerank.return_value = [candidate]
        with patch.object(rag, "plan_queries", return_value=(["refund"], "llm")), \
             patch.object(rag, "make_answer", return_value="Claim [R001]."), \
             patch.object(rag, "verify_answer", side_effect=RuntimeError("provider down")):
            result = rag.run("Refund?", retriever=retriever, reranker=reranker)
        self.assertEqual(result["answer"], "Claim [R001].")
        self.assertIn("citation audit unavailable", result["answer_method"])


if __name__ == "__main__":
    unittest.main()
