"""Offline tests for the local-only embedding service.

No embedding API is called and no network is used. The end-to-end retrieval test writes to a
temporary SQLite store, never to data/vector_store.db.
"""

import hashlib
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.core.config import Settings, settings
from app.services.embedding_service import (
    SUPPORTED_EMBEDDING_PROVIDERS,
    EmbeddingService,
    UnsupportedEmbeddingProviderError,
    embedding_service,
)

ROOT = Path(__file__).resolve().parents[1]

# Fingerprint of the local embedder's output, recorded before the OpenAI provider was removed.
# If this changes, existing vector stores would need to be re-ingested.
FINGERPRINT_TEXTS = ["Paris museums and Metro", "Tokyo  FOOD tour", "", "a a a b", "Café Été 東京"]
FINGERPRINT_SHA256 = "aff136857454d7138cdd0e744595e9c14d3eabe75a8a4c6a0d6d2d8b707d2655"


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


class LocalEmbeddingBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.service = EmbeddingService(provider="local", dimensions=2048)

    def test_default_service_is_local_with_configured_dimensions(self):
        self.assertEqual(SUPPORTED_EMBEDDING_PROVIDERS, ("local",))
        self.assertEqual(settings.EMBEDDING_PROVIDER, "local")
        self.assertEqual(embedding_service.provider, "local")
        self.assertEqual(embedding_service.dimensions, settings.LOCAL_EMBEDDING_DIMENSIONS)

    def test_output_is_unchanged_from_the_original_local_embedder(self):
        vectors = self.service.embed(FINGERPRINT_TEXTS)
        self.assertEqual(hashlib.sha256(json.dumps(vectors).encode()).hexdigest(), FINGERPRINT_SHA256)

    def test_vectors_have_configured_dimensions_and_unit_length(self):
        for dimensions in (16, 2048):
            service = EmbeddingService(provider="local", dimensions=dimensions)
            [vector] = service.embed(["Louvre museum tickets"])
            self.assertEqual(len(vector), dimensions)
            self.assertAlmostEqual(math.sqrt(sum(v * v for v in vector)), 1.0, places=9)

    def test_embedding_is_deterministic_and_case_and_whitespace_insensitive(self):
        first = self.service.embed_query("Paris Museums")
        self.assertEqual(first, self.service.embed_query("Paris Museums"))
        self.assertEqual(first, self.service.embed_query("  paris   MUSEUMS "))
        self.assertEqual(first, EmbeddingService(provider="local", dimensions=2048).embed_query("paris museums"))

    def test_empty_text_gives_a_zero_vector(self):
        self.assertEqual(self.service.embed_query(""), [0.0] * 2048)

    def test_embed_query_matches_batch_embedding(self):
        texts = ["Shibuya crossing at night", "Seine river cruise"]
        batch = self.service.embed(texts)
        self.assertEqual([self.service.embed_query(t) for t in texts], batch)
        self.assertEqual(self.service.embed([]), [])

    def test_related_text_scores_higher_than_unrelated_text(self):
        query = self.service.embed_query("Paris museums")
        related = self.service.embed_query("museums in Paris near the Seine")
        unrelated = self.service.embed_query("Tokyo ramen street food")
        self.assertGreater(cosine(query, related), cosine(query, unrelated))

    def test_provider_value_is_normalized(self):
        self.assertEqual(EmbeddingService(provider=" LOCAL ", dimensions=8).provider, "local")


class UnsupportedProviderTests(unittest.TestCase):
    def test_unsupported_providers_are_rejected_without_calling_anything(self):
        for provider in ("openai", "OpenAI", "remote", "", "local-ish"):
            with self.subTest(provider=provider):
                service = EmbeddingService(provider=provider, dimensions=8)
                # Any attempt to import an SDK during embedding would raise ImportError instead.
                with patch.dict(sys.modules, {"openai": None}):
                    with self.assertRaises(UnsupportedEmbeddingProviderError) as raised:
                        service.embed(["Paris"])
                    with self.assertRaises(UnsupportedEmbeddingProviderError):
                        service.embed_query("Paris")
                self.assertIsInstance(raised.exception, ValueError)
                self.assertIn("only 'local'", str(raised.exception))
                if provider:
                    self.assertNotIn(provider, str(raised.exception))

    def test_configured_unsupported_provider_is_rejected(self):
        with patch("app.services.embedding_service.settings.EMBEDDING_PROVIDER", "openai"):
            service = EmbeddingService()
        with self.assertRaises(UnsupportedEmbeddingProviderError):
            service.embed(["Paris"])


class OpenAIConfigurationRemovedTests(unittest.TestCase):
    def test_settings_have_no_openai_or_remote_embedding_fields(self):
        self.assertNotIn("OPENAI_API_KEY", Settings.model_fields)
        self.assertNotIn("EMBEDDING_MODEL", Settings.model_fields)
        self.assertIn("EMBEDDING_PROVIDER", Settings.model_fields)

    def test_stale_openai_variables_are_ignored(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-stale-not-used", "EMBEDDING_MODEL": "x"}):
            loaded = Settings()
        self.assertFalse(hasattr(loaded, "OPENAI_API_KEY"))
        self.assertFalse(hasattr(loaded, "EMBEDDING_MODEL"))

    def test_embedding_service_has_no_remote_client(self):
        service = EmbeddingService(provider="local", dimensions=8)
        for attribute in ("api_key", "model", "_client", "_openai_client"):
            self.assertFalse(hasattr(service, attribute), attribute)

    def test_dependencies_and_env_example_do_not_reference_openai(self):
        for name in ("requirements.txt", ".env.example"):
            with self.subTest(file=name):
                text = (ROOT / name).read_text(encoding="utf-8").casefold()
                self.assertNotIn("openai", text)
                self.assertNotIn("embedding_model", text)
        self.assertIn("EMBEDDING_PROVIDER=local", (ROOT / ".env.example").read_text(encoding="utf-8"))

    def test_application_source_does_not_import_openai(self):
        for path in (ROOT / "app").rglob("*.py"):
            with self.subTest(file=str(path.relative_to(ROOT))):
                source = path.read_text(encoding="utf-8")
                self.assertNotRegex(source, r"(?m)^\s*(from|import)\s+openai\b")


class LocalRetrievalEndToEndTests(unittest.TestCase):
    def test_ingest_and_search_use_local_embeddings(self):
        from app.services.knowledge_service import KnowledgeService
        from app.services.vector_store import SQLiteVectorStore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            docs = Path(directory) / "docs"
            docs.mkdir()
            (docs / "paris-guide.md").write_text(
                "# Paris\n\nThe Louvre and Orsay museums sit along the Seine in Paris.", encoding="utf-8"
            )
            (docs / "tokyo-guide.md").write_text(
                "# Tokyo\n\nTokyo ramen shops and street food stalls fill Shinjuku.", encoding="utf-8"
            )
            service = KnowledgeService(store=SQLiteVectorStore(str(Path(directory) / "store.db")))
            with patch.dict(sys.modules, {"openai": None}):
                self.assertGreaterEqual(service.ingest(docs), 2)
                results = service.search("Paris museums Louvre", top_k=2)

        self.assertTrue(results)
        self.assertEqual(results[0]["metadata"]["destination"], "Paris")


if __name__ == "__main__":
    unittest.main()
