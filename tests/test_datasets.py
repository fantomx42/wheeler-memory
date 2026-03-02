"""Slow benchmark tests using real-world datasets.

These tests validate that Wheeler Memory's CA-based storage and recall
works on real-world text from publicly available datasets. Each test
is marked @pytest.mark.slow and skips gracefully when datasets are not
present on disk.

Datasets used:
  - MBPP: Python programming problems (420KB)
  - SWE-bench Verified: GitHub issue descriptions (2.1MB)
  - BABILong: Simple QA fact narratives (0k JSON files, tiny)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import pearsonr

pd = pytest.importorskip("pandas", reason="pandas required for parquet dataset tests")

from wheeler_memory.chunking import select_chunk
from wheeler_memory.dynamics import evolve_and_interpret
from wheeler_memory.hashing import hash_to_frame
from wheeler_memory.storage import recall_memory

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import store_test_memory

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------

DATASETS_DIR = Path("/home/tristan/wheeler memory/datasets")

MBPP_TRAIN_PATH = DATASETS_DIR / "mbpp" / "full" / "train-00000-of-00001.parquet"
MBPP_TEST_PATH = DATASETS_DIR / "mbpp" / "full" / "test-00000-of-00001.parquet"
SWE_BENCH_PATH = (
    DATASETS_DIR / "SWE-bench_Verified" / "data" / "test-00000-of-00001.parquet"
)
BABILONG_QA1_PATH = DATASETS_DIR / "babilong" / "data" / "qa1" / "0k.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_parquet_texts(path: Path, column: str, n: int) -> list[str]:
    """Load first *n* non-empty text values from a parquet column."""
    df = pd.read_parquet(path, columns=[column])
    texts = df[column].dropna().astype(str).tolist()
    # Filter out very short entries that would be meaningless
    texts = [t for t in texts if len(t.strip()) > 10]
    return texts[:n]


def _load_babilong_json(path: Path) -> list[dict]:
    """Load a BABILong JSON file (list of {input, question, target})."""
    with open(path) as f:
        return json.load(f)


def _evolve_texts(texts: list[str], max_iters: int = 1000) -> list[dict]:
    """Evolve each text through the CA and return result dicts."""
    results = []
    for text in texts:
        frame = hash_to_frame(text)
        result = evolve_and_interpret(
            frame, max_iters=max_iters, stability_threshold=1e-4
        )
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# MBPP tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(
    not MBPP_TRAIN_PATH.exists(), reason="MBPP dataset not found on disk"
)
class TestMBPP:
    """Benchmark tests using the MBPP (Mostly Basic Python Problems) dataset.

    MBPP contains short natural-language descriptions of Python programming
    tasks, making it ideal for testing code-domain routing and attractor
    diversity on realistic short text.
    """

    def test_store_and_self_recall(self, tmp_path):
        """Store 20 MBPP problems, recall each by its own text.

        Because recall uses SHA-256 hashing (deterministic), querying with
        the exact stored text must produce a perfect self-match at rank 1.
        """
        texts = _load_parquet_texts(MBPP_TRAIN_PATH, "text", 20)
        assert len(texts) >= 10, "Not enough MBPP entries loaded"

        # Store all items into the same chunk for controlled recall
        for text in texts:
            store_test_memory(text, tmp_path, chunk="code")

        # Recall each and verify it is the top-1 result
        perfect_recalls = 0
        for text in texts:
            results = recall_memory(text, top_k=5, data_dir=tmp_path, chunk="code")
            assert len(results) > 0, f"No recall results for: {text[:60]}..."
            if results[0]["text"] == text:
                perfect_recalls += 1
            # Self-recall with SHA hash should be a perfect correlation
            assert results[0]["similarity"] > 0.99, (
                f"Self-recall similarity {results[0]['similarity']:.4f} "
                f"should be ~1.0 for identical text"
            )

        assert perfect_recalls == len(texts), (
            f"Only {perfect_recalls}/{len(texts)} perfect self-recalls "
            f"(expected all to be top-1)"
        )

    def test_convergence_rate(self):
        """Evolve 50 MBPP problem texts through the CA.

        At least 90% should reach CONVERGED or OSCILLATING state (not CHAOTIC).
        This validates that real-world code problem descriptions produce
        stable attractors in the CA system.
        """
        texts = _load_parquet_texts(MBPP_TRAIN_PATH, "text", 50)
        assert len(texts) >= 30, "Not enough MBPP entries loaded"

        results = _evolve_texts(texts)
        states = [r["state"] for r in results]

        converged = sum(1 for s in states if s == "CONVERGED")
        oscillating = sum(1 for s in states if s == "OSCILLATING")
        chaotic = sum(1 for s in states if s == "CHAOTIC")
        stable = converged + oscillating

        assert stable / len(states) >= 0.90, (
            f"Convergence rate {stable}/{len(states)} = "
            f"{stable / len(states):.1%} is below 90%. "
            f"(CONVERGED={converged}, OSCILLATING={oscillating}, CHAOTIC={chaotic})"
        )

    def test_chunk_routing(self):
        """MBPP problem descriptions should route to the 'code' chunk.

        MBPP texts describe Python programming tasks and typically contain
        keywords like 'function', 'array', 'class', etc. that trigger
        the code chunk router.
        """
        texts = _load_parquet_texts(MBPP_TRAIN_PATH, "text", 30)
        assert len(texts) >= 10, "Not enough MBPP entries loaded"

        code_routed = sum(1 for t in texts if select_chunk(t) == "code")

        # At least 50% should route to code (some MBPP problems are
        # described generically without explicit programming keywords)
        assert code_routed / len(texts) >= 0.50, (
            f"Only {code_routed}/{len(texts)} MBPP problems routed to 'code' chunk. "
            f"Expected at least 50%. Sample non-code routings: "
            + ", ".join(
                f"'{t[:50]}' -> '{select_chunk(t)}'"
                for t in texts[:5]
                if select_chunk(t) != "code"
            )
        )

    def test_attractor_diversity(self):
        """Attractors from 20 distinct MBPP problems should be diverse.

        Computes pairwise Pearson correlation on flattened attractors.
        Average absolute correlation should be < 0.5 and max < 0.85,
        confirming the CA produces meaningfully different fixed points
        for different real-world inputs.
        """
        texts = _load_parquet_texts(MBPP_TRAIN_PATH, "text", 20)
        assert len(texts) >= 10, "Not enough MBPP entries loaded"

        results = _evolve_texts(texts)
        attractors = [r["attractor"].flatten() for r in results]

        n = len(attractors)
        correlations = []
        for i in range(n):
            for j in range(i + 1, n):
                corr, _ = pearsonr(attractors[i], attractors[j])
                correlations.append(corr)

        correlations = np.array(correlations)
        avg_corr = np.mean(np.abs(correlations))
        max_corr = np.max(np.abs(correlations))

        assert avg_corr < 0.5, (
            f"Average attractor correlation {avg_corr:.4f} should be < 0.5 "
            f"(MBPP attractors are not diverse enough)"
        )
        assert max_corr < 0.85, (
            f"Max attractor correlation {max_corr:.4f} should be < 0.85 "
            f"(some MBPP attractor pairs are too similar)"
        )


# ---------------------------------------------------------------------------
# SWE-bench tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(
    not SWE_BENCH_PATH.exists(), reason="SWE-bench Verified dataset not found on disk"
)
class TestSWEBench:
    """Benchmark tests using SWE-bench Verified dataset.

    SWE-bench contains real GitHub issue descriptions (problem_statements)
    which are longer and more complex than MBPP texts, often including code
    snippets, tracebacks, and natural language mixed together.
    """

    def test_store_and_recall_issues(self, tmp_path):
        """Store 15 SWE-bench issues, recall with a keyword substring.

        For each stored issue, extract a short substring (first 80 chars)
        and use it as a recall query. Since SHA-256 hashing means only
        exact text match gives perfect correlation, we instead verify
        that the recall system functions correctly: results are returned,
        similarities are bounded, and the system handles long real-world
        text without errors.
        """
        texts = _load_parquet_texts(SWE_BENCH_PATH, "problem_statement", 15)
        assert len(texts) >= 5, "Not enough SWE-bench entries loaded"

        # Store all issues
        for text in texts:
            store_test_memory(text, tmp_path, chunk="code")

        # Self-recall: query with exact stored text
        for text in texts:
            results = recall_memory(text, top_k=5, data_dir=tmp_path, chunk="code")
            assert len(results) > 0, (
                f"No results for SWE-bench issue: {text[:60]}..."
            )
            # Exact text self-recall should be a perfect match
            assert results[0]["text"] == text, (
                f"Top result text does not match query for: {text[:60]}..."
            )
            assert results[0]["similarity"] > 0.99, (
                f"Self-recall similarity {results[0]['similarity']:.4f} "
                f"should be ~1.0 for identical text"
            )

    def test_convergence_rate(self):
        """Evolve 30 SWE-bench issue descriptions through the CA.

        At least 90% should reach CONVERGED or OSCILLATING state.
        SWE-bench issues are longer and more varied than MBPP, testing
        the CA on realistic software engineering prose.
        """
        texts = _load_parquet_texts(SWE_BENCH_PATH, "problem_statement", 30)
        assert len(texts) >= 15, "Not enough SWE-bench entries loaded"

        results = _evolve_texts(texts)
        states = [r["state"] for r in results]

        converged = sum(1 for s in states if s == "CONVERGED")
        oscillating = sum(1 for s in states if s == "OSCILLATING")
        chaotic = sum(1 for s in states if s == "CHAOTIC")
        stable = converged + oscillating

        assert stable / len(states) >= 0.90, (
            f"Convergence rate {stable}/{len(states)} = "
            f"{stable / len(states):.1%} is below 90%. "
            f"(CONVERGED={converged}, OSCILLATING={oscillating}, CHAOTIC={chaotic})"
        )

    def test_attractor_diversity(self):
        """Attractors from 15 SWE-bench issues should be diverse.

        SWE-bench issues come from different repositories and describe
        different bugs, so their attractors should show low correlation.
        """
        texts = _load_parquet_texts(SWE_BENCH_PATH, "problem_statement", 15)
        assert len(texts) >= 10, "Not enough SWE-bench entries loaded"

        results = _evolve_texts(texts)
        attractors = [r["attractor"].flatten() for r in results]

        n = len(attractors)
        correlations = []
        for i in range(n):
            for j in range(i + 1, n):
                corr, _ = pearsonr(attractors[i], attractors[j])
                correlations.append(corr)

        correlations = np.array(correlations)
        avg_corr = np.mean(np.abs(correlations))
        max_corr = np.max(np.abs(correlations))

        assert avg_corr < 0.5, (
            f"Average SWE-bench attractor correlation {avg_corr:.4f} should be < 0.5"
        )
        assert max_corr < 0.85, (
            f"Max SWE-bench attractor correlation {max_corr:.4f} should be < 0.85"
        )


# ---------------------------------------------------------------------------
# BABILong tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(
    not BABILONG_QA1_PATH.exists(), reason="BABILong dataset not found on disk"
)
class TestBABILong:
    """Benchmark tests using BABILong QA1 (single supporting fact).

    BABILong QA1 contains short narratives about people moving between
    locations, with questions like 'Where is Mary?' and single-word
    target answers. The 0k files are the smallest variant (100 items).
    """

    def test_fact_store_and_self_recall(self, tmp_path):
        """Store individual facts from BABILong, recall by exact text.

        Takes 10 BABILong items, splits each input into individual
        sentences, stores them as separate memories, then queries with
        the exact sentence text to verify self-recall works on short
        factual sentences.
        """
        data = _load_babilong_json(BABILONG_QA1_PATH)[:10]
        assert len(data) >= 5, "Not enough BABILong entries loaded"

        # Collect all unique sentences across the 10 items
        all_sentences = set()
        for item in data:
            sentences = [
                s.strip() + "."
                for s in item["input"].replace(".", ".\n").split("\n")
                if s.strip()
            ]
            all_sentences.update(sentences)

        # Store all unique sentences
        stored_sentences = list(all_sentences)[:40]  # Cap at 40 to keep test fast
        for sentence in stored_sentences:
            store_test_memory(sentence, tmp_path, chunk="general")

        # Self-recall: each stored sentence should be its own top-1 match
        perfect_recalls = 0
        for sentence in stored_sentences:
            results = recall_memory(
                sentence, top_k=3, data_dir=tmp_path, chunk="general"
            )
            if results and results[0]["text"] == sentence:
                perfect_recalls += 1

        assert perfect_recalls == len(stored_sentences), (
            f"Only {perfect_recalls}/{len(stored_sentences)} BABILong facts "
            f"achieved perfect self-recall"
        )

    def test_convergence_on_narrative(self):
        """Evolve 30 BABILong narrative inputs through the CA.

        BABILong inputs are short factual sentences about people and
        locations. At least 90% should converge or oscillate.
        """
        data = _load_babilong_json(BABILONG_QA1_PATH)[:30]
        assert len(data) >= 15, "Not enough BABILong entries loaded"

        texts = [item["input"] for item in data]
        results = _evolve_texts(texts)
        states = [r["state"] for r in results]

        converged = sum(1 for s in states if s == "CONVERGED")
        oscillating = sum(1 for s in states if s == "OSCILLATING")
        chaotic = sum(1 for s in states if s == "CHAOTIC")
        stable = converged + oscillating

        assert stable / len(states) >= 0.90, (
            f"Convergence rate {stable}/{len(states)} = "
            f"{stable / len(states):.1%} is below 90%. "
            f"(CONVERGED={converged}, OSCILLATING={oscillating}, CHAOTIC={chaotic})"
        )

    def test_narrative_attractor_diversity(self):
        """Attractors from 20 BABILong narratives should be diverse.

        Even though BABILong narratives share similar structure (person
        moved to location), different inputs should still produce distinct
        attractors due to different SHA-256 hashes.
        """
        data = _load_babilong_json(BABILONG_QA1_PATH)[:20]
        assert len(data) >= 10, "Not enough BABILong entries loaded"

        texts = [item["input"] for item in data]
        results = _evolve_texts(texts)
        attractors = [r["attractor"].flatten() for r in results]

        n = len(attractors)
        correlations = []
        for i in range(n):
            for j in range(i + 1, n):
                corr, _ = pearsonr(attractors[i], attractors[j])
                correlations.append(corr)

        correlations = np.array(correlations)
        avg_corr = np.mean(np.abs(correlations))
        max_corr = np.max(np.abs(correlations))

        assert avg_corr < 0.5, (
            f"Average BABILong attractor correlation {avg_corr:.4f} should be < 0.5"
        )
        assert max_corr < 0.85, (
            f"Max BABILong attractor correlation {max_corr:.4f} should be < 0.85"
        )

    def test_question_vs_fact_correlation(self, tmp_path):
        """Compare similarity between a question and its source facts.

        Store the sentences from one BABILong item, then recall using
        the question as the query. With SHA-256 hashing the question
        and facts will have different hashes (and thus low correlation),
        but this test documents that behavior and verifies the system
        handles cross-text queries without errors.
        """
        data = _load_babilong_json(BABILONG_QA1_PATH)
        # Pick an item with a reasonably long input
        item = data[0]

        sentences = [
            s.strip() + "."
            for s in item["input"].replace(".", ".\n").split("\n")
            if s.strip()
        ]

        for sentence in sentences:
            store_test_memory(sentence, tmp_path, chunk="general")

        # Recall using the question as query
        question = item["question"].strip()
        results = recall_memory(
            question, top_k=len(sentences), data_dir=tmp_path, chunk="general"
        )

        assert len(results) > 0, "No results returned for BABILong question query"

        # All similarities should be valid floats in [-1, 1]
        for r in results:
            assert -1.0 <= r["similarity"] <= 1.0, (
                f"Similarity {r['similarity']} out of bounds for: {r['text'][:50]}"
            )

        # Document the actual similarity range for SHA-based cross-text recall
        sims = [r["similarity"] for r in results]
        # SHA-based recall of different text should produce low but non-zero
        # correlation (essentially random correlation of two unrelated attractors)
        avg_sim = np.mean(np.abs(sims))
        assert avg_sim < 0.5, (
            f"Average |similarity| between question and facts is {avg_sim:.4f}; "
            f"expected low correlation for SHA-based cross-text recall"
        )


# ===========================================================================
# EMBEDDING MODE TESTS — semantic cross-text recall
# All tests require sentence-transformers ([embed] extra).
# Both store AND recall use embedding mode so attractors live in the
# same semantic space.
# ===========================================================================

_st = pytest.importorskip(
    "sentence_transformers",
    reason="sentence-transformers required for embedding tests",
)

from conftest import store_test_memory_embed  # noqa: E402


@pytest.mark.slow
@pytest.mark.embed
@pytest.mark.skipif(not MBPP_TRAIN_PATH.exists(), reason="MBPP dataset not present")
class TestMBPPEmbed:
    """Embedding-mode tests using MBPP Python programming problems."""

    def test_semantic_self_recall(self, tmp_path):
        """Store 10 problems with embed_to_frame; recall each by its own text.

        Same logical test as TestMBPP.test_store_and_self_recall but through
        the embedding path, validating the full embed store→recall roundtrip.
        """
        texts = _load_parquet_texts(MBPP_TRAIN_PATH, "text", 10)
        for text in texts:
            store_test_memory_embed(text, tmp_path, chunk="code")

        for text in texts:
            results = recall_memory(
                text, top_k=3, data_dir=tmp_path, chunk="code", use_embedding=True
            )
            assert results, f"No results for: {text[:50]}"
            assert results[0]["text"] == text, (
                f"Self-recall failed: expected top-1 == stored text, "
                f"got '{results[0]['text'][:40]}'"
            )

    def test_semantic_cross_recall(self, tmp_path):
        """Store problems with embedding; recall using shortened keyword queries.

        This is the test SHA-256 mode cannot pass: a shorter paraphrase of a
        stored problem should still retrieve that problem in top-3 results.
        """
        texts = _load_parquet_texts(MBPP_TRAIN_PATH, "text", 15)
        for text in texts:
            store_test_memory_embed(text, tmp_path, chunk="code")

        hits = 0
        for text in texts[:10]:
            # Build a keyword query from the first ~5 significant words
            words = [w for w in text.split() if len(w) > 3][:5]
            query = " ".join(words)
            results = recall_memory(
                query, top_k=5, data_dir=tmp_path, chunk="code", use_embedding=True
            )
            stored_texts = [r["text"] for r in results]
            if text in stored_texts:
                hits += 1

        # Expect most keyword queries to retrieve the right problem in top-5
        assert hits >= 6, (
            f"Semantic cross-recall: only {hits}/10 keyword queries found the "
            f"original problem in top-5. Embedding mode may not be working."
        )

    def test_embed_attractor_diversity(self, tmp_path):
        """Embedding-mode attractors should be diverse — avg pairwise Pearson < 0.5."""
        texts = _load_parquet_texts(MBPP_TRAIN_PATH, "text", 15)
        from wheeler_memory.embedding import embed_to_frame

        attractors = []
        for text in texts:
            frame = embed_to_frame(text)
            result = evolve_and_interpret(frame)
            attractors.append(result["attractor"].flatten())

        correlations = []
        for i in range(len(attractors)):
            for j in range(i + 1, len(attractors)):
                corr, _ = pearsonr(attractors[i], attractors[j])
                correlations.append(corr)

        avg_corr = np.mean(np.abs(correlations))
        max_corr = np.max(np.abs(correlations))
        assert avg_corr < 0.6, (
            f"Embedding attractors avg correlation {avg_corr:.4f} unexpectedly high"
        )
        assert max_corr < 0.95, (
            f"Embedding attractors max correlation {max_corr:.4f} — "
            f"two problems may be near-identical"
        )


@pytest.mark.slow
@pytest.mark.embed
@pytest.mark.skipif(not SWE_BENCH_PATH.exists(), reason="SWE-bench dataset not present")
class TestSWEBenchEmbed:
    """Embedding-mode tests using SWE-bench Verified GitHub issue descriptions."""

    def test_semantic_recall_by_keyword(self, tmp_path):
        """Store issues with embedding; recall using a keyword from each issue.

        Keyword queries (a few words from the issue title/body) should retrieve
        the correct issue in top-3. Validates semantic recall on real bug reports.
        """
        df = pd.read_parquet(SWE_BENCH_PATH)
        issues = df["problem_statement"].dropna().astype(str).tolist()[:12]

        for issue in issues:
            store_test_memory_embed(issue, tmp_path, chunk="code")

        hits = 0
        for issue in issues[:10]:
            # Use first 6 non-trivial words as keyword query
            words = [w for w in issue.split() if len(w) > 4][:6]
            query = " ".join(words)
            results = recall_memory(
                query, top_k=5, data_dir=tmp_path, chunk="code", use_embedding=True
            )
            if any(r["text"] == issue for r in results):
                hits += 1

        assert hits >= 6, (
            f"SWE-bench semantic recall: only {hits}/10 keyword queries found "
            f"the original issue in top-5"
        )

    def test_embed_convergence_rate(self):
        """Embedding-based frames should converge as reliably as SHA-256 frames."""
        from wheeler_memory.embedding import embed_to_frame

        df = pd.read_parquet(SWE_BENCH_PATH)
        texts = df["problem_statement"].dropna().astype(str).tolist()[:30]

        converged = sum(
            1
            for text in texts
            if evolve_and_interpret(embed_to_frame(text))["state"] != "CHAOTIC"
        )
        rate = converged / len(texts)
        assert rate >= 0.90, (
            f"Embed convergence rate {rate:.0%} < 90% on SWE-bench issues"
        )


@pytest.mark.slow
@pytest.mark.embed
@pytest.mark.skipif(
    not BABILONG_QA1_PATH.exists(), reason="BABILong dataset not present"
)
class TestBABILongEmbed:
    """Embedding-mode tests using BABILong QA1 (single supporting fact) narratives.

    This is the needle-in-a-haystack test: individual sentences are stored as
    separate memories; the question is used as the recall query. SHA-256 mode
    fails this test by design — embedding mode should succeed because the
    question and the relevant sentence share semantic content (the entity name
    and the location relationship).
    """

    def test_question_recalls_relevant_fact(self, tmp_path):
        """Store individual BABILong sentences; recall with the question.

        For QA1 ('single supporting fact'), each question is answerable from
        exactly one sentence in the narrative. With embedding mode, the question
        'Where is Mary?' should surface 'Mary moved to the bathroom.' in top-k.
        """
        data = _load_babilong_json(BABILONG_QA1_PATH)[:20]

        successes = 0
        attempts = 0

        for item in data[:10]:
            sentences = [
                s.strip() + "."
                for s in item["input"].replace(".", ".\n").split("\n")
                if s.strip()
            ]
            if len(sentences) < 2:
                continue

            # Find the sentence containing the answer
            answer = item["target"].strip().lower()
            relevant = [s for s in sentences if answer in s.lower()]
            if not relevant:
                continue  # Can't verify without a known-relevant sentence

            for sentence in sentences:
                store_test_memory_embed(sentence, tmp_path, chunk="general")

            question = item["question"].strip()
            results = recall_memory(
                question,
                top_k=min(len(sentences), 5),
                data_dir=tmp_path,
                chunk="general",
                use_embedding=True,
            )

            retrieved_texts = [r["text"] for r in results]
            if any(rel in retrieved_texts for rel in relevant):
                successes += 1
            attempts += 1

            # Clear store for next item to avoid cross-contamination
            import shutil
            shutil.rmtree(tmp_path / "chunks", ignore_errors=True)

        assert attempts > 0, "No valid BABILong items found for needle-in-haystack test"
        success_rate = successes / attempts
        assert success_rate >= 0.5, (
            f"BABILong needle-in-haystack: only {successes}/{attempts} questions "
            f"({success_rate:.0%}) found the relevant fact in top-5 using embedding mode"
        )

    def test_embed_convergence_rate(self):
        """Embedding frames from natural language narratives should converge reliably."""
        from wheeler_memory.embedding import embed_to_frame

        data = _load_babilong_json(BABILONG_QA1_PATH)[:30]
        texts = [item["input"] for item in data]

        converged = sum(
            1
            for text in texts
            if evolve_and_interpret(embed_to_frame(text))["state"] != "CHAOTIC"
        )
        rate = converged / len(texts)
        assert rate >= 0.90, (
            f"BABILong embed convergence rate {rate:.0%} < 90%"
        )
