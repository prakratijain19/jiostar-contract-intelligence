import hashlib
import json

import pytest

from contract_assistant.chunking import chunk_document
from contract_assistant.data import Example, assign_split, load_benchmark, load_prepared_split, validate_spans
from contract_assistant.evaluation import query_scores, span_metrics
from contract_assistant.improve import accept_proposal
from contract_assistant.retrieval import BM25Index, SearchResult, TfidfIndex


def test_mini_is_exact_and_deterministic():
    records = [{"query": str(i), "snippets": [{"file_path": f"doc-{i % 20}"}]} for i in range(250)]
    expected = sorted(records, key=lambda item: ( __import__("random").seed(item["snippets"][0]["file_path"]), __import__("random").random())[1])[:194]
    assert load_mini(records) == expected
    assert load_mini(records) == expected


def load_mini(records):
    from contract_assistant.data import mini
    return mini(records)


def test_split_matches_sha256_specification():
    query = "  question with whitespace  "
    expected = "test" if int(hashlib.sha256(query.strip().encode()).hexdigest(), 16) % 10 < 3 else "train"
    assert assign_split(query) == expected
    assert assign_split(query) == assign_split(query)


def test_chunk_offsets_map_back_exactly():
    text = "Heading\n\n" + ("Alpha clause describes rights and obligations. " * 60)
    chunks = chunk_document("contract.txt", text, max_chars=160, overlap=20)
    assert chunks
    for chunk in chunks:
        assert text[chunk.start:chunk.end] == chunk.text
    assert chunks[0].start == 0
    assert chunks[-1].end == len(text)


def test_chunk_rejects_bad_configuration():
    with pytest.raises(ValueError):
        chunk_document("x", "body", max_chars=10, overlap=10)


def test_bm25_ranks_matching_passage_and_preserves_offsets():
    documents = {"a.txt": "Sublicensing rights allow distribution to third party platforms.", "b.txt": "Payment is due within thirty days."}
    index = BM25Index.from_documents(documents, max_chars=100, overlap=10)
    hits = index.search("Can we sublicense content to a third-party platform?", top_k=2)
    assert hits[0].document_id == "a.txt"
    assert hits[0].text == documents["a.txt"]
    assert hits[0].start == 0 and hits[0].end == len(documents["a.txt"])
    assert hits[0].rank == 1 and hits[0].score > 0


def test_abstains_for_empty_or_unmatched_queries():
    index = BM25Index.from_documents({"x.txt": "Payment is due in thirty days."}, max_chars=100, overlap=10)
    assert index.search("  ") == []
    assert index.search("galactic neutrino telemetry", threshold=0.0) == []


def test_tfidf_uses_same_ranked_offset_result_contract():
    index = TfidfIndex.from_documents({"a.txt": "Sublicensing rights include digital platform distribution.",
                                      "b.txt": "Payment deadlines are thirty days."}, max_chars=80, overlap=10)
    hit = index.search("platform distribution rights", top_k=1)[0]
    assert hit.document_id == "a.txt"
    assert hit.rank == 1 and hit.score > 0
    assert hit.text == "Sublicensing rights include digital platform distribution."


def test_character_span_metrics_hand_checked():
    precision, recall = span_metrics([(10, 20)], [(8, 15), (14, 22)])
    assert precision == pytest.approx(10 / 14)
    assert recall == pytest.approx(1.0)
    assert span_metrics([(0, 4)], [])[1] == 0


def test_official_character_metrics_count_wrong_document_chars_in_precision():
    example = Example("question", [{"file_path": "gold.txt", "span": [10, 20], "answer": "0123456789"}], "train")
    hits = [SearchResult("wrong.txt", "x" * 10, 0, 10, 2.0, 1),
            SearchResult("gold.txt", "x" * 10, 10, 20, 1.0, 2)]
    scores = query_scores(example, hits, (2,))
    assert scores["char_precision@2"] == pytest.approx(.5)
    assert scores["char_recall@2"] == pytest.approx(1.0)


def test_annotation_offset_validation(tmp_path):
    root = tmp_path
    (root / "corpus" / "cuad").mkdir(parents=True)
    (root / "corpus" / "cuad" / "doc.txt").write_text("prefix answer suffix", encoding="utf-8")
    example = Example("question", [{"file_path": "cuad/doc.txt", "span": [7, 13], "answer": "answer"}], "train")
    assert validate_spans([example], root) == 1
    invalid = Example("question", [{"file_path": "cuad/doc.txt", "span": [7, 13], "answer": "wrong!"}], "train")
    with pytest.raises(ValueError, match="Gold span mismatch"):
        validate_spans([invalid], root)


def test_training_loader_never_opens_test_split(tmp_path):
    split_dir = tmp_path / "derived" / "splits"
    split_dir.mkdir(parents=True)
    (split_dir / "train.jsonl").write_text(json.dumps({"query": "train only", "snippets": [], "split": "train"}) + "\n", encoding="utf-8")
    (split_dir / "test.jsonl").write_text("this intentionally is not valid JSON\n", encoding="utf-8")
    loaded = load_prepared_split(tmp_path, "train")
    assert len(loaded) == 1 and loaded[0].query == "train only"


def test_improvement_loop_acceptance_and_rejection():
    baseline = {"char_recall@8": .5, "char_precision@8": .4}
    assert accept_proposal({"char_recall@8": .51, "char_precision@8": .39}, baseline)
    assert not accept_proposal({"char_recall@8": .6, "char_precision@8": .37}, baseline)
    assert not accept_proposal({"char_recall@8": .49, "char_precision@8": .5}, baseline)


def test_cli_help_runs(capsys):
    from contract_assistant.cli import main
    with pytest.raises(SystemExit) as exc:
        import sys
        old = sys.argv
        try:
            sys.argv = ["contract-search", "--help"]
            main()
        finally:
            sys.argv = old
    assert exc.value.code == 0


