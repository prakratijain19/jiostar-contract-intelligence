# Requirement checklist

| Requirement | Status | Evidence / verification |
|---|---|---|
| Exact 194-query mini selection and SHA-256 split | Complete | `data.py`; 194 queries selected, train/test 140/54 |
| Only selected contracts indexed | Complete | `cli.build_index`; exactly 29 referenced files |
| UTF-8 handling and every gold offset validated | Complete | `data.validate_spans`; 272 spans verified |
| Download and validation workflow | Complete | `download_data.py`; downloader and local validation tested |
| 30 labeled unanswerable queries, split and pre-experiment commit | Complete | `data/unanswerable.jsonl`, commit `1d36fb1`, counts 22/8 |
| Negative-set semantic validation | Partial | Future audited financial/market questions and requested label phrases are absent from the contracts; no independent legal reviewer |
| BM25 retrieval, configurable chunking, original offsets | Complete | `chunking.py`, `retrieval.py`; offset tests pass |
| Abstention threshold selected on train only | Complete | `evaluation.select_abstention_threshold`; both label groups from train |
| CLI and Streamlit UI | Partial | CLI query/help verified; `app.py` implemented, but Streamlit is not installed in the execution environment |
| Official character precision/recall at multiple k | Complete | `evaluation.py`, k={1,2,4,8,16,32,64}; hand-checked tests |
| Train-only self-improvement loop | Complete | `improve.py`; seven genuine experiments recorded in `experiments/history.json`; test access false |
| Baseline/final train/test evaluation and report | Complete | Four saved metric outputs generated; `REPORT.md` records four test evaluations |
| 10 concrete test-set failure analyses | Complete | Ten real cases and evidence excerpts in `REPORT.md` and `results/failure_analyses.json` |
| Tests | Complete | 13 unit tests pass; all 272 data spans verified |
| Public repository publication | Not done | Requires user-owned GitHub remote and publication step |
