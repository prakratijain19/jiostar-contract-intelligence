# JioStar Contract Intelligence Assistant

A local retrieval demo for finding evidence passages in CUAD contracts. It returns the source document, original character offsets, rank, and BM25 score. It does not draft legal conclusions. Search results are evidence for human review.

## Setup

Requires Python 3.10 or newer. A CPU-only laptop is sufficient; the selected corpus is small (29 contracts for this exact mini benchmark). No model API key or GPU is needed.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pytest
```

The Streamlit dependency is declared in `pyproject.toml`. `requirements.txt` is an alternative package list.

## Data

The downloader gets the CUAD benchmark from the `awinml/legalbench-rag` Hugging Face dataset mirror and downloads only documents selected by the official mini function. It validates all gold offsets before returning success.

```powershell
python -m contract_assistant.download_data --data data/legalbench-rag
python -m contract_assistant.prepare_data --data data/legalbench-rag --negatives data/unanswerable.jsonl
python scripts/validate_unanswerable.py
```

Expected paths are `data/legalbench-rag/benchmarks/cuad.json` and `data/legalbench-rag/corpus/cuad/*.txt`. The full 4,042-query benchmark file is downloaded; the program selects exactly 194 queries, then resolves and indexes only their referenced contracts. `prepare_data` materializes separate train/test files and a document manifest before experiments; the improvement loop reads only the train file, train negatives, and this contract corpus manifest. Source data is ignored by Git and is not redistributed here. The dataset has per-source licensing; consult its [dataset card](https://huggingface.co/datasets/awinml/legalbench-rag) before reuse.

Selection is the LegalBench-RAG authors' `mini(tests, n=194)` function: for each test, seed Python's `random` with the first snippet's `file_path`, use the next `random.random()` as the sort key, and keep the first 194. The split hashes `query.strip().encode("utf-8")` with SHA-256 and labels `test` iff the integer digest modulo 10 is below 3. This run selected 194 queries, 29 documents, and 272 spans; the query split is 140 train / 54 test. All span offsets were verified against the corpus.

The frozen negative query set is `data/unanswerable.jsonl` (30 items; 22 train / 8 test) and was committed before the improvement experiments. These ask for post-contract audited revenue and market capitalization data not stated in the agreements. This is a small, manually constructed negative set; it is not a substitute for a larger legal review of answerability.

## Run the system

```powershell
python -m contract_assistant.cli "Consider the Amendment No. 3 to Wireless Content License Agreement between Fox Mobile Entertainment, Inc. and Glu Mobile Inc.; What is the expiration date of this contract?" --data data/legalbench-rag --top-k 5 --config configs/final.json
streamlit run src/contract_assistant/app.py
```

The CLI prints passages with source, `[start:end)` offsets, rank, and score. Blank questions and missing or malformed data produce explicit messages. When no passage reaches the configured threshold, both interfaces display an explicit no-results message. By default, the CLI and app load the train-selected threshold from `configs/final.json`; `--threshold` overrides it. Set a threshold of `0` to accept any lexical match. Evaluation reselects a threshold on train positives and negatives only.

## Tests and evaluation

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q
```

Evaluate the baseline on train:

```powershell
python -m contract_assistant.run_evaluation --split train --config configs/baseline.json --output results/baseline_train.json
```

Run at least five genuine training-only iterations (default seven here):

```powershell
python -m contract_assistant.improve --iterations 7 --log experiments/history.json --final-config configs/final.json
```

The loop compares chunk-size/overlap variants and a sparse cosine TF-IDF alternative against the incumbent on training character recall@8. It accepts only if recall rises and character precision@8 drops by no more than 0.02. It records failure categories, metrics, changes, and decisions. Abstention threshold selection uses only train positives and train negatives. The loop never loads the test queries or labels.

Final reporting commands (test set is evaluated once per command, after tuning):

```powershell
python -m contract_assistant.run_evaluation --split train --config configs/final.json --output results/final_train.json
python -m contract_assistant.run_evaluation --split test --config configs/baseline.json --output results/baseline_test.json
python -m contract_assistant.run_evaluation --split test --config configs/final.json --output results/final_test.json
python scripts/build_report.py
```

The test commands above are for final reporting only; no test value informs threshold or configuration choice. Test evaluations are recorded in `REPORT.md`. Each JSON output stores character precision/recall at k âˆˆ {1, 2, 4, 8, 16, 32, 64}. Official character precision divides overlapping gold characters by all returned characters, including wrong-document passages. Recall divides by all gold characters. Results are macro-averaged over queries in the CUAD split. Abstention metrics are reported separately.

## Architecture and configuration

`data.py` implements mini selection, split assignment, safe path resolution, and strict span validation. `chunking.py` prefers paragraph/heading boundaries and keeps half-open offsets into the original document. `retrieval.py` implements lexical BM25. `evaluation.py` scores character spans and selects an abstention threshold from training data. `cli.py` and `app.py` share the same index builder. `improve.py` runs reproducible training-only parameter experiments. The pipeline has no LLM or generation stage; deterministic retrieval is cheaper, easier to inspect, and avoids sending contracts to a hosted model.

Chunk length and overlap are in the JSON files under `configs/`. BM25 k1 and b are exposed in the Python API. Indexing the 29 selected contracts and scoring 140 train questions takes tens of seconds on this environment; expect runtime to vary with CPU and Python version.

## Assumptions and limits

- CUAD is the only scope. The 29 contracts are the documents referenced by the selected 194 queries.
- Query title phrases remain in the retrieval input, exactly as benchmark text provides them.
- Clause-aware boundaries are heuristic. Definitions and cross-references are not resolved.
- No LLM is used. The tool only retrieves evidence; users must inspect full contracts and seek counsel for legal decisions.
- Abstention threshold optimization uses a small negative set, and search score is not calibrated confidence.
- AI tool usage: OpenAI Codex assisted with repository inspection, implementation, test execution, and documentation. All code and experimental claims are available for review; the author should understand the pipeline before submission.

## Reproducing the saved report

Start with the download command, run tests, save baseline train results, run the seven-iteration improvement command, then run final train and the two final-reporting test commands exactly as listed. `REPORT.md` records command outputs and the final test-evaluation count. Benchmark corpus files and the confidential assignment PDF are not committed.




