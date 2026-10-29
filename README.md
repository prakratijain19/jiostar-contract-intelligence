# JioStar Contract Intelligence Assistant

## Overview

This project searches the CUAD portion of LegalBench-RAG for contract passages that may answer a natural-language question. It returns ranked evidence excerpts with the source document, original character offsets, and retrieval score. It can abstain when scores fall below a configured threshold. It does not generate legal conclusions or replace review of the full agreement.

## Setup

Requires Python 3.10 or newer. A CPU-only machine is sufficient. No LLM, GPU, or API key is required.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e . -r requirements.txt
```

The project dependencies include Streamlit and pytest. If PowerShell does not allow virtual-environment activation, invoke tools directly, for example `\.venv\Scripts\python.exe -m pytest -q`.

## Download and prepare data

The data command downloads the official CUAD benchmark file from the `awinml/legalbench-rag` Hugging Face mirror, selects the prescribed mini subset, then downloads only the contract files referenced by those selected queries. It validates the answer spans against the source text. The next command materializes split-specific views and a manifest of the selected contract corpus. A separate audit checks that the frozen negative questions' requested fact categories do not occur in the indexed agreements.

```powershell
python -m contract_assistant.download_data --data data/legalbench-rag
python -m contract_assistant.prepare_data --data data/legalbench-rag --negatives data/unanswerable.jsonl
python scripts/validate_unanswerable.py
```

The generated data layout is:

```text
data/legalbench-rag/
  benchmarks/cuad.json
  corpus/cuad/<selected contract>.txt
  derived/selected_documents.json
  derived/splits/train.jsonl
  derived/splits/test.jsonl
  derived/splits/unanswerable_train.jsonl
  derived/splits/unanswerable_test.jsonl
```

The benchmark selection uses the LegalBench-RAG authors' deterministic `mini(tests, n=194)` function: seed Python's `random` with the first snippet's `file_path`, use the next `random.random()` value as the sort key, and keep the first 194 records. Split assignment hashes `query.strip().encode("utf-8")` with SHA-256 and assigns `test` when the integer digest modulo 10 is less than 3. Text is read as UTF-8 in Python text mode so offsets follow the dataset convention. The downloaded corpus is excluded from Git; it is not redistributed in this repository. Review the [dataset card and source licenses](https://huggingface.co/datasets/awinml/legalbench-rag) before reuse.

`data/unanswerable.jsonl` is the frozen negative-query set. Its facts concern corporate or market information outside the contract terms. The repository validator performs a text-category check across the selected corpus; it is not a legal expert review or a formal proof of semantic unanswerability.

## How to run

### CLI

```powershell
python -m contract_assistant.cli "Consider the Amendment No. 3 to Wireless Content License Agreement between Fox Mobile Entertainment, Inc. and Glu Mobile Inc.; What is the expiration date of this contract?" --data data/legalbench-rag --top-k 5 --config configs/final.json
```

The output lists passages with rank, score, source path, and `[start:end)` offsets into the original document. The CLI uses the train-selected abstention threshold from the configuration file by default; pass `--threshold` to override it. A blank question, invalid configuration, or missing data produces an explicit error. If no passage reaches the threshold, it prints a no-results message.

### Streamlit app

```powershell
streamlit run src/contract_assistant/app.py
```

Enter a question and select **Search**. The app shows a loading state, ranked passages, source and offset information, scores, and an explicit no-results state. The sidebar lets you change the dataset path, top-k, chunk size, overlap, and threshold.

## Tests and evaluation

Run the unit tests from the repository root:

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q
```

To reproduce the baseline and final evaluation files, first download and prepare the data as shown above. Run baseline training evaluation before the improvement loop:

```powershell
python -m contract_assistant.run_evaluation --split train --config configs/baseline.json --output results/baseline_train.json
```

Run the self-improvement loop (seven proposals by default):

```powershell
python -m contract_assistant.improve --iterations 7 --log experiments/history.json --final-config configs/final.json
```

The loop uses prepared training queries and training negatives only. It tests BM25 chunk-size/overlap settings and sparse cosine TF-IDF, maximizes training character recall@8, and accepts a proposal only when character precision@8 stays within the declared guardrail. It logs the proposal, rationale, training metrics, failure diagnosis, and decision. It does not load the test split.

After choosing the configuration, run final training and baseline/final test reporting:

```powershell
python -m contract_assistant.run_evaluation --split train --config configs/final.json --output results/final_train.json
python -m contract_assistant.run_evaluation --split test --config configs/baseline.json --output results/baseline_test.json
python -m contract_assistant.run_evaluation --split test --config configs/final.json --output results/final_test.json
python scripts/build_report.py
```

The evaluation command selects its abstention threshold from training positives and negatives only. Test evaluation is for reporting; do not use test metrics to choose a configuration or threshold. Each evaluation JSON stores character-level precision and recall at k = 1, 2, 4, 8, 16, 32, and 64, along with abstention metrics. The metric results, experiment analysis, test-evaluation count, and failure cases belong in `REPORT.md` and the files under `results/`, not in this README. The saved `results/test_evaluation_log.json` records the evaluation history for the report; update that log if you conduct additional test evaluations.

## Architecture and configuration

- `data.py` implements exact mini selection, SHA-256 split assignment, safe path resolution, span validation, and prepared split loading.
- `prepare_data.py` writes separate train/test files and a selected-document manifest before tuning. The improvement loop reads the training files and corpus manifest, not test queries or labels.
- `chunking.py` prefers paragraph and heading boundaries, falls back to character windows for long blocks, and keeps source offsets.
- `retrieval.py` provides BM25 and sparse cosine TF-IDF over the same passages.
- `evaluation.py` implements character-span metrics, abstention metrics, and train-only threshold selection.
- `cli.py` and `app.py` use the shared index builder; `improve.py` runs and logs training-only experiments.

Pipeline settings live in `configs/baseline.json` and `configs/final.json`. Chunk size, overlap, BM25 parameters, top-k, retriever, and abstention threshold are configurable. The system retrieves excerpts only; definitions and cross-references are not resolved automatically.

## Reproducing the saved report

From a fresh clone, install dependencies, run the three data preparation commands, run tests, save baseline training metrics, run the improvement loop, then run final training and baseline/final test evaluations. Finish with `python scripts/build_report.py`. The saved report and machine-readable files are based on actual execution outputs. On the available CPU environment, one evaluation pass took roughly tens of seconds and the seven-proposal loop took a few minutes; network speed affects data setup time.

## Assumptions and limitations

- Scope is limited to the selected CUAD contracts. Data preparation selects the benchmark-referenced corpus; it does not index unrelated files.
- Benchmark questions include contract-identifying wording. Real user questions may not have that context, and lexical retrieval scores are not calibrated confidence.
- Paragraph and heading detection is heuristic. Tables, unusual formatting, long clauses, definitions, and cross-references can reduce retrieval quality.
- The negative set is small, and its text check cannot establish semantic unanswerability for every conceivable question.
- The retrieval tool does not interpret clauses or provide legal advice. Users should inspect the cited passage in its full agreement and consult qualified counsel for legal decisions.

## AI tool usage

OpenAI Codex assisted with repository inspection, implementation, test execution, and documentation. No LLM is used in the application or evaluation pipeline. The code and experiment outputs are included for review; anyone presenting this project should understand and be able to explain the implementation.
