# Evaluation Report — JioStar Contract Intelligence Assistant

## 1. Executive summary

The final BM25 configuration achieved character recall@8 of **20.69%** and precision@8 of **1.02%** on the 54-query test split. The baseline measured 11.33% recall@8 and 0.69% precision@8. The final system also correctly abstained on 8/8 test negatives, with 100% answerable coverage. These abstention results are based on eight test negatives and should be treated as preliminary.

The retriever is a small lexical system, not a legal reasoning or drafting tool. Its recall remains low, and every returned excerpt requires review in its full contract context.

## 2. System design and alternatives

The system selects the authors' 194-query CUAD mini subset, loads only its 29 referenced contracts, validates the gold spans, chunks each document while preserving source offsets, then ranks passages. BM25 is the selected method. A sparse cosine TF-IDF implementation was also evaluated on training data and rejected because its recall@8 was lower than the incumbent. Dense retrieval and reranking were not attempted: the corpus fits a simple lexical index, no embedding service or model weights were configured, and the priority was an inspectable offline baseline.

Chunk boundaries prefer paragraph breaks and likely headings, then split long blocks with overlap. The CLI and Streamlit interface share the same index builder. The system returns passages only; it does not synthesize unsupported legal conclusions.

## 3. LLM usage

No LLM is used at indexing, query time, evaluation, or in the improvement loop. This removes API cost and latency and avoids sending contract text to an external provider. Query rewriting and reranking remain future experiments; they would need explicit privacy controls and cached outputs before evaluation.

## 4. Evaluation method and sample sizes

The benchmark mirror contained 4,042 CUAD tests. The exact authors' seeded mini-selection function returned 194 queries referencing 29 contracts. The prescribed SHA-256 query split produced 140 train and 54 test queries. We validated all 272 selected answer spans using `document[start:end] == answer` after UTF-8 text-mode reading.

The frozen unanswerable set has 30 queries, split into 22 train and 8 test with the same hash function. These ask for future audited revenue or market capitalization values not stated in the dated contract corpus. A phrase scan confirmed the requested data labels are absent. This set is intentionally small and lacks independent attorney review.

Official character precision counts all returned characters in its denominator, including wrong-document passages; its numerator counts character overlap with gold spans in the correct document. Recall divides the same overlap by all gold characters. Scores are macro-averaged by query. We report k ∈ {1, 2, 4, 8, 16, 32, 64}. Abstention threshold maximizes balanced accuracy on train positives and negatives only. The test set was scored **4 times**: an initial baseline/final reporting pair, then both configurations were evaluated again after the train-only TF-IDF comparison and split-isolation refactor. The loop and all acceptance decisions use prepared train files only; no test score or label enters tuning.

## 5. Baseline versus final

The primary selection metric was mean official character recall@8. Precision@8 could not decrease by more than 0.02 absolute from the incumbent. The selected final setting is BM25, 3,000-character chunks, 300-character overlap, k1=1.5, b=0.75. Its abstention threshold, 28.5844, was selected from train only.

| k | Baseline train P/R | Final train P/R | Baseline test P/R | Final test P/R |
|---:|---:|---:|---:|---:|
| 1 | 0.51% / 3.06% | 1.43% / 10.12% | 0.31% / 2.56% | 0.36% / 5.56% |
| 2 | 1.02% / 7.13% | 2.28% / 19.83% | 0.76% / 5.77% | 0.61% / 10.62% |
| 4 | 1.37% / 13.07% | 2.21% / 25.10% | 0.56% / 9.48% | 0.75% / 15.29% |
| 8 | 1.31% / 14.18% | 2.11% / 26.10% | 0.69% / 11.33% | 1.02% / 20.69% |
| 16 | 1.31% / 14.18% | 2.11% / 26.10% | 0.69% / 11.33% | 1.02% / 20.69% |
| 32 | 1.31% / 14.18% | 2.11% / 26.10% | 0.69% / 11.33% | 1.02% / 20.69% |
| 64 | 1.31% / 14.18% | 2.11% / 26.10% | 0.69% / 11.33% | 1.02% / 20.69% |

Final train recall@8 is 26.10%; test recall@8 is 20.69%, a 5.41 percentage-point gap. Final train precision@8 is 2.11%; test precision@8 is 1.02%. The test scores are lower, and the shortfall is consistent with lexical matching finding a relevant document but often missing the annotated clause.

| Split/config | Answerable coverage | Incorrect abstentions | Correct negative abstentions | False acceptance rate |
|---|---:|---:|---:|---:|
| Train baseline | 100.0% | 0.0% | 100.0% (22/22) | 0.0% |
| Train final | 100.0% | 0.0% | 100.0% (22/22) | 0.0% |
| Test baseline | 100.0% | 0.0% | 100.0% (8/8) | 0.0% |
| Test final | 100.0% | 0.0% | 100.0% (8/8) | 0.0% |

Complete machine-readable metric files are in `results/`. No metric is estimated or filled in when output is absent.

## 6. Objective and full experiment history

Objective and guardrail were written in the loop before its experiments: maximize train character recall@8, while allowing at most a 0.02 absolute loss in train precision@8. Each change rebuilt and evaluated the index on all 140 train questions, tuned the abstention threshold using 140 train positives and 22 train negatives, and recorded a train-only failure diagnosis. The loop does not load test queries.

| Iteration | Proposal source | Configuration and rationale | Train P@8 | Train R@8 | Decision |
|---:|---|---|---:|---:|---|
| 0 | fixed baseline | bm25, 1800 chars / 180 overlap — BM25 baseline | 1.31% | 14.18% | baseline |
| 1 | predeclared parameter experiment | bm25, 1200 chars / 120 overlap — Smaller passages may improve local evidence focus. | 0.95% | 8.02% | rejected |
| 2 | predeclared parameter experiment | bm25, 2400 chars / 240 overlap — Larger passages may cover long annotated clauses. | 1.96% | 25.67% | accepted |
| 3 | predeclared parameter experiment | bm25, 1800 chars / 0 overlap — No overlap tests whether duplicate spans dilute precision. | 1.16% | 13.44% | rejected |
| 4 | predeclared parameter experiment | bm25, 1800 chars / 360 overlap — More overlap may preserve evidence near boundaries. | 1.15% | 13.06% | rejected |
| 5 | predeclared parameter experiment | bm25, 900 chars / 180 overlap — Shorter passages test fine-grained retrieval. | 0.46% | 1.80% | rejected |
| 6 | predeclared parameter experiment | bm25, 3000 chars / 300 overlap — Long passages test contextual clause retrieval. | 2.11% | 26.10% | accepted |
| 7 | predeclared parameter experiment | tfidf, 3000 chars / 300 overlap — Compare sparse cosine TF-IDF with BM25 on the same train split. | 1.65% | 23.46% | rejected |

The accepted improvements were 2,400/240 and 3,000/300 BM25 chunk/overlap settings. The other five experiments, including TF-IDF, were rejected because they missed the recall objective or precision guardrail. The full log contains all k metrics, thresholds, proposal descriptions, and diagnostic counts in `experiments/history.json`.

## 7. Test-set failure analysis

The following ten cases are drawn from the final test output. IDs are the stable one-based order in the selected mini list because LegalBench-RAG records have no query ID. The expected evidence is shown with original offsets. Retrieved text is abbreviated only for readability; full passages are in `results/failure_analyses.json`.

### Correct document, wrong passage

**CUAD-mini-004** (final recall@4 0.000)

- Query: Consider the Service Agreement between Theravance Biopharma UK Limited and Brett Haumann; Does this contract provide for joint intellectual property ownership?
- Expected: `cuad/THERAVANCEBIOPHARMA,INC_05_08_2020-EX-10.2-SERVICE AGREEMENT.txt[16945:17247]` — To the extent that the Company IP does not vest automatically in the Company the Executive hereby assigns all right, title and interest in the Company IP to the Company with full title guarantee by way of a present assignment of all future rights and shall oth…
- Retrieved top result: `cuad/THERAVANCEBIOPHARMA,INC_05_08_2020-EX-10.2-SERVICE AGREEMENT.txt[0:2881]`, score 49.286 — Exhibit 10.2  STRICTLY PRIVATE AND CONFIDENTIAL  1 April, 2020  THERAVANCE BIOPHARMA UK LIMITED and BRETT HAUMANN  SERVICE AGREEMENT      THIS AGREEMENT is entered into between the parties on 1 April, 2020.  PARTIES (1) Theravance Biopharma UK Limited is a com…
- Diagnosis: title and topic terms brought a passage from the correct contract into the top four, but the returned characters did not include the annotated span; where recall is nonzero, the passage captured only part of the expected evidence.
- Potential fix: test clause-aware sentence windows and a train-tuned passage reranker that downweights repeated title/header language. This is a proposal, not a measured result.

**CUAD-mini-005** (final recall@4 0.000)

- Query: Consider the Sponsorship Agreement between R. C. Boyd Enterprises, LLC and Cano Petroleum, Inc. for 'Honey Hole'; What is the governing law for this contract?
- Expected: `cuad/CANOPETROLEUM,INC_12_13_2007-EX-10.1-Sponsorship Agreement.txt[4159:4489]` — This Agreement shall be governed by and construed in accordance with the laws of the State of Texas and venue for  any legal action brought in State Court shall lie exclusively in Tarrant County, Texas and venue for any action brought in federal court shall  l…
- Retrieved top result: `cuad/CANOPETROLEUM,INC_12_13_2007-EX-10.1-Sponsorship Agreement.txt[0:2872]`, score 57.478 — QuickLinks -- Click here to rapidly navigate through this document      EXHIBIT 10.1  Sponsorship Agreement  THIS AGREEMENT made this 5th day of December, 2007, between R. C. Boyd Enterprises, LLC, a Texas limited liability company, whose principal  place of b…
- Diagnosis: title and topic terms brought a passage from the correct contract into the top four, but the returned characters did not include the annotated span; where recall is nonzero, the passage captured only part of the expected evidence.
- Potential fix: test clause-aware sentence windows and a train-tuned passage reranker that downweights repeated title/header language. This is a proposal, not a measured result.

**CUAD-mini-007** (final recall@4 0.000)

- Query: Consider the Sponsorship Agreement between Excite, Inc. and Vitamin Shoppe Industries Inc. for Promotion on WebCrawler Channels; Is there uncapped liability under this contract?
- Expected: `cuad/VITAMINSHOPPECOMINC_09_13_1999-EX-10.26-SPONSORSHIP AGREEMENT.txt[54279:54600]` — EXCEPT UNDER SECTIONS 13(a) AND 13(b), THE LIABILITY OF EITHER                  PARTY FOR DAMAGES OR ALLEGED DAMAGES HEREUNDER, WHETHER IN                  CONTRACT, TORT OR ANY OTHER LEGAL THEORY, IS LIMITED TO, AND                  WILL NOT EXCEED, THE AMOUN…
- Retrieved top result: `cuad/VITAMINSHOPPECOMINC_09_13_1999-EX-10.26-SPONSORSHIP AGREEMENT.txt[0:2534]`, score 49.098 — 1                                                                 EXHIBIT 10.26                           CONFIDENTIAL MATERIALS OMITTED AND FILED SEPARATELY                                     WITH THE SECURITIES AND EXCHANGE COMMISSION.                      …
- Diagnosis: title and topic terms brought a passage from the correct contract into the top four, but the returned characters did not include the annotated span; where recall is nonzero, the passage captured only part of the expected evidence.
- Potential fix: test clause-aware sentence windows and a train-tuned passage reranker that downweights repeated title/header language. This is a proposal, not a measured result.

**CUAD-mini-008** (final recall@4 0.000)

- Query: Consider the Strategic Alliance Agreement between Information System Associates, Inc. and Rubicon Software Group plc; Is there a non-compete clause in this contract?
- Expected: `cuad/DUOSTECHNOLOGIESGROUP,INC_04_21_2009-EX-10.1-STRATEGIC ALLIANCE AGREEMENT.txt[20426:21063]` — During the period beginning on the Effective Date and ending on the later of (i) the end of the Initial Term and (ii) the date falling two  years after completion of the most recent services provided by Rubicon under clause 7 (the "Non-Competition Period") and…
- Expected: `cuad/DUOSTECHNOLOGIESGROUP,INC_04_21_2009-EX-10.1-STRATEGIC ALLIANCE AGREEMENT.txt[21989:22639]` — During the period beginning on the Effective Date and ending on the later of (i) the end of the Initial Term and (ii) the date falling two  years after completion of the most recent services provided by Rubicon under clause 7 (the "Non-Competition Period")  an…
- Retrieved top result: `cuad/DUOSTECHNOLOGIESGROUP,INC_04_21_2009-EX-10.1-STRATEGIC ALLIANCE AGREEMENT.txt[0:2999]`, score 46.797 — Exhibit 10.1    STRATEGIC ALLIANCE AGREEMENT    This Strategic Alliance Agreement (this "Agreement"), effective as of 15 April, 2009 (the "Effective Date"), is by and between Information  System Associates, Inc., a Florida Corporation whose registered office i…
- Diagnosis: title and topic terms brought a passage from the correct contract into the top four, but the returned characters did not include the annotated span; where recall is nonzero, the passage captured only part of the expected evidence.
- Potential fix: test clause-aware sentence windows and a train-tuned passage reranker that downweights repeated title/header language. This is a proposal, not a measured result.

**CUAD-mini-009** (final recall@4 0.000)

- Query: Consider the Strategic Alliance Agreement between Information System Associates, Inc. and Rubicon Software Group plc; Is there an anti-assignment clause in this contract?
- Expected: `cuad/DUOSTECHNOLOGIESGROUP,INC_04_21_2009-EX-10.1-STRATEGIC ALLIANCE AGREEMENT.txt[36711:37219]` — Neither Party shall transfer, assign or cede any rights or delegate any obligations hereunder, in whole or in part,  whether voluntarily or by operation of law, without the prior written consent of the other Party, which consent may be withheld at the other Pa…
- Retrieved top result: `cuad/DUOSTECHNOLOGIESGROUP,INC_04_21_2009-EX-10.1-STRATEGIC ALLIANCE AGREEMENT.txt[0:2999]`, score 47.312 — Exhibit 10.1    STRATEGIC ALLIANCE AGREEMENT    This Strategic Alliance Agreement (this "Agreement"), effective as of 15 April, 2009 (the "Effective Date"), is by and between Information  System Associates, Inc., a Florida Corporation whose registered office i…
- Diagnosis: title and topic terms brought a passage from the correct contract into the top four, but the returned characters did not include the annotated span; where recall is nonzero, the passage captured only part of the expected evidence.
- Potential fix: test clause-aware sentence windows and a train-tuned passage reranker that downweights repeated title/header language. This is a proposal, not a measured result.

**CUAD-mini-010** (final recall@4 0.000)

- Query: Consider the Strategic Alliance Agreement between Information System Associates, Inc. and Rubicon Software Group plc; Does this contract include any revenue or profit-sharing arrangements?
- Expected: `cuad/DUOSTECHNOLOGIESGROUP,INC_04_21_2009-EX-10.1-STRATEGIC ALLIANCE AGREEMENT.txt[11393:11877]` — In addition to the payments set forth in sections 4.2.1 and 4.2.2, above, ISA will pay Rubicon 30% of ISA's profits (defined as  gross revenues less costs directly incurred in the generation of such revenues) on projects for which Rubicon has provided Contract…
- Retrieved top result: `cuad/DUOSTECHNOLOGIESGROUP,INC_04_21_2009-EX-10.1-STRATEGIC ALLIANCE AGREEMENT.txt[0:2999]`, score 47.972 — Exhibit 10.1    STRATEGIC ALLIANCE AGREEMENT    This Strategic Alliance Agreement (this "Agreement"), effective as of 15 April, 2009 (the "Effective Date"), is by and between Information  System Associates, Inc., a Florida Corporation whose registered office i…
- Diagnosis: title and topic terms brought a passage from the correct contract into the top four, but the returned characters did not include the annotated span; where recall is nonzero, the passage captured only part of the expected evidence.
- Potential fix: test clause-aware sentence windows and a train-tuned passage reranker that downweights repeated title/header language. This is a proposal, not a measured result.

**CUAD-mini-011** (final recall@4 0.000)

- Query: Consider the Strategic Alliance Agreement between Information System Associates, Inc. and Rubicon Software Group plc; Are there any services to be provided after the termination of this contract?
- Expected: `cuad/DUOSTECHNOLOGIESGROUP,INC_04_21_2009-EX-10.1-STRATEGIC ALLIANCE AGREEMENT.txt[12612:12792]` — Each Party shall retain the financial records relating to all payments owed and/or paid under this Agreement for a period of six  years from the date such payment obligation arose.
- Retrieved top result: `cuad/DUOSTECHNOLOGIESGROUP,INC_04_21_2009-EX-10.1-STRATEGIC ALLIANCE AGREEMENT.txt[0:2999]`, score 50.319 — Exhibit 10.1    STRATEGIC ALLIANCE AGREEMENT    This Strategic Alliance Agreement (this "Agreement"), effective as of 15 April, 2009 (the "Effective Date"), is by and between Information  System Associates, Inc., a Florida Corporation whose registered office i…
- Diagnosis: title and topic terms brought a passage from the correct contract into the top four, but the returned characters did not include the annotated span; where recall is nonzero, the passage captured only part of the expected evidence.
- Potential fix: test clause-aware sentence windows and a train-tuned passage reranker that downweights repeated title/header language. This is a proposal, not a measured result.

**CUAD-mini-012** (final recall@4 0.000)

- Query: Consider the Strategic Alliance Agreement between Information System Associates, Inc. and Rubicon Software Group plc; Is there uncapped liability under this contract?
- Expected: `cuad/DUOSTECHNOLOGIESGROUP,INC_04_21_2009-EX-10.1-STRATEGIC ALLIANCE AGREEMENT.txt[32028:32397]` — EXCEPT IN CONNECTION WITH A BREACH BY EITHER PARTY OF SECTION 8, NEITHER PARTY WILL  BE LIABLE FOR ANY SPECIAL, INDIRECT, CONSEQUENTIAL, EXEMPLARY, PUNITIVE OR INCIDENTAL DAMAGES ARISING OUT OF OR  RELATED TO THIS AGREEMENT, HOWEVER CAUSED AND UNDER ANY THEORY…
- Expected: `cuad/DUOSTECHNOLOGIESGROUP,INC_04_21_2009-EX-10.1-STRATEGIC ALLIANCE AGREEMENT.txt[32504:32693]` — EXCEPT IN CONNECTION WITH A BREACH BY EITHER PARTY OF SECTION 8, EACH PARTY'S LIABILITY FOR DAMAGES HEREUNDER (OTHER THAN IN RESPECT OF ANY CLAIM FOR MONIES DUE)  SHALL NOT EXCEED £200,000.
- Retrieved top result: `cuad/DUOSTECHNOLOGIESGROUP,INC_04_21_2009-EX-10.1-STRATEGIC ALLIANCE AGREEMENT.txt[0:2999]`, score 47.205 — Exhibit 10.1    STRATEGIC ALLIANCE AGREEMENT    This Strategic Alliance Agreement (this "Agreement"), effective as of 15 April, 2009 (the "Effective Date"), is by and between Information  System Associates, Inc., a Florida Corporation whose registered office i…
- Diagnosis: title and topic terms brought a passage from the correct contract into the top four, but the returned characters did not include the annotated span; where recall is nonzero, the passage captured only part of the expected evidence.
- Potential fix: test clause-aware sentence windows and a train-tuned passage reranker that downweights repeated title/header language. This is a proposal, not a measured result.


### Partial evidence coverage

**CUAD-mini-018** (final recall@4 0.734)

- Query: Consider the Marketing and Transportation Services Agreement between Purolator Courier Ltd. and Parcelway Courier Systems Canada Ltd. (Dynamex Inc.); What happens in the event of a change of control of one of the parties in this contract?
- Expected: `cuad/DYNAMEXINC_06_06_1996-EX-10.4-TRANSPORTATION SERVICES AGREEMENT.txt[33342:33500]` — In the event of a change in control of a Party, the other Party shall         have the right, upon written prior notice, to terminate this          Agreement.
- Expected: `cuad/DYNAMEXINC_06_06_1996-EX-10.4-TRANSPORTATION SERVICES AGREEMENT.txt[47980:48417]` — In the event this Agreement is terminated pursuant to the provisions of         paragraph 19.2, then the Party whose control has changed shall not         enter into an agreement with any other Party to provide services similar         to those provided herein…
- Retrieved top result: `cuad/DYNAMEXINC_06_06_1996-EX-10.4-TRANSPORTATION SERVICES AGREEMENT.txt[0:2906]`, score 72.849 — 1                                                                     EXHIBIT 10.4               MARKETING AND TRANSPORTATION SERVICES AGREEMENT    THIS AGREEMENT made in duplicate this 20 day of November, 1995.  B E T W E E N:                           PUROLA…
- Diagnosis: title and topic terms brought a passage from the correct contract into the top four, but the returned characters did not include the annotated span; where recall is nonzero, the passage captured only part of the expected evidence.
- Potential fix: test clause-aware sentence windows and a train-tuned passage reranker that downweights repeated title/header language. This is a proposal, not a measured result.

**CUAD-mini-034** (final recall@4 0.523)

- Query: Consider the Endorsement Agreement between ABG-Shaq, LLC and Papa John's International, Inc.; Are there any services to be provided after the termination of this contract?
- Expected: `cuad/PapaJohnsInternationalInc_20190617_8-K_EX-10.1_11707365_EX-10.1_Endorsement Agreement.txt[47413:48223]` — Upon expiration or termination of this Agreement by PAPA JOHN'S (but not in the event of termination by ABG), and subject to PAPA JOHN'S ongoing compliance with the terms and conditions of this Agreement, PAPA JOHN'S shall have the following rights to use the …
- Expected: `cuad/PapaJohnsInternationalInc_20190617_8-K_EX-10.1_11707365_EX-10.1_Endorsement Agreement.txt[51519:52259]` — If this Agreement is terminated by ABG for any of the reasons provided in Section 7.B. above, then CELEBRITY shall be entitled to immediately vest in all of the RSUs for the eighteen (18) months following the effective date of termination (including, without l…
- Retrieved top result: `cuad/PapaJohnsInternationalInc_20190617_8-K_EX-10.1_11707365_EX-10.1_Endorsement Agreement.txt[0:2988]`, score 37.170 — Exhibit 10.1  ENDORSEMENT AGREEMENT  THIS ENDORSEMENT AGREEMENT ("Agreement") is made and entered into effective March 15, 2019 (the "Effective Date") by and among, on the one hand, ABG-Shaq, LLC, a Delaware limited liability company ("ABG") for the personal s…
- Diagnosis: title and topic terms brought a passage from the correct contract into the top four, but the returned characters did not include the annotated span; where recall is nonzero, the passage captured only part of the expected evidence.
- Potential fix: test clause-aware sentence windows and a train-tuned passage reranker that downweights repeated title/header language. This is a proposal, not a measured result.

## 8. Limitations and next steps

- Character recall remains 20.69% at k=8 on test; the system is not suitable for unaided legal review.
- The negative set has only 30 examples (8 test) and its labels are not independently reviewed by legal counsel; perfect observed abstention does not establish reliable answerability detection.
- The lexical score is not a calibrated probability. The selected threshold is corpus- and dataset-specific.
- Chunking uses simple paragraph and heading patterns. Definitions, cross-references, tables, OCR errors, and long clauses can still fragment evidence.
- The benchmark contains public historical contracts and one gold document per question. It does not test confidential JioStar agreements, multi-contract reasoning, current agreements, or answer generation.
- Before real legal use: expand and attorney-review unanswerable cases; evaluate on consented, representative internal contracts; improve clause segmentation and cross-reference handling; test hybrid/dense retrieval and reranking on train only; calibrate abstention against adjudicated errors; add audit trails, access controls, and privacy review; then conduct a prospective legal-user study.

## Reproduction provenance

The baseline/final train and test JSON files are actual command outputs. `test_set_evaluated` is true only in `results/baseline_test.json` and `results/final_test.json`. The result generator used those saved test outputs to prepare this report and did not re-run retrieval. The frozen query dataset was committed before any experiment (`1d36fb1`) and has not been changed.

AI tools assisted with code authoring, test fixes, and report generation; the implementation and all measured results are inspectable in this repository.
