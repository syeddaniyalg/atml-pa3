# CS6304 Programming Assignment 3: Fine-Tuning, Visual Grounding, and Test-Time Compute

This repository contains the code, configurations, split files, saved results, and report source for Programming Assignment 3. It covers three tasks:

1. **Task 1.** Mathematical fine-tuning of Qwen3-0.6B with Full FT, Full FT with L2-SP, LoRA (ranks 4 and 16), and OPLoRA, evaluated for adaptation and retention.
2. **Task 2.** A SigLIP-Qwen vision-language model trained in three stages, evaluated for visual grounding with complementary image pairs. The optional 49-token experiment is included.
3. **Task 3.** Test-time compute: answer selection over supplied rollouts, a live comparison of parallel sampling and refinement, and adaptive stopping.

All experiments use seed 6304.

## 1. Repository layout

```
common/        Shared utilities: paths, logging, LoRA and OPLoRA modules, job launcher, downloads
task1/         Task 1 code: data, training, evaluation, diagnostics, qualitative analysis, tables and figures
task2/         Task 2 code: data construction, feature caching, model, training, evaluation, analysis
task3/         Task 3 code: data loading, answer grading, selection, adaptive stopping, live generation
notebooks/     One notebook per task used to run the experiments on Kaggle (Task 1) and Google Colab (Tasks 2 and 3)
splits/        Saved split and example identifiers
results/       Saved machine-readable results, tables, figures, logs, and per-example outputs
paper/         LaTeX source of the report (NeurIPS 2026 format)
requirements.txt
```

## 2. Environment

The code runs on Python 3 with the packages in `requirements.txt`. PyTorch is not pinned because it is supplied by the platform. A CUDA GPU is required for training and for generation.

```bash
pip install -r requirements.txt
```

Versions used for the reported results:

| Component | Task 1 | Tasks 2 and 3 |
|---|---|---|
| Platform | Kaggle, two NVIDIA T4 GPUs | Google Colab, one NVIDIA T4 GPU |
| PyTorch | 2.11.0 (CUDA 13.0) | 2.14.1 (CUDA 13.0) |
| transformers | 4.57.1 | 4.57.1 |
| lm-evaluation-harness | 0.4.9.1 | not used |
| math-verify | 0.9.0 | 0.9.0 |
| datasets | 3.6.0 | 3.6.0 |

Each run stores its software versions and configuration in a `config.json` file inside its results folder.

### Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `PA3_RESULTS` | Output folder for results | `results/` |
| `PA3_CACHE` | Cache for downloaded data and extracted features | `cache/` |
| `PA3_CKPT` | Folder for large recovery checkpoints | `ckpt/` |
| `PA3_T3_DATA` | Location of the supplied Task 3 data (`ATML-PA3-Data`) | downloaded automatically |

## 3. Reproducing each task

All commands are run from the repository root. Every stage is resumable: finished runs are skipped, and interrupted runs continue from the latest recovery checkpoint when `--resume` is given. The notebooks in `notebooks/` contain the exact setup cells used for the platform runs.

### Task 1: LLM fine-tuning

```bash
python -m task1.data                      # builds the 40,000-example training set and saves its IDs
python -m task1.pipeline --resume         # preflight, baseline, five training runs, evaluation, diagnostics, tables
python -m task1.bootstrap_ci              # paired bootstrap intervals for all benchmark differences
```

Options of `task1.pipeline`:

- `--methods fullft l2sp lora4 lora16 oplora16 baseline` selects a subset of runs.
- `--gpus 0,1` assigns one worker per GPU.
- `--smoke` and `--tiny` run very small versions for testing.
- `python -m task1.analyze` rebuilds tables and figures from the saved evaluations.

Reported cost: about 2.4 to 3.5 hours of training per method on one T4, plus evaluation of seven conditions (about 12 minutes for the likelihood benchmarks and 35 minutes for GSM8K generation per condition).

### Task 2: SigLIP-Qwen vision-language model

```bash
python -m task2.run --mode prep                          # builds splits and caches SigLIP features
python -m task2.run --mode all --resume --gpus 0         # trains the 196-token model, evaluates, analyses
python -m task2.run --mode all --resume --gpus 0 --optional   # also trains the 49-token model
python -m task2.run --mode analyze                       # rebuilds tables and figures
```

`--tokens 49` with `--mode train` trains only the compressed model. The first call downloads the COCO 2014 annotations and the VQAv2 questions, annotations, and complementary pairs, then downloads the selected images once and stores their features (about 2.7 GB, 5.5 minutes). The 196-token run takes about 95 minutes on one T4.

### Task 3: test-time compute

```bash
python -m task3.run --stage all --workers 2    # live generation on the GPU, offline analysis on the CPU
python -m task3.run --stage offline            # offline analysis only
python -m task3.run --stage live               # live MATH generation only (use --live-limit 2 for a quick check)
python -m task3.run --stage analyze            # rebuilds tables and figures
```

The supplied rollout data (`ATML-PA3-Data`) is downloaded automatically from the shared folder given in `task3/config.py`. Set `PA3_T3_DATA` to use a local copy. The MathVista question metadata is fetched from the Hugging Face dataset `AI4Math/MathVista`. The offline stage needs no GPU. The live stage uses `Qwen/Qwen2.5-3B-Instruct` at the revision pinned in `task3/config.py` and takes about 2.5 hours for 50 questions on one T4.

## 4. Saved results

Each task writes to `results/<task>/`. The main files are listed below.

| Task | Files |
|---|---|
| Task 1 | `tables/benchmarks.csv`, `tables/controlled_comparisons.csv`, `tables/bootstrap_ci.csv`, `tables/resources.csv`, `tables/diagnostics.csv`, `tables/displacement.csv`, `tables/qualitative_*`; `eval/<condition>/` (scores, per-example GSM8K and likelihood outputs, 24 qualitative responses per condition); `runs/<method>/` (configuration, training log, training summary); `diagnostics.json`; `oplora_constraint_check.json`; `figures/` |
| Task 2 | `run_196/` and `run_49/` (configuration, development metrics, training log, resources, final metrics, per-example predictions, cached evaluations); `tables/`; `split_report.json`; `feature_cache_resources.json`; `figures/` |
| Task 3 | `curves_*.csv`, `differences_*.csv`, `signals_*.csv`, `selection_results_*.parquet` (per-question, per-order outputs), `answer_groups_*.json`, `adaptive_*` files, `live_records.jsonl`, `live_strategies.csv`, `live_summary.json`, `trajectories.json`, example files, `figures/` |

Large binary artifacts are not committed. This covers model weights, adapters, weight-difference files, and Task 2 stage weights (about 430 MB in total) and is enforced by `.gitignore`. All reported numbers can be traced to the committed tables, logs, and per-example files.

## 5. Splits and example identifiers

The folder `splits/` stores every selection made with the seed:

| File | Content |
|---|---|
| `task1_train_ids.json` | Indices of the 40,000 MetaMathQA examples in source order |
| `task1_probe_ids.json` | The 32 GSM8K test and 32 HellaSwag validation examples used for the activation probes |
| `task2_splits.json` | Selected images, captions, grounding pairs, and natural VQAv2 pairs for training, development, and test |
| `task3_*_split_ids.json` | Development, held-out, and live question IDs for MATH and MathVista |
| `task3_*_orders.npz` | The ten rollout orders per question |

## 6. Report

The report source is in `paper/` (`main.tex`, `neurips_2026.sty`, and `figures/`). It can be compiled with any standard LaTeX distribution, for example:

```bash
cd paper
pdflatex main.tex && pdflatex main.tex
```

The report covers the three tasks in a single document with a separate results and analysis section for each.

## 7. Attribution of reused code, models, and data

- **lm-evaluation-harness** (version 0.4.9.1) provides the stock task definitions, prompts, stopping rules, and answer filters for GSM8K (`gsm8k_cot`), HellaSwag, BoolQ, and ARC-Easy. Only a thin wrapper that disables thinking in the chat template is added in `task1/evaluate.py`.
- **Math-Verify** (version 0.9.0) is used for mathematical answer equivalence in `task3/answers.py`.
- **Hugging Face transformers and datasets** load the models and datasets. Models: `Qwen/Qwen3-0.6B`, `google/siglip-base-patch16-224`, and `Qwen/Qwen2.5-3B-Instruct`. Datasets: MetaMathQA, GSM8K, COCO 2014, VQAv2, MATH, and MathVista (metadata only). The supplied rollouts come from the assignment data release.
- `task2/vqa_norm.py` re-implements the answer normalization and soft-accuracy rule of the official VQA evaluation code.
- `task1/first_plot_questions.yaml` is the prompt file supplied with the assignment.
- LoRA, OPLoRA, the L2-SP penalty, the training loops, the visual connector, the selection and stopping methods, and the analysis code were written for this assignment, following the descriptions in the cited papers and the assignment manual.
