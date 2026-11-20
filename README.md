# EN4554 Assignment 1: LoRA finetuning of a VLM, and a tool-calling agent

**De Zoysa A. S. I. (220106D)** · **Jayasekara S. P. R. (220257N)** · EN4554 Deep Learning for Vision

| Part | Report | Code |
|---|---|---|
| 1. Does LoRA+ transfer to vision–language models? | [report_part1.pdf](Part1_LoRAplus_SmolVLM/report/report_part1.pdf) (3 pages) | 5 notebooks in [Part1_LoRAplus_SmolVLM](Part1_LoRAplus_SmolVLM) |
| 2. Teaching a 0.5B model to call tools: baseline vs. LoRA | [report_part2.pdf](Part2_ToolCalling_Agent/report/report_part2.pdf) (1 page) | [agent_tool_calling.ipynb](Part2_ToolCalling_Agent/agent_tool_calling.ipynb) |

## Repository layout

```
Part1_LoRAplus_SmolVLM/
├── 01_setup_data_zeroshot.ipynb     data subsets (TextVQA, VQAv2) + zero-shot baseline        [Colab T4]
├── 02_lora_vs_loraplus_grid.ipynb   LoRA vs LoRA+ grid over (eta_A, lambda), seed 42          [Colab T4]
├── 03_seed_repeats.ipynb            best settings re-run on fresh seeds 1-3                    [Colab T4]
├── 04_resolution_test.ipynb         zero-shot accuracy under 5 visual-input settings           [Colab T4]
├── 05_report_numbers.ipynb          recomputes every number in the report from results/       [CPU, ~1 min]
├── results/                         result files written by notebooks 01-04
└── report/                          report_part1.pdf, LaTeX source, figures, make_figures.py

Part2_ToolCalling_Agent/
├── agent_tool_calling.ipynb         tools, agent loop, data, baseline, LoRA training, evaluation  [Colab T4]
├── data/                            generated dataset: train.jsonl (688), test.jsonl (100)
├── results/                         trajectories of all 3 models, per-request scores, summary, training log
└── report/                          report_part2.pdf, LaTeX source
```

## Part 1: LoRA+ on SmolVLM-256M

The paper's claims are tested on SmolVLM-256M-Instruct. LoRA is compared with LoRA+ (B learns λ× faster than A) on TextVQA and VQAv2 (2,000 training and 500 evaluation questions each). A grid search picks the best settings, which are then confirmed on 3 fresh seeds with paired bootstrap CIs.

- **H1, accuracy: not supported.** Tuned LoRA+ minus tuned LoRA: −1.7 [−3.7, +0.3] points on TextVQA and −0.0 [−1.7, +1.6] on VQAv2.
- **H1s, speed: supported.** At the same η_A, LoRA+ reaches a training loss below 1.0 in 2.2× (TextVQA) and 5× (VQAv2) fewer steps.
- **H2, larger gain on the harder task: not supported.** The difference between tasks is −1.7 [−4.3, +0.9].
- **Vision-specific finding (tested).** The visual token budget moves accuracy by about 18 points, against at most 2 for the learning-rate ratio. Finetuning at 64 visual tokens recovers 95% of the gain from the model's default tiling on VQAv2, but only 48% on TextVQA.

**Run:** notebooks 01 → 04 on a Colab T4 GPU, in order. Notebook 01 builds the data subsets on Google Drive (`MyDrive/EN4554_LoRAplus/`), and notebooks 02–04 are resumable after a disconnect. Notebook 05 needs no GPU: it reads `results/` and reproduces every number in the report. To regenerate the figures, run `python report/make_figures.py`.

## Part 2: tool-calling agent, baseline vs. LoRA

Qwen2.5-0.5B-Instruct is placed in an agent loop with 4 tools that use made-up data (weather, calendar, unit conversion, contacts). Training uses 688 template-generated conversations; the 100 test requests use phrasings and date formats that never appear in training. LoRA (r = 16, 2 epochs) is trained with the loss on the assistant turns only. The base model is also evaluated with a better hand-written prompt.

| Metric (%) | Baseline | + better prompt | LoRA | LoRA − baseline [95% CI] |
|---|---|---|---|---|
| Call validity (73 tool requests) | 97.3 | 95.9 | **100.0** | +2.7 [0.0, +6.8] |
| Call accuracy (73) | 67.1 | 75.3 | **93.2** | +26.0 [+16.4, +35.6] |
| Abstention (27 no-tool requests) | 29.6 | 29.6 | **96.3** | +66.7 [+48.1, +85.2] |
| Task success (100) | 58.0 | 62.0 | **85.0** | +27.0 [+15.0, +39.0] |

- **What LoRA learns:** mainly whether to call a tool and in what order. The base model calls a tool on 19 of 27 requests that need none, and solves 0 of 8 two-step requests; LoRA solves 6 of 8.
- **What LoRA introduces:** 8 answers with an invented weekday, a habit copied from the training answers.
- **The better prompt:** it fixes a convention (unrequested Fahrenheit) but not these decisions.

The report also gives the failure analysis (fixed / not fixed / introduced) and a robustness check on the Fahrenheit convention.

**Run:** open the notebook in Colab with a T4 GPU and choose *Run all* (about 30 min the first time). Everything is cached in `MyDrive/EN4554_Agent/`, so a rerun continues where it stopped.

## Reproducibility

`transformers==5.19.0` and `peft==0.21.2` are installed by the first cell of each Colab notebook. Training uses fp16 on a T4 (no bf16). Seeds are fixed: 42 for the grid and Part 2, and 1–3 for the fresh-seed repeats.
