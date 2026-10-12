# Multi-Turn Jailbreak Evaluation of Safety Alignment Methods

> **Content warning:** this repository contains harmful requests and prompts used to attack language models. They are included for research on model safety only.

This repository contains the code for the anonymous submission *Evaluating Jailbreak Defense Methods Against Multi-Turn Adversarial Attacks in Multi-Agent Debate*.

We compare three safety alignment methods, supervised fine-tuning (SFT), direct preference optimization (DPO), and latent adversarial training (LAT), under the same multi-turn persuasion attack. Three LLM agents take part in each dialogue:

- **Jailbreaker**: receives a harmful goal and a persuasion category and tries to make the victim fulfill the goal over five turns.
- **Victim**: the model under attack (the baseline or one of the aligned models).
- **Judge**: rates every victim response as safe or unsafe.

Robustness is reported as the defense success rate (DSR), the share of dialogues in which the victim gives no unsafe response. Helpfulness is measured as accuracy on IFEval and MMLU-Pro and as the compliance rate on XSTest and OR-Bench-Hard.

## Repository structure

```
.
├── data/
│   ├── pilot/                        # data from preliminary experiments
│   ├── train_sft.jsonl               # SFT training data (refusals + benign examples)
│   ├── train_sft_refusal_only.jsonl  # SFT ablation data (refusals only)
│   └── train_dpo.jsonl               # DPO preference data (prompt, chosen, rejected)
├── prompts/
│   ├── persuader_prompts.json        # jailbreaker prompts
│   ├── techniques.json               # persuasion categories and their techniques
│   ├── prompts_new.json              # prompt templates
│   ├── harm_text.json                # harmful reference texts
│   └── peaceful_text.json            # benign reference texts
├── src/
│   ├── agents/                       # jailbreaker, victim, and judge agents
│   ├── tools/                        # single-turn and multi-turn dialogue loops
│   ├── fine-tuning/                  # SFT (LoRA) and DPO training scripts and sanity checks
│   ├── run_test.py                   # main entry point for the multi-turn attack runs
│   ├── run_xstest_overrefusal.py     # XSTest compliance rate
│   ├── run_orbench_generation.py     # generates victim responses on OR-Bench-Hard
│   ├── judge_orbench.py              # classifies OR-Bench-Hard responses with an LLM judge
│   ├── summarize_orbench.py          # computes the OR-Bench-Hard compliance rate
│   ├── sample_for_annotation.py      # samples responses for manual inspection
│   └── test_model.py                 # quick check of a single model
├── calculate_turn.py                 # turn at which successful jailbreaks occur
├── testing_judge.py                  # judge validation on labeled data
├── hugging_face.py                   # model download helper
├── requirements.txt
└── LICENSE
```

LAT adapters are trained with the reference implementation of LLM-LAT using the hyperparameters listed in Appendix B of the paper.

## Setup

```bash
pip install -r requirements.txt
```

The main experiments are started with `src/run_test.py`. Model choices and inference settings follow Section 3.4 and Appendix A of the paper.

## License

See `LICENSE`. All datasets and models used are publicly available and subject to their own licenses.
