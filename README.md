# PySplitLock: Static Partition-Taint Analysis and Metamorphic Patch Rejection for Autonomous Machine Learning Agents

**Author:** Venkata Sai Swathi Gattamaneni

`PySplitLock` is a deterministic verification harness designed to prevent silent data leakage, evaluator tampering, and state collapse in self-modifying machine learning agents (`MLAgentBench`, `The AI Scientist`, Aider-based coding loops).

## Key Features
1. **Stage 1 — Cryptographic Evaluator & Support Lock:** Isolates `eval.py` and holdout row indices via SHA-256 checksums and blocks silent `dropna()` sample-support mutations.
2. **Stage 2 — Static AST Partition-Taint Analysis:** Parses candidate Python scripts prior to execution (`~1.4 ms` overhead) to block stateful fitting (`.fit()`, `.fit_transform()`, global imputers) on `UNSPLIT` or `HOLDOUT` partitions.
3. **Stage 3 — Dynamic Metamorphic Patch Rejection:** Runs fast micro-slice sanity checks (`~0.38s` overhead) using **MR-1 (Target Label Randomization)** and **MR-2 (Batch-Order Invariance)**.

## Quick Start

```bash
git clone [https://github.com/swathi420/pysplitlock.git](https://github.com/swathi420/pysplitlock.git)
cd pysplitlock
pip install -e .
python run_benchmarks.py