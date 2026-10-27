"""
pysplitlock.py - Static Partition-Taint Analysis and Metamorphic Patch Rejection
Author: Venkata Sai Swathi Gattamaneni
License: Apache-2.0
"""
import ast
import hashlib
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np


class TaintLevel(IntEnum):
    UNTAINTED = 0
    TRAIN = 1
    UNSPLIT = 2
    HOLDOUT = 3


@dataclass
class VerificationIssue:
    stage: str
    lineno: int
    rule_id: str
    message: str


class PartitionTaintAnalyzer(ast.NodeVisitor):
    """
    Stage 2: Static AST Partition-Taint Analyzer for Python ML scripts.
    Tracks variable lineage from file/dataset loads through train_test_split()
    and flags stateful fitting or global aggregations on UNSPLIT or HOLDOUT data.
    """

    DATA_LOADERS = {
        "read_csv", "read_parquet", "read_table", "load_dataset",
        "load", "fetch_openml", "make_classification", "load_breast_cancer"
    }
    SPLIT_FUNCS = {"train_test_split"}
    FIT_SINKS = {"fit", "fit_transform", "fit_resample"}
    STAT_SINKS = {"mean", "std", "median", "quantile", "agg", "dropna", "fillna"}

    def __init__(self) -> None:
        self.symbol_taint: Dict[str, TaintLevel] = {}
        self.issues: List[VerificationIssue] = []

    def _expr_taint(self, node: Optional[ast.AST]) -> TaintLevel:
        if node is None:
            return TaintLevel.UNTAINTED
        max_level = TaintLevel.UNTAINTED
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and child.id in self.symbol_taint:
                max_level = max(max_level, self.symbol_taint[child.id])
        return max_level

    def visit_Assign(self, node: ast.Assign) -> None:
        if isinstance(node.value, ast.Call):
            func_name = self._get_call_name(node.value)

            # Handle train_test_split() partition transitions
            if func_name in self.SPLIT_FUNCS and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Tuple) and len(target.elts) in (2, 4):
                    elts = target.elts
                    if len(elts) == 2:
                        self._set_target(elts[0], TaintLevel.TRAIN)
                        self._set_target(elts[1], TaintLevel.HOLDOUT)
                    elif len(elts) == 4:
                        self._set_target(elts[0], TaintLevel.TRAIN)
                        self._set_target(elts[1], TaintLevel.HOLDOUT)
                        self._set_target(elts[2], TaintLevel.TRAIN)
                        self._set_target(elts[3], TaintLevel.HOLDOUT)
                    self.generic_visit(node)
                    return

            # Handle primary dataset loaders
            if func_name in self.DATA_LOADERS:
                is_holdout = any(
                    isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                    and any(k in arg.value.lower() for k in ("val", "test", "holdout"))
                    for arg in node.value.args
                )
                level = TaintLevel.HOLDOUT if is_holdout else TaintLevel.UNSPLIT
                for t in node.targets:
                    self._set_target(t, level)
                self.generic_visit(node)
                return

        rhs_taint = self._expr_taint(node.value)
        for t in node.targets:
            self._set_target(t, rhs_taint)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func_name = self._get_call_name(node)

        if func_name in self.FIT_SINKS or func_name in self.STAT_SINKS:
            receiver_taint = TaintLevel.UNTAINTED
            receiver_name = "<expr>"
            if isinstance(node.func, ast.Attribute):
                receiver_taint = self._expr_taint(node.func.value)
                if isinstance(node.func.value, ast.Name):
                    receiver_name = node.func.value.id

            arg_taints = [self._expr_taint(arg) for arg in node.args]
            max_arg_taint = max(arg_taints, default=TaintLevel.UNTAINTED)

            if func_name in self.FIT_SINKS:
                if max_arg_taint in {TaintLevel.UNSPLIT, TaintLevel.HOLDOUT}:
                    label = TaintLevel(max_arg_taint).name
                    self.issues.append(VerificationIssue(
                        stage="STAGE2_AST",
                        lineno=node.lineno,
                        rule_id="PSL-AST-001",
                        message=f"Stateful '{func_name}()' invoked on {label}-tainted argument."
                    ))
                    # Taint receiver object (e.g., scaler) if fit on leaked data
                    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                        self.symbol_taint[node.func.value.id] = max_arg_taint

            elif func_name in self.STAT_SINKS:
                if receiver_taint == TaintLevel.UNSPLIT and func_name != "fillna":
                    self.issues.append(VerificationIssue(
                        stage="STAGE2_AST",
                        lineno=node.lineno,
                        rule_id="PSL-AST-002",
                        message=f"Global operation '{func_name}()' on UNSPLIT data '{receiver_name}' before split."
                    ))
                elif receiver_taint == TaintLevel.HOLDOUT and func_name == "dropna":
                    self.issues.append(VerificationIssue(
                        stage="STAGE2_AST",
                        lineno=node.lineno,
                        rule_id="PSL-AST-003",
                        message=f"Sample-support mutation: 'dropna()' called on HOLDOUT split '{receiver_name}'."
                    ))

        self.generic_visit(node)

    def _set_target(self, target: ast.AST, level: TaintLevel) -> None:
        if isinstance(target, ast.Name):
            self.symbol_taint[target.id] = level
        elif isinstance(target, ast.Tuple):
            for elt in target.elts:
                self._set_target(elt, level)
        elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
            cur = self.symbol_taint.get(target.value.id, TaintLevel.UNTAINTED)
            self.symbol_taint[target.value.id] = max(cur, level)

    @staticmethod
    def _get_call_name(node: ast.Call) -> str:
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ""


class PySplitLockSupervisor:
    """
    Full 3-Stage Verification Supervisor for Autonomous ML Agents.
    """

    def __init__(self, canonical_eval_code: str, expected_val_len: int) -> None:
        self.eval_hash = hashlib.sha256(canonical_eval_code.encode("utf-8")).hexdigest()
        self.expected_val_len = expected_val_len

    def verify_stage1_hash_and_support(
        self, current_eval_code: str, val_preds: np.ndarray
    ) -> Tuple[bool, List[VerificationIssue]]:
        issues: List[VerificationIssue] = []
        curr_hash = hashlib.sha256(current_eval_code.encode("utf-8")).hexdigest()
        if curr_hash != self.eval_hash:
            issues.append(VerificationIssue(
                stage="STAGE1_HASH",
                lineno=0,
                rule_id="PSL-SEC-001",
                message="Evaluator script SHA-256 digest mismatch (tampering detected)."
            ))
        if len(val_preds) != self.expected_val_len or np.isnan(val_preds).any():
            issues.append(VerificationIssue(
                stage="STAGE1_SUPPORT",
                lineno=0,
                rule_id="PSL-SEC-002",
                message=f"Validation support mismatch: expected {self.expected_val_len} valid rows, got {len(val_preds)}."
            ))
        return len(issues) == 0, issues

    def verify_stage2_ast(self, candidate_code: str) -> Tuple[bool, List[VerificationIssue]]:
        try:
            tree = ast.parse(candidate_code)
        except SyntaxError as exc:
            return False, [VerificationIssue("STAGE2_AST", exc.lineno or 0, "PSL-SYNTAX", str(exc))]
        analyzer = PartitionTaintAnalyzer()
        analyzer.visit(tree)
        return len(analyzer.issues) == 0, analyzer.issues

    def verify_stage3_metamorphic(
        self,
        pipeline_runner: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[float, np.ndarray]],
        X_micro_train: np.ndarray,
        y_micro_train: np.ndarray,
        X_micro_val: np.ndarray,
        y_micro_val: np.ndarray,
        random_auc_ceiling: float = 0.62
    ) -> Tuple[bool, List[VerificationIssue]]:
        issues: List[VerificationIssue] = []
        rng = np.random.default_rng(42)

        # MR-1: Target Label Permutation Check
        y_train_perm = rng.permutation(y_micro_train)
        y_val_perm = rng.permutation(y_micro_val)
        perm_auc, preds_normal = pipeline_runner(X_micro_train, y_train_perm, X_micro_val, y_val_perm)

        if perm_auc > random_auc_ceiling:
            issues.append(VerificationIssue(
                stage="STAGE3_META",
                lineno=0,
                rule_id="PSL-META-001",
                message=f"MR-1 Failed: Model achieved {perm_auc:.3f} AUC on randomized labels (target leakage/exploit)."
            ))

        # MR-2: Batch-Order Invariance & Non-Collapse Check
        _, base_preds = pipeline_runner(X_micro_train, y_micro_train, X_micro_val, y_micro_val)
        if np.std(base_preds) < 1e-6:
            issues.append(VerificationIssue(
                stage="STAGE3_META",
                lineno=0,
                rule_id="PSL-META-002",
                message="MR-2 Failed: Degenerate constant predictions across validation slice."
            ))
        else:
            perm_idx = rng.permutation(len(X_micro_val))
            _, perm_preds = pipeline_runner(X_micro_train, y_micro_train, X_micro_val[perm_idx], y_micro_val[perm_idx])
            max_diff = np.max(np.abs(perm_preds - base_preds[perm_idx]))
            if max_diff > 1e-5:
                issues.append(VerificationIssue(
                    stage="STAGE3_META",
                    lineno=0,
                    rule_id="PSL-META-003",
                    message=f"MR-2 Failed: Batch-order sensitivity detected (max delta = {max_diff:.2e})."
                ))

        return len(issues) == 0, issues