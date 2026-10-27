from pysplitlock import verify_code_patch

passed, issues = verify_code_patch(candidate_python_code)
if not passed:
    for issue in issues:
        print(f"[Line {issue.lineno}] {issue.rule_id}: {issue.message}")