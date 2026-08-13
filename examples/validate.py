#!/usr/bin/env python3
"""Execute every documented example and compare it with its expected result."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml


EXAMPLES = Path(__file__).resolve().parent
REPOSITORY = EXAMPLES.parent


def load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=REPOSITORY,
        text=True,
        capture_output=True,
        check=False,
    )


def main() -> int:
    failed = False
    with tempfile.TemporaryDirectory(prefix="jinest-examples-") as temp_dir:
        temporary = Path(temp_dir)
        actual_results: dict[str, Any] = {}

        for example in sorted(EXAMPLES.glob("*/example.yml")):
            name = example.parent.name
            output = temporary / f"{name}.yml"

            if name in {"06_imports", "08_python_api"}:
                process = run([sys.executable, str(example.parent / "run.py")])
                if process.returncode == 0:
                    output.write_text(process.stdout, encoding="utf-8")
            else:
                process = run(
                    [
                        sys.executable,
                        str(REPOSITORY / "jinest.py"),
                        str(example),
                        "--output-format",
                        "yaml",
                        "-o",
                        str(output),
                    ]
                )

            if process.returncode != 0:
                failed = True
                print(f"FAIL {name}: process exited {process.returncode}")
                print(process.stderr.rstrip())
                continue

            expected = load_yaml(example.parent / "result.yml")
            actual = load_yaml(output)
            actual_results[name] = actual
            if actual != expected:
                failed = True
                print(f"FAIL {name}: YAML result differs")
                print(yaml.safe_dump(actual, allow_unicode=True, sort_keys=False))
            else:
                print(f"OK   {name}")

        # The code-loop and compose versions intentionally implement the same
        # matrix. Keep this invariant checked independently of their fixtures.
        code_matrix = actual_results.get("12_matrix_by_code")
        compose_matrix = actual_results.get("13_matrix_by_composition")
        if code_matrix is not None and compose_matrix is not None:
            if code_matrix != compose_matrix:
                failed = True
                print("FAIL matrix equivalence: 12_matrix_by_code != 13_matrix_by_composition")
                print("matrix_by_code:")
                print(yaml.safe_dump(code_matrix, allow_unicode=True, sort_keys=False))
                print("matrix_by_composition:")
                print(yaml.safe_dump(compose_matrix, allow_unicode=True, sort_keys=False))
            else:
                print("OK   matrix equivalence: 12_matrix_by_code == 13_matrix_by_composition")

        # Byte escapes are a textual JSON contract, so compare them exactly.
        extended = EXAMPLES / "09_extended_values"
        json_output = temporary / "09_extended_values.json"
        process = run(
            [
                sys.executable,
                str(REPOSITORY / "jinest.py"),
                str(extended / "example.yml"),
                "--output-format",
                "json",
                "-o",
                str(json_output),
            ]
        )
        if process.returncode != 0:
            failed = True
            print(f"FAIL 09_extended_values JSON: exited {process.returncode}")
            print(process.stderr.rstrip())
        else:
            expected_json = (extended / "result.json").read_text(
                encoding="utf-8"
            ).rstrip("\n")
            actual_json = json_output.read_text(encoding="utf-8").rstrip("\n")
            if actual_json != expected_json:
                failed = True
                print("FAIL 09_extended_values: JSON result differs")
            else:
                print("OK   09_extended_values JSON escapes")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
