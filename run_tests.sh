#!/usr/bin/env bash
# Run portable CLI fixtures in tests/<test-name>/.
#
# input.yml is required. output.yml and stderr.txt contain optional, exact
# expected streams (an omitted file means an empty stream). exit_code contains
# an optional expected exit status and defaults to 0.
set -euo pipefail

if (( $# > 1 )); then
  printf 'Usage: %s [JINEST_CMD]\n' "$0" >&2
  exit 2
fi

# A positional command has precedence over the environment and the default.
JINEST_CMD="${1:-${JINEST_CMD:-python3 jinest.py}}"
test_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/tests"

total=0
passed=0
failed=0
fails_list=()
temporary_dir="$(mktemp -d "${TMPDIR:-/tmp}/jinest-tests.XXXXXX")"
trap 'rm -rf -- "$temporary_dir"' EXIT

shopt -s nullglob
for test_dir in "$test_root"/*/; do
  test_name="$(basename "$test_dir")"
  input="$test_dir/input.yml"
  output="$test_dir/output.yml"
  expected_stderr="$test_dir/stderr.txt"
  expected_exit_file="$test_dir/exit_code"

  ((total += 1))
  if [[ ! -f "$input" ]]; then
    printf 'FAIL %s (input.yml is missing)\n' "$test_name"
    ((failed += 1))
    fails_list+=("$test_name")
    continue
  fi

  expected_exit=0
  if [[ -f "$expected_exit_file" ]]; then
    expected_exit="$(<"$expected_exit_file")"
  fi
  if [[ ! "$expected_exit" =~ ^[0-9]+$ ]]; then
    printf 'FAIL %s (exit_code must be a non-negative integer)\n' "$test_name"
    ((failed += 1))
    fails_list+=("$test_name")
    continue
  fi

  actual_stdout="$temporary_dir/$test_name.stdout"
  actual_stderr="$temporary_dir/$test_name.stderr"
  set +e
  bash -c "$JINEST_CMD \"\$1\"" jinest-command "$input" \
    >"$actual_stdout" 2>"$actual_stderr"
  actual_exit=$?
  set -e

  streams_match=true
  if [[ -f "$output" ]]; then
    cmp -s "$actual_stdout" "$output" || streams_match=false
  elif [[ -s "$actual_stdout" ]]; then
    streams_match=false
  fi
  if [[ -f "$expected_stderr" ]]; then
    cmp -s "$actual_stderr" "$expected_stderr" || streams_match=false
  elif [[ -s "$actual_stderr" ]]; then
    streams_match=false
  fi

  if [[ "$actual_exit" == "$expected_exit" && "$streams_match" == true ]]; then
    printf 'PASS %s\n' "$test_name"
    ((passed += 1))
  else
    printf 'FAIL %s\n' "$test_name"
    ((failed += 1))
    fails_list+=("$test_name")
  fi
done

if (( failed > 0 )); then
  printf '\nFAILED:\n'
  for test_name in "${fails_list[@]}"; do
    printf '  %s\n' "$test_name"
  done
fi

printf '\nPassed: %d/%d\n' "$passed" "$total"
if (( failed > 0 )); then
  printf 'Failed: %d\n' "$failed"
  exit 1
fi
