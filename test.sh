#!/usr/bin/env bash
# ctxengine — the test gate. Run it through `make test`.
#
# Stages, all of which run even when an earlier one fails:
#   format  golangci-lint fmt --diff (verifies; `make fmt` rewrites)
#   vet     go vet ./...
#   lint    golangci-lint run ./...
#   tests   go test -race -count=1 ./...
#
# The gate never modifies the working tree. Its logs go to a temporary
# directory that is removed on exit; pass --keep to preserve it for debugging.
# Exits non-zero if any stage fails or a prerequisite is missing.

set -u

cd "$(dirname "$0")" || exit 1

GOLANGCI_LINT_VERSION="${GOLANGCI_LINT_VERSION:-v2.13.2}"
GOLANGCI_LINT="${GOLANGCI_LINT:-$(command -v golangci-lint 2>/dev/null || echo "$(go env GOPATH)/bin/golangci-lint")}"

keep=0
for arg in "$@"; do
	case "$arg" in
	--keep) keep=1 ;;
	-h | --help)
		echo "usage: ./test.sh [--keep]"
		echo "  --keep  keep the stage logs (their directory is printed at the end)"
		exit 0
		;;
	*)
		echo "unknown argument: $arg (try --help)" >&2
		exit 2
		;;
	esac
done

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
	RED=$'\033[31m' GREEN=$'\033[32m' YELLOW=$'\033[33m' BOLD=$'\033[1m' RESET=$'\033[0m'
else
	RED='' GREEN='' YELLOW='' BOLD='' RESET=''
fi

if ! command -v go >/dev/null 2>&1; then
	echo "${RED}ERROR:${RESET} go not found on PATH." >&2
	exit 1
fi
if [ ! -x "$GOLANGCI_LINT" ]; then
	echo "${RED}ERROR:${RESET} golangci-lint not found (looked for $GOLANGCI_LINT)." >&2
	echo "Install it with: go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@$GOLANGCI_LINT_VERSION" >&2
	echo "or set GOLANGCI_LINT=/path/to/golangci-lint" >&2
	exit 1
fi

logdir="$(mktemp -d "${TMPDIR:-/tmp}/ctxengine-test.XXXXXX")"
cleanup() {
	if [ "$keep" -eq 1 ]; then
		echo "Logs kept in $logdir"
	else
		rm -rf "$logdir"
	fi
}
trap cleanup EXIT

stage_names=()
stage_results=()
failed_stages=()

# record NAME STATUS — note a stage's outcome for the summary.
record() {
	stage_names+=("$1")
	if [ "$2" -eq 0 ]; then
		stage_results+=("PASS")
		echo "${GREEN}PASS${RESET}"
	else
		stage_results+=("FAIL")
		failed_stages+=("$1")
		echo "${RED}FAIL${RESET}"
	fi
}

# run_stage NAME CMD... — run a command, show its output only on failure.
run_stage() {
	local name="$1" log="$logdir/$1.log" status
	shift
	echo "${BOLD}==> $name${RESET}"
	"$@" >"$log" 2>&1
	status=$?
	if [ "$status" -ne 0 ]; then
		cat "$log"
	fi
	record "$name" "$status"
}

# format_check — golangci-lint fmt --diff prints a diff and changes nothing;
# any diff is a failure.
format_check() {
	local log="$logdir/format.log" status
	echo "${BOLD}==> format${RESET}"
	"$GOLANGCI_LINT" fmt --diff >"$log" 2>&1
	status=$?
	if [ "$status" -ne 0 ] || [ -s "$log" ]; then
		cat "$log"
		echo "Run 'make fmt' to apply the formatting."
		status=1
	fi
	record format "$status"
}

total=0 passed=0 failed=0 skipped=0

# go_tests — the race-enabled suite, bypassing the test cache. Counts cover
# every test and subtest; failing tests' output is shown.
go_tests() {
	local log="$logdir/tests.log" status
	echo "${BOLD}==> tests${RESET}"
	go test -race -count=1 -v ./... >"$log" 2>&1
	status=$?
	passed=$(grep -cE '^[[:space:]]*--- PASS' "$log")
	failed=$(grep -cE '^[[:space:]]*--- FAIL' "$log")
	skipped=$(grep -cE '^[[:space:]]*--- SKIP' "$log")
	total=$((passed + failed + skipped))
	if [ "$status" -ne 0 ] || [ "$failed" -ne 0 ]; then
		grep -vE '^[[:space:]]*(=== (RUN|PAUSE|CONT|NAME)|--- (PASS|SKIP)|PASS$|ok[[:space:]])' "$log"
		status=1
	fi
	record tests "$status"
}

format_check
run_stage vet go vet ./...
run_stage lint "$GOLANGCI_LINT" run ./...
go_tests

echo
echo "${BOLD}Summary${RESET}"
for i in "${!stage_names[@]}"; do
	result="${stage_results[$i]}"
	colour="$GREEN"
	[ "$result" = "FAIL" ] && colour="$RED"
	printf '  %-8s %s%s%s\n' "${stage_names[$i]}" "$colour" "$result" "$RESET"
done
skip_colour=""
[ "$skipped" -ne 0 ] && skip_colour="$YELLOW"
echo "  Tests: $total  passed: $passed  failed: $failed  ${skip_colour}skipped: $skipped${RESET}"

if [ "${#failed_stages[@]}" -ne 0 ]; then
	echo "${RED}${BOLD}FAILED:${RESET} ${failed_stages[*]}"
	exit 1
fi
echo "${GREEN}${BOLD}All stages passed.${RESET}"
