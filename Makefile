SHELL := /bin/bash
# ctxengine — build and test entry points (see ~/.claude/standards/makefile.md).
#
#   make          run the full test suite, then build (only if the tests pass)
#   make test     the one gate (./test.sh): format check, go vet, golangci-lint
#                 and go test -race, with a summary; exits non-zero on any failure
#   make build    compile the packages (a library: nothing is produced)
#   make clean    remove test artifacts and the Go test cache
#   make fmt      rewrite formatting (the gate only verifies it)
#   make vet      go vet on its own
#   make lint     golangci-lint on its own
#
# There is no install target: this is a library, consumed as a Go module.

# golangci-lint: on PATH if present, otherwise the Go bin directory (go install
# puts it there, which is not on everyone's PATH). Override with GOLANGCI_LINT=.
GOLANGCI_LINT_VERSION ?= v2.13.2
GOLANGCI_LINT ?= $(shell command -v golangci-lint 2>/dev/null || echo "$$(go env GOPATH)/bin/golangci-lint")
export GOLANGCI_LINT GOLANGCI_LINT_VERSION

.PHONY: all test build clean fmt vet lint require-golangci-lint

all: test build

test:
	@./test.sh

# A library has no binaries; building checks that every package compiles.
build:
	@go build ./...

# The gate keeps its logs in a temporary directory it removes, so the only
# artifact to clear is the test cache.
clean:
	@go clean -testcache

fmt: require-golangci-lint
	@$(GOLANGCI_LINT) fmt

vet:
	@go vet ./...

lint: require-golangci-lint
	@$(GOLANGCI_LINT) run ./...

require-golangci-lint:
	@test -x "$(GOLANGCI_LINT)" || { \
	  echo "ERROR: golangci-lint not found (looked for $(GOLANGCI_LINT))."; \
	  echo "Install it with: go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@$(GOLANGCI_LINT_VERSION)"; \
	  echo "or set GOLANGCI_LINT=/path/to/golangci-lint"; exit 1; }
