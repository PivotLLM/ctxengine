# ctxengine — build and test gate.
#
# `make test` runs the whole suite with the race detector and exits non-zero on
# any failure; `make check` adds the formatting and vet gates in front of it.

GO ?= go

.PHONY: test check fmt-check vet

test:
	$(GO) test -race -count=1 ./...

fmt-check:
	@unformatted="$$(gofmt -l .)"; \
	if [ -n "$$unformatted" ]; then \
		echo "gofmt: the following files need formatting:"; echo "$$unformatted"; exit 1; \
	fi

vet:
	$(GO) vet ./...

check: fmt-check vet test
