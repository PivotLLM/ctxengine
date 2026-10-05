/******************************************************************************
 * Copyright (c) 2026 Tenebris Technologies Inc.                              *
 * Please see LICENSE file for details.                                       *
 ******************************************************************************/

package session

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/PivotLLM/ctxengine/internal/iox"
	"github.com/PivotLLM/ctxengine/logger"
	"github.com/PivotLLM/ctxengine/memory"
)

// maxLegacyLineSize is the largest JSON line the JSONL layout could hold; tool
// results (file reads, web fetches) made lines large.
const maxLegacyLineSize = 10 * 1024 * 1024

// legacyMeta is the <base>.meta.json record of the JSONL layout.
type legacyMeta struct {
	Key                         string    `json:"key"`
	Summary                     string    `json:"summary"`
	Skip                        int       `json:"skip"`
	CreatedAt                   time.Time `json:"created_at"`
	UpdatedAt                   time.Time `json:"updated_at"`
	NextSeq                     int64     `json:"next_seq"`
	MeaningfulCount             int       `json:"meaningful_count"`
	CompressedAtMeaningfulCount int       `json:"compressed_at_meaningful_count"`
	SummaryGeneratedAt          time.Time `json:"summary_generated_at"`
	SummaryModel                string    `json:"summary_model"`
	CompressionCooling          bool      `json:"compression_cooling"`
	CoolingSinceCount           int       `json:"cooling_since_count"`
	ActiveModelIndex            int       `json:"active_model_index"`
	ExposeReasoning             bool      `json:"expose_reasoning"`
	ShowToolActivity            bool      `json:"show_tool_activity"`
}

// MigrateReport is the outcome of one MigrateJSONL run.
type MigrateReport struct {
	Migrated []string         // session keys folded into their archive DB
	Skipped  []string         // keys whose DB already holds a window (already migrated)
	Errors   map[string]error // by session key, or by file name when the key is unknown
}

// MigrateJSONL folds every <base>.jsonl + <base>.meta.json pair in dir into
// <base>.archive.db and renames the sources to *.migrated. The key comes from
// meta.json; a .jsonl with no meta.json is reported as an error, never guessed
// from the file name. The first meta `skip` lines are gone; legacy lines
// without a seq get skip+lineNo. Every meta field maps into session_state and
// pending_turn is written 0 (the service is stopped while this runs).
// <base>.summaries.jsonl is left in place (the archive imports it on open) and
// so is any ancient <base>.archive.jsonl. Idempotent: a session whose
// session_state.next_seq > 0 is skipped untouched. Run with the service
// stopped. Returns nil, nil if dir does not exist.
func MigrateJSONL(dir string) (MigrateReport, error) {
	report := MigrateReport{Errors: make(map[string]error)}
	entries, err := os.ReadDir(dir)
	if os.IsNotExist(err) {
		return report, nil
	}
	if err != nil {
		return report, fmt.Errorf("session: migrate: %w", err)
	}

	metas := make(map[string]bool)
	windows := make(map[string]bool)
	for _, entry := range entries {
		name := entry.Name()
		switch {
		case entry.IsDir():
		case strings.HasSuffix(name, ".meta.json"):
			metas[strings.TrimSuffix(name, ".meta.json")] = true
		case strings.HasSuffix(name, ".summaries.jsonl"), strings.HasSuffix(name, ".archive.jsonl"):
		case strings.HasSuffix(name, ".jsonl"):
			windows[strings.TrimSuffix(name, ".jsonl")] = true
		}
	}

	bases := make([]string, 0, len(metas)+len(windows))
	for base := range metas {
		bases = append(bases, base)
	}
	for base := range windows {
		if !metas[base] {
			bases = append(bases, base)
		}
	}
	sort.Strings(bases)

	for _, base := range bases {
		if !metas[base] {
			report.Errors[base+".jsonl"] = errors.New("no .meta.json alongside; session key unknown")
			continue
		}
		key, skipped, err := migrateSession(dir, base, windows[base])
		switch {
		case err != nil && key == "":
			report.Errors[base+".meta.json"] = err
		case err != nil:
			report.Errors[key] = err
		case skipped:
			report.Skipped = append(report.Skipped, key)
		default:
			report.Migrated = append(report.Migrated, key)
		}
	}
	return report, nil
}

// migrateSession folds one session. key is "" when meta.json could not be
// read; skipped is true when the archive already holds a window.
func migrateSession(dir, base string, hasWindow bool) (key string, skipped bool, err error) {
	metaPath := filepath.Join(dir, base+".meta.json")
	data, err := os.ReadFile(metaPath) //nolint:gosec // G304: a file found by listing the host's sessions directory
	if err != nil {
		return "", false, fmt.Errorf("read meta: %w", err)
	}
	var meta legacyMeta
	if err = json.Unmarshal(data, &meta); err != nil {
		return "", false, fmt.Errorf("decode meta: %w", err)
	}
	if meta.Key == "" {
		return "", false, errors.New("meta.json has no key")
	}
	key = meta.Key

	a, err := memory.Open(memory.ArchivePath(dir, key))
	if err != nil {
		return key, false, fmt.Errorf("open archive: %w", err)
	}
	defer iox.CloseQuietly("session", a)

	st, err := a.State()
	if err != nil {
		return key, false, fmt.Errorf("read state: %w", err)
	}
	if st.NextSeq > 0 {
		return key, true, nil
	}

	var msgs []memory.StoredMessage
	windowPath := filepath.Join(dir, base+".jsonl")
	if hasWindow {
		msgs, err = readLegacyWindow(windowPath, meta.Skip)
		if err != nil {
			return key, false, err
		}
	}
	nextSeq := meta.NextSeq
	for _, m := range msgs {
		if m.Seq > nextSeq {
			nextSeq = m.Seq
		}
	}
	st = memory.SessionState{
		Key:       key,
		Summary:   meta.Summary,
		NextSeq:   nextSeq,
		CreatedAt: meta.CreatedAt,
		UpdatedAt: meta.UpdatedAt,
		Compaction: memory.CompactionState{
			MeaningfulCount:             meta.MeaningfulCount,
			CompressedAtMeaningfulCount: meta.CompressedAtMeaningfulCount,
			Cooling:                     meta.CompressionCooling,
			CoolingSinceCount:           meta.CoolingSinceCount,
			SummaryGeneratedAt:          meta.SummaryGeneratedAt,
			SummaryModel:                meta.SummaryModel,
			ActiveModelIndex:            meta.ActiveModelIndex,
			ExposeReasoning:             meta.ExposeReasoning,
			ShowToolActivity:            meta.ShowToolActivity,
		},
	}
	if err := a.ReplaceWindow(msgs, st); err != nil {
		return key, false, fmt.Errorf("write window: %w", err)
	}
	if err := a.Close(); err != nil {
		return key, false, fmt.Errorf("close archive: %w", err)
	}

	if hasWindow {
		if err := os.Rename(windowPath, windowPath+".migrated"); err != nil {
			return key, false, fmt.Errorf("rename window: %w", err)
		}
	}
	if err := os.Rename(metaPath, metaPath+".migrated"); err != nil {
		return key, false, fmt.Errorf("rename meta: %w", err)
	}
	logger.InfoCF("session", "migrated JSONL session",
		map[string]any{"session": key, "messages": len(msgs), "next_seq": nextSeq})
	return key, false, nil
}

// readLegacyWindow reads the JSON lines of a <base>.jsonl file after the first
// skip non-empty lines. Lines that do not parse (a partial write from a
// crash) are logged and dropped. Lines written before seqs existed (seq == 0)
// are assigned skip+lineNo, the position the JSONL store reported for them.
func readLegacyWindow(path string, skip int) ([]memory.StoredMessage, error) {
	f, err := os.Open(path) //nolint:gosec // G304: a file found by listing the host's sessions directory
	if err != nil {
		return nil, fmt.Errorf("open window: %w", err)
	}
	defer iox.CloseQuietly("session", f)

	var msgs []memory.StoredMessage
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 64*1024), maxLegacyLineSize)
	lineNum := 0
	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}
		lineNum++
		if lineNum <= skip {
			continue
		}
		var stored memory.StoredMessage
		if err := json.Unmarshal(line, &stored); err != nil {
			logger.WarnCF("session", "migrate: skipping corrupt line",
				map[string]any{"line": lineNum, "path": filepath.Base(path), "error": err.Error()})
			continue
		}
		if stored.Seq == 0 {
			stored.Seq = int64(skip + lineNum)
		}
		msgs = append(msgs, stored)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scan window: %w", err)
	}
	return msgs, nil
}
