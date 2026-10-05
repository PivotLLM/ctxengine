/******************************************************************************
 * Copyright (c) 2026 Tenebris Technologies Inc.                              *
 * Please see LICENSE file for details.                                       *
 ******************************************************************************/

package memory

import (
	"time"

	"github.com/PivotLLM/spawnllm"
)

// StoredMessage is a spawnllm.Message decorated with a monotonically increasing
// sequence number assigned at write time and the UTC timestamp of when it was stored.
// Seq and CreatedAt are storage-layer-only and are never part of spawnllm.Message.
//
// Construct via NewStoredMessage or NewStoredMessageAt; direct struct-literal
// construction outside this package's tests will leave CreatedAt zero and corrupt
// on-disk metadata (covered_seq_*_at on context summaries, etc.).
type StoredMessage struct {
	Seq       int64     `json:"seq"`
	CreatedAt time.Time `json:"created_at"`
	spawnllm.Message
}

// NewStoredMessage builds a StoredMessage and stamps CreatedAt with the current
// UTC wall-clock time. Use this on the write path when no prior timestamp exists.
func NewStoredMessage(seq int64, msg spawnllm.Message) StoredMessage {
	return NewStoredMessageAt(seq, msg, time.Now().UTC())
}

// NewStoredMessageAt builds a StoredMessage with an explicit CreatedAt. If the
// supplied time is zero, CreatedAt is stamped with time.Now().UTC() to preserve
// the non-zero invariant. Use this when reading a timestamp back from a durable
// source (e.g. the archive DB).
func NewStoredMessageAt(seq int64, msg spawnllm.Message, createdAt time.Time) StoredMessage {
	if createdAt.IsZero() {
		createdAt = time.Now().UTC()
	} else {
		createdAt = createdAt.UTC()
	}
	return StoredMessage{
		Seq:       seq,
		CreatedAt: createdAt,
		Message:   msg,
	}
}
