/******************************************************************************
 * Copyright (c) 2026 Tenebris Technologies Inc.                              *
 * Please see LICENSE file for details.                                       *
 ******************************************************************************/

package memory

// NoiseKeyFunc identifies messages that are repeated fires of one source (a
// scheduled job, for instance) by a key: two consecutive messages with the
// same key are duplicates whatever their timestamps say. ok is false for
// content that is not such a message. See session.SQLiteStore.SetNoiseKey.
type NoiseKeyFunc func(content string) (key string, ok bool)

// NoiseCache tracks the last stored message per role and the last noise key
// for one session so that a duplicate can be classified as noise. It is not
// safe for concurrent use; the session store guards it with the per-session
// lock.
type NoiseCache struct {
	lastByRole map[string]string // role -> last stored Content
	lastKey    string
}

// NewNoiseCache returns an empty cache.
func NewNoiseCache() *NoiseCache {
	return &NoiseCache{lastByRole: make(map[string]string)}
}

// IsNoise returns true if msg is a duplicate that contributes no new
// information. Messages the noise key recognises dedup on that key (for a
// scheduled job, the job fingerprint when present, else the payload) so that
// repeated fires of the same job — which embed differing timestamps — collapse
// to one. All other messages dedup on identical same-role content. A nil noise
// recognises nothing.
func (c *NoiseCache) IsNoise(msg StoredMessage, noise NoiseKeyFunc) bool {
	if c == nil {
		return false
	}
	if noise != nil {
		if key, ok := noise(msg.Content); ok {
			return c.lastKey == key
		}
	}
	return c.lastByRole[msg.Role] == msg.Content && msg.Content != ""
}

// Record stores msg in the cache for future IsNoise checks.
func (c *NoiseCache) Record(msg StoredMessage, noise NoiseKeyFunc) {
	if c == nil {
		return
	}
	if noise != nil {
		if key, ok := noise(msg.Content); ok {
			c.lastKey = key
			return
		}
	}
	c.lastByRole[msg.Role] = msg.Content
}
