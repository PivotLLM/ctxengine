// ctxengine
// License: MIT

package ctxengine

import (
	"context"
	"strings"
	"testing"

	"github.com/PivotLLM/ctxengine/memory"
	"github.com/PivotLLM/ctxengine/session"
	"github.com/PivotLLM/spawnllm"
)

// mockStore is a minimal in-memory SessionStore for trigger tests.
type mockStore struct {
	history map[string][]spawnllm.Message
	summary map[string]string
}

func newMockStore() *mockStore {
	return &mockStore{
		history: make(map[string][]spawnllm.Message),
		summary: make(map[string]string),
	}
}

func (s *mockStore) AddMessage(sessionKey, role, content string) error {
	s.history[sessionKey] = append(s.history[sessionKey], spawnllm.Message{Role: role, Content: content})
	return nil
}

func (s *mockStore) AddFullMessage(sessionKey string, msg spawnllm.Message) (int64, error) {
	s.history[sessionKey] = append(s.history[sessionKey], msg)
	return int64(len(s.history[sessionKey])), nil
}

func (s *mockStore) GetHistory(key string) []spawnllm.Message {
	src := s.history[key]
	if len(src) == 0 {
		return nil
	}
	cp := make([]spawnllm.Message, len(src))
	copy(cp, src)
	return cp
}

func (s *mockStore) GetSummary(key string) string { return s.summary[key] }
func (s *mockStore) SetSummary(key, summary string) error {
	s.summary[key] = summary
	return nil
}

func (s *mockStore) SetHistory(key string, h []spawnllm.Message) error {
	cp := make([]spawnllm.Message, len(h))
	copy(cp, h)
	s.history[key] = cp
	return nil
}

func (s *mockStore) TruncateHistory(key string, keepLast int) error {
	h := s.history[key]
	if keepLast <= 0 {
		s.history[key] = nil
		return nil
	}
	if keepLast >= len(h) {
		return nil
	}
	s.history[key] = h[len(h)-keepLast:]
	return nil
}
func (s *mockStore) SetPendingTurn(_ string) error            { return nil }
func (s *mockStore) ClearPendingTurn(_ string) error          { return nil }
func (s *mockStore) GetArchiveBounds(_ string) (int64, int64) { return 0, 0 }
func (s *mockStore) ListPendingSessions() ([]string, error)   { return nil, nil }
func (s *mockStore) Save(_ string) error                      { return nil }
func (s *mockStore) Close() error                             { return nil }
func (s *mockStore) GetHistoryWithSeqs(key string) []memory.StoredMessage {
	src := s.history[key]
	stored := make([]memory.StoredMessage, len(src))
	for i, msg := range src {
		stored[i] = memory.StoredMessage{Seq: int64(i + 1), Message: msg}
	}
	return stored
}

// newTestManager creates a Manager with the given options and returns the
// concrete *Manager so tests can call SetTestCompressHook.
func newTestManager(store session.SessionStore, opts ...Option) *Manager {
	// Zero the fixed per-request reserve by default. These tests use tiny
	// context windows (1000-10000 tokens) to make the trigger arithmetic legible,
	// and the real 4000-token reserve would dominate every one of them. Tests
	// that care about the reserve pass WithOverheadTokens explicitly — options
	// are applied in order, so a later one wins.
	cm := New("test-session", store, append([]Option{WithOverheadTokens(0)}, opts...)...)
	return cm.(*Manager)
}

// msgWithContent builds a spawnllm.Message with the given content.
func msgWithContent(content string) spawnllm.Message {
	return spawnllm.Message{Role: "user", Content: content}
}

// TestTrigger_BelowFloor verifies that compress is never called when the
// token count stays below minPercent regardless of message count.
func TestTrigger_BelowFloor(t *testing.T) {
	store := newMockStore()
	// contextWindow=10000, minPercent=20 → floor at 2000 tokens.
	// Each empty message contributes 0 tokens; use 1-char messages → 0 tokens (integer math).
	mgr := newTestManager(store,
		WithContextWindow(10000),
		WithMinPercent(20),
		WithNormalPercent(50),
		WithSafetyPercent(80),
		WithMessageThreshold(5),
	)

	called := false
	mgr.SetTestCompressHook(func(_ bool) { called = true })

	ctx := context.Background()
	// Add 10 messages with tiny content well below floor.
	for i := 0; i < 10; i++ {
		if _, err := mgr.AddUserMessage(ctx, msgWithContent("hi")); err != nil {
			t.Fatalf("AddUserMessage error: %v", err)
		}
	}

	if called {
		t.Error("compress should not be called when tokens are below minPercent floor")
	}
}

// TestTrigger_CountTriggered verifies that compress fires when the message
// count threshold is crossed, even when token usage is above the floor.
func TestTrigger_CountTriggered(t *testing.T) {
	store := newMockStore()
	// Use contextWindow=1000, minPercent=10 → floor at 100 tokens.
	// Each message has 60 chars → ~24 tokens; 5 msgs → ~120 tokens → 12% → above floor.
	// normalPercent=90 ensures the token trigger won't fire; messageThreshold=5.
	mgr := newTestManager(store,
		WithContextWindow(1000),
		WithMinPercent(10),
		WithNormalPercent(90),
		WithSafetyPercent(95),
		WithMessageThreshold(5),
	)

	var hookCalls []bool
	mgr.SetTestCompressHook(func(safetyNet bool) { hookCalls = append(hookCalls, safetyNet) })

	ctx := context.Background()
	// 100-char content → 25 tokens each; 5 msgs → 125 tokens → 12.5% → above 10% floor.
	content := strings.Repeat("x", 100)
	for i := 0; i < 5; i++ {
		if _, err := mgr.AddUserMessage(ctx, msgWithContent(content)); err != nil {
			t.Fatalf("AddUserMessage error: %v", err)
		}
	}

	if len(hookCalls) == 0 {
		t.Fatal("compress should be called when messageThreshold is reached")
	}
	if hookCalls[0] {
		t.Error("compress should be called with safetyNet=false for count trigger")
	}
}

// TestTrigger_NormalPercentTriggered verifies that compress fires when token
// usage crosses normalPercent but not safetyPercent.
func TestTrigger_NormalPercentTriggered(t *testing.T) {
	store := newMockStore()
	// contextWindow=1000, normalPercent=50 → trigger at 500 tokens.
	// safetyPercent=80. messageThreshold=100 (won't reach it).
	// 2100 chars in a single message → 525 tokens → 52.5% → crosses normal (50%).
	mgr := newTestManager(store,
		WithContextWindow(1000),
		WithMinPercent(10),
		WithNormalPercent(50),
		WithSafetyPercent(80),
		WithMessageThreshold(100),
	)

	var hookCalls []bool
	mgr.SetTestCompressHook(func(safetyNet bool) { hookCalls = append(hookCalls, safetyNet) })

	ctx := context.Background()
	// Add a message large enough to cross normalPercent (50% of 1000 = 500 tokens).
	// 2100 chars → 2100/4 = 525 tokens → 52.5%.
	bigContent := strings.Repeat("a", 2100)
	if _, err := mgr.AddUserMessage(ctx, msgWithContent(bigContent)); err != nil {
		t.Fatalf("AddUserMessage error: %v", err)
	}

	if len(hookCalls) == 0 {
		t.Fatal("compress should be called when tokens cross normalPercent")
	}
	if hookCalls[0] {
		t.Error("compress should be called with safetyNet=false for normal percent trigger")
	}
}

// TestTrigger_SafetyNetTriggered verifies that compress fires with safetyNet=true
// when token usage crosses safetyPercent.
func TestTrigger_SafetyNetTriggered(t *testing.T) {
	store := newMockStore()
	// contextWindow=1000, safetyPercent=80 → trigger at 800 tokens.
	// 3300 chars → 3300/4 = 825 tokens → 82.5% → crosses safetyPercent=80.
	mgr := newTestManager(store,
		WithContextWindow(1000),
		WithMinPercent(10),
		WithNormalPercent(50),
		WithSafetyPercent(80),
		WithMessageThreshold(100),
	)

	var hookCalls []bool
	mgr.SetTestCompressHook(func(safetyNet bool) { hookCalls = append(hookCalls, safetyNet) })

	ctx := context.Background()
	// 3300 chars → 825 tokens → 82.5% → crosses safetyPercent=80.
	bigContent := strings.Repeat("a", 3300)
	if _, err := mgr.AddUserMessage(ctx, msgWithContent(bigContent)); err != nil {
		t.Fatalf("AddUserMessage error: %v", err)
	}

	if len(hookCalls) == 0 {
		t.Fatal("compress should be called when tokens cross safetyPercent")
	}
	if !hookCalls[0] {
		t.Error("compress should be called with safetyNet=true when crossing safetyPercent")
	}
}

// TestTrigger_CountResetAfterCompress verifies that after compress fires via
// count trigger, the count window resets so the next batch of messageThreshold
// messages triggers again (not immediately).
func TestTrigger_CountResetAfterCompress(t *testing.T) {
	store := newMockStore()
	// contextWindow=1000, minPercent=10, normalPercent=90 (won't fire on tokens),
	// messageThreshold=5. 60-char messages stay above floor but below normalPercent.
	mgr := newTestManager(store,
		WithContextWindow(1000),
		WithMinPercent(10),
		WithNormalPercent(90),
		WithSafetyPercent(95),
		WithMessageThreshold(5),
	)

	callCount := 0
	mgr.SetTestCompressHook(func(_ bool) { callCount++ })

	ctx := context.Background()
	content := strings.Repeat("x", 100) // 25 tokens → 2.5% each

	// First batch: 5 messages → compress fires once.
	for i := 0; i < 5; i++ {
		if _, err := mgr.AddUserMessage(ctx, msgWithContent(content)); err != nil {
			t.Fatalf("AddUserMessage error: %v", err)
		}
	}
	if callCount != 1 {
		t.Fatalf("expected 1 compress call after first batch; got %d", callCount)
	}

	// Add 4 more messages: should NOT trigger (count since last compress = 4 < 5).
	for i := 0; i < 4; i++ {
		if _, err := mgr.AddUserMessage(ctx, msgWithContent(content)); err != nil {
			t.Fatalf("AddUserMessage error: %v", err)
		}
	}
	if callCount != 1 {
		t.Fatalf("expected still 1 compress call after 4 more messages; got %d", callCount)
	}

	// The 5th message since last compress should fire again.
	if _, err := mgr.AddUserMessage(ctx, msgWithContent(content)); err != nil {
		t.Fatalf("AddUserMessage error: %v", err)
	}
	if callCount != 2 {
		t.Fatalf("expected 2 compress calls after second batch of 5; got %d", callCount)
	}
}

// TestTrigger_NoContextWindow verifies that with contextWindow=0 no compression
// is attempted regardless of message count.
func TestTrigger_NoContextWindow(t *testing.T) {
	store := newMockStore()
	mgr := newTestManager(store,
		WithContextWindow(0),
		WithMessageThreshold(2),
	)

	called := false
	mgr.SetTestCompressHook(func(_ bool) { called = true })

	ctx := context.Background()
	for i := 0; i < 5; i++ {
		if _, err := mgr.AddUserMessage(ctx, msgWithContent("hello")); err != nil {
			t.Fatalf("AddUserMessage error: %v", err)
		}
	}

	if called {
		t.Error("compress should not be called when contextWindow is 0")
	}
}
