// ctxengine
// License: MIT

package ctxengine

// Option is a functional option for configuring a Manager.
type Option func(*managerConfig)

// FailureDumpFunc writes a diagnostic snapshot of one failed summarization
// attempt. kind is the dump reason ("compress_fail"), meta carries the
// session, model, status and detail, and input/output are the JSON-encoded
// request and raw response.
type FailureDumpFunc func(kind string, meta map[string]any, input, output string) error

// RefusalClassifier decides whether a model reply that did not yield a valid
// summary was a content-policy refusal rather than a malformed answer, and
// returns a short human-readable detail when it was.
type RefusalClassifier func(finishReason, content string) (refused bool, detail string)

// NoiseKeyFunc identifies messages that are repeated fires of one source (a
// scheduled job, for instance) by a key: two messages with the same key are
// duplicates whatever their timestamps say, and runs of them collapse. ok is
// false for content that is not such a message.
type NoiseKeyFunc func(content string) (key string, ok bool)

// noNoiseKey is the default NoiseKeyFunc: nothing collapses.
func noNoiseKey(string) (string, bool) { return "", false }

const (
	defaultMinPercent       = 20
	defaultNormalPercent    = 50
	defaultSafetyPercent    = 80
	defaultMessageThreshold = 100
	// defaultRetainTokenPercent must stay clearly BELOW defaultMinPercent. When
	// the retained tail is the same size as the floor that gates compaction,
	// every pass shaves back to exactly the floor and the next message crosses it
	// again — the session compacts constantly without ever shrinking. See
	// validation rule (d) in New().
	defaultRetainTokenPercent    = 10
	defaultRetainMinMessages     = 2
	defaultMinCompressionGain    = 0.05
	defaultCooldownMessages      = 5
	defaultLargeMsgOffset        = 20
	defaultArchiveMessageCount   = 0
	defaultCompressTargetFactor  = 0.5
	defaultMinLoopGain           = 0.10
	defaultMaxCompressIterations = 3
	defaultOverheadTokens        = 4000
	defaultCharsPerToken         = 4.0
	defaultTokenSafetyMargin     = 1.0

	// defaultTriggerDays fires compaction once the oldest message in the live
	// window is older than this many days, regardless of how little of the
	// context window it occupies. Without it a low-volume session never reaches
	// any percentage threshold and its history sits in the window indefinitely.
	defaultTriggerDays = 7
	// defaultRetainMaxAgeDays caps the age of the retained tail: compaction
	// summarizes anything older, subject to retainMinMessages and the
	// last-user-message clamp. It is deliberately LOWER than defaultTriggerDays
	// so each pass buys (trigger - retain) days of quiet instead of pinning the
	// session to the trigger boundary and re-firing on every message.
	defaultRetainMaxAgeDays = 5

	// defaultRetainMaxTokens is an absolute ceiling on the retained tail, applied
	// alongside the percentage budget (the smaller wins).
	//
	// It exists because the percentage scales with the context window: 10% of a
	// 128k model is 12.8k tokens, but 10% of a million-token model is 100k, and
	// nothing about a larger window makes a larger tail more useful. On the
	// production instance this figure binds only on the two large-window agents
	// and is a no-op for the 128k ones, where the percentage already binds
	// tighter.
	//
	// Sized deliberately rather than aggressively. Once the request prefix is
	// stable the retained tail is cached and bills at a fraction of full price,
	// so trimming it buys far less than it did when every turn re-sent the whole
	// window — and the messages it trims are the RECENT ones. 40k keeps a few
	// hundred messages of working context for a busy agent while capturing most
	// of the available saving.
	defaultRetainMaxTokens = 40000

	// mediaTokensPerItem is the flat token cost charged per media item by the
	// estimator. Providers bill images by resolution tiles, so the rune length
	// of a base64 data: URI overstates the real cost by orders of magnitude;
	// a fixed per-item figure is far closer than either counting or ignoring it.
	mediaTokensPerItem = 1500

	// defaultMaxConsecutiveCompactFailures is the number of consecutive failed
	// automatic compactions after which the automatic compaction path is
	// suppressed for a session (the failure circuit breaker trips).
	defaultMaxConsecutiveCompactFailures = 3
	// defaultCompactFailureCooldownMessages is how many additional messages must
	// accumulate after the breaker trips before the automatic path is retried.
	defaultCompactFailureCooldownMessages = 20
)

// managerConfig holds resolved configuration for a Manager.
type managerConfig struct {
	minPercent         int
	normalPercent      int
	safetyPercent      int
	messageThreshold   int
	retainTokenPercent int
	retainMinMessages  int
	// targetPercent is the stop condition for the compaction loop: iterate until
	// the live window is below this percentage of the context window. 0 derives
	// it from normalPercent (see compressTargetPercent), preserving the historic
	// normalPercent*0.5 behaviour for configs that do not set it.
	targetPercent int
	// retainMaxTokens is an absolute ceiling on the retained tail, applied
	// alongside retainTokenPercent (the smaller wins). Percentages alone scale
	// with the window, so a 1M-token model inherits a tail budget tuned for
	// 128k and keeps an absurd absolute amount. 0 disables the cap.
	retainMaxTokens int
	// retainMaxAgeDays caps the age of the retained tail. 0 disables the cap.
	retainMaxAgeDays int
	// triggerDays fires compaction when the oldest live message exceeds this
	// age, bypassing minPercent. 0 disables the trigger.
	triggerDays   int
	compressModel ModelChain
	// caller is the host's summarization model; nil disables compaction.
	caller              ModelCaller
	archiveMessageCount int
	archiveDays         int
	// summaryMaxCount caps the number of stored summaries (keep newest N).
	// 0 (the default) disables the count cap.
	summaryMaxCount int
	// summaryRetentionDays deletes summaries older than N days. 0 (the default)
	// disables the age cutoff.
	summaryRetentionDays int
	archiveDir           string
	contextWindow        int
	overheadTokens       int
	maxSummaryTokens     int // 0 = use 20% of contextWindow at truncation time
	// charsPerToken is the divisor used to convert a rune count into an
	// estimated token count. Lower values estimate more tokens per character
	// (more conservative). Default: 4.0.
	charsPerToken float64
	// tokenSafetyMargin multiplies the token estimate so it errs high. A value
	// of 1.1 inflates the estimate by 10%, triggering compression slightly
	// earlier. Default: 1.0 (no inflation).
	tokenSafetyMargin float64
	// archiveContentMaxBytes caps per-message content stored in the archive.
	// 0 (the default) resolves to archiveContentMaxBytes at write time.
	archiveContentMaxBytes int
	notifyCallback         func(msg string)
	// reportCallback, when set, is invoked by the automatic compaction path with
	// the current call's channel/chatID and the formatted compaction report so it
	// can be delivered to the user. The manual /compact path returns the report
	// directly instead and does not use this callback.
	reportCallback func(channel, chatID, text string)
	// compactDebug enables verbatim request/response capture of each
	// summarization LLM invocation to <compressionProfileDir>/compact.jsonl.
	compactDebug bool
	// compressionProfileDir is the agent workspace directory. If non-empty and
	// a file named "COMPRESSION.md" (or legacy "compression.md") exists there, its content is appended to the
	// summarization prompt so agents can declare role-specific compression rules.
	compressionProfileDir string
	// failureDump, when set, receives the request + raw response of each FAILED
	// summarization attempt for diagnosis.
	failureDump FailureDumpFunc
	// refusalClassifier recognises content refusals in unusable replies. Nil
	// resolves to the built-in classifier.
	refusalClassifier RefusalClassifier
	// noiseKey identifies repeated fires of one source so runs collapse. Nil
	// resolves to noNoiseKey (no collapsing).
	noiseKey NoiseKeyFunc
	// eviction is the per-turn tool-result eviction policy. Defaults to
	// DefaultEvictionPolicy() (enabled); override via WithEvictionPolicy.
	eviction EvictionPolicy
}

func defaultManagerConfig() managerConfig {
	return managerConfig{
		minPercent:          defaultMinPercent,
		normalPercent:       defaultNormalPercent,
		safetyPercent:       defaultSafetyPercent,
		messageThreshold:    defaultMessageThreshold,
		retainTokenPercent:  defaultRetainTokenPercent,
		retainMinMessages:   defaultRetainMinMessages,
		retainMaxAgeDays:    defaultRetainMaxAgeDays,
		retainMaxTokens:     defaultRetainMaxTokens,
		triggerDays:         defaultTriggerDays,
		archiveMessageCount: defaultArchiveMessageCount,
		contextWindow:       128000,
		overheadTokens:      defaultOverheadTokens,
		charsPerToken:       defaultCharsPerToken,
		tokenSafetyMargin:   defaultTokenSafetyMargin,
		eviction:            DefaultEvictionPolicy(),
	}
}

// WithEvictionPolicy sets the per-turn tool-result eviction policy. The agent
// layer resolves per-agent + defaults config into a single EvictionPolicy and
// passes it here. When unset, DefaultEvictionPolicy() applies.
func WithEvictionPolicy(p EvictionPolicy) Option {
	return func(c *managerConfig) { c.eviction = p }
}

func WithMinPercent(pct int) Option {
	return func(c *managerConfig) { c.minPercent = pct }
}

func WithNormalPercent(pct int) Option {
	return func(c *managerConfig) { c.normalPercent = pct }
}

func WithSafetyPercent(pct int) Option {
	return func(c *managerConfig) { c.safetyPercent = pct }
}

func WithMessageThreshold(n int) Option {
	return func(c *managerConfig) { c.messageThreshold = n }
}

func WithRetainTokenPercent(pct int) Option {
	return func(c *managerConfig) { c.retainTokenPercent = pct }
}

func WithRetainMinMessages(n int) Option {
	return func(c *managerConfig) { c.retainMinMessages = n }
}

// WithTargetPercent sets the compaction loop's stop condition: iterate until the
// live window falls below this percentage of the context window. 0 restores the
// derived default (normalPercent * defaultCompressTargetFactor).
func WithTargetPercent(pct int) Option {
	return func(c *managerConfig) { c.targetPercent = pct }
}

// WithRetainMaxTokens sets an absolute ceiling on the retained tail, applied
// alongside the percentage budget (the smaller wins). 0 disables the cap.
func WithRetainMaxTokens(n int) Option {
	return func(c *managerConfig) { c.retainMaxTokens = n }
}

// WithRetainMaxAgeDays caps the age of the retained tail. Anything older is
// summarized, subject to the retainMinMessages floor and the last-user-message
// clamp. 0 disables the cap.
func WithRetainMaxAgeDays(d int) Option {
	return func(c *managerConfig) { c.retainMaxAgeDays = d }
}

// WithTriggerDays fires compaction once the oldest live message is older than d
// days. Unlike the percentage and count triggers it bypasses minPercent, so a
// low-volume session still ages out. 0 disables the trigger.
func WithTriggerDays(d int) Option {
	return func(c *managerConfig) { c.triggerDays = d }
}

// WithCompressModel records the model chain for stats and logging only.
func WithCompressModel(model ModelChain) Option {
	return func(c *managerConfig) { c.compressModel = model }
}

// WithModelCaller sets the host's summarization model. The host walks its own
// chain (fallbacks, cooldowns) inside Complete; the engine re-calls with a
// longer Exclude list when a reply is unusable. Without a caller, compaction
// reports "nothing" and never summarizes.
func WithModelCaller(c ModelCaller) Option {
	return func(cfg *managerConfig) { cfg.caller = c }
}

// WithFailureDump sets the sink for diagnostic snapshots of failed
// summarization attempts. Nil (the default) disables the dumps.
func WithFailureDump(fn FailureDumpFunc) Option {
	return func(c *managerConfig) { c.failureDump = fn }
}

// WithRefusalClassifier replaces the built-in content-refusal classifier.
func WithRefusalClassifier(fn RefusalClassifier) Option {
	return func(c *managerConfig) { c.refusalClassifier = fn }
}

// WithNoiseKey sets the function that recognises repeated fires of one source
// (the host's scheduled-job wrapper, say) so the tail and the summarizer input
// collapse runs of them. Without it no message is treated as a repeat.
func WithNoiseKey(fn NoiseKeyFunc) Option {
	return func(c *managerConfig) { c.noiseKey = fn }
}

func WithArchiveMessageCount(n int) Option {
	return func(c *managerConfig) { c.archiveMessageCount = n }
}

// WithArchiveDays limits the retrievable archive window to the last n days.
// 0 (the default) means no time-based limit.
func WithArchiveDays(n int) Option {
	return func(c *managerConfig) { c.archiveDays = n }
}

// WithSummaryMaxCount caps the number of stored summary checkpoints, keeping the
// newest n. 0 (the default) disables the count cap (falls back to days).
func WithSummaryMaxCount(n int) Option {
	return func(c *managerConfig) { c.summaryMaxCount = n }
}

// WithSummaryRetentionDays deletes stored summaries older than n days. 0 (the
// default) means no time-based limit.
func WithSummaryRetentionDays(n int) Option {
	return func(c *managerConfig) { c.summaryRetentionDays = n }
}

func WithContextWindow(tokens int) Option {
	return func(c *managerConfig) { c.contextWindow = tokens }
}

// WithArchiveDir sets the directory used to store per-session SQLite archive
// databases. The ContextManager derives the archive path as
// filepath.Join(dir, sanitizedKey+".archive.db") on first write.
// If dir is empty, archive writes are silently skipped.
func WithArchiveDir(dir string) Option {
	return func(c *managerConfig) { c.archiveDir = dir }
}

// WithOverheadTokens sets the fixed token overhead added to the post-Build token
// estimate in CheckAndCompress. This accounts for the system prompt, rendered
// summary, tool definitions, and completion budget combined. Default: 4000.
func WithOverheadTokens(n int) Option {
	return func(c *managerConfig) { c.overheadTokens = n }
}

// WithMaxSummaryTokens sets the maximum token budget for the serialized summary.
// If n == 0 (the default), the effective limit is 20% of contextWindow, computed
// at truncation time. After successful summarization the summary is truncated by
// removing the oldest key_moments and retrievable_history entries until it fits.
func WithMaxSummaryTokens(n int) Option {
	return func(c *managerConfig) { c.maxSummaryTokens = n }
}

func WithNotifyCallback(fn func(msg string)) Option {
	return func(c *managerConfig) { c.notifyCallback = fn }
}

// WithCompactionReporter sets the callback used by the automatic compaction path
// to deliver the formatted compaction report to the user's channel.
func WithCompactionReporter(fn func(channel, chatID, text string)) Option {
	return func(c *managerConfig) { c.reportCallback = fn }
}

// WithCompactDebug enables verbatim capture of each summarization request and
// response to <workspace>/compact.jsonl. Debugging only; off by default.
func WithCompactDebug(enabled bool) Option {
	return func(c *managerConfig) { c.compactDebug = enabled }
}

// WithCharsPerToken sets the divisor used to convert a rune count into an
// estimated token count. Lower values produce a higher (more conservative)
// token estimate. Values <= 0 are ignored and the default (4.0) is retained.
func WithCharsPerToken(v float64) Option {
	return func(c *managerConfig) {
		if v > 0 {
			c.charsPerToken = v
		}
	}
}

// WithTokenSafetyMargin sets the multiplier applied to every token estimate so
// it errs high, triggering compression earlier. A value of 1.1 inflates the
// estimate by 10%. Values <= 0 are ignored and the default (1.0) is retained.
func WithTokenSafetyMargin(v float64) Option {
	return func(c *managerConfig) {
		if v > 0 {
			c.tokenSafetyMargin = v
		}
	}
}

// WithArchiveContentMaxBytes sets the maximum per-message content size stored
// in the archive. Messages whose Content exceeds this are truncated before
// writing. Values <= 0 are ignored and the default (archiveContentMaxBytes) is
// used at write time.
func WithArchiveContentMaxBytes(n int) Option {
	return func(c *managerConfig) {
		if n > 0 {
			c.archiveContentMaxBytes = n
		}
	}
}

// WithCompressionProfileDir sets the agent workspace directory. If the file
// "COMPRESSION.md" (or legacy "compression.md") exists there it is appended verbatim to every summarization
// prompt, letting agents declare role-specific compression rules and structure.
func WithCompressionProfileDir(dir string) Option {
	return func(c *managerConfig) { c.compressionProfileDir = dir }
}

// CompressionSettings is a read-only snapshot of the compaction knobs an Option
// set resolves to. It exists so callers that translate configuration into
// Options — and the tests that guard them — can verify what those Options
// actually say, without reaching into Manager's unexported state.
type CompressionSettings struct {
	MinPercent         int
	NormalPercent      int
	SafetyPercent      int
	TargetPercent      int
	MessageThreshold   int
	TriggerDays        int
	RetainTokenPercent int
	RetainMaxTokens    int
	RetainMaxAgeDays   int
	RetainMinMessages  int
	ContextWindow      int
	OverheadTokens     int
	CharsPerToken      float64
	TokenSafetyMargin  float64
}

// SettingsFromOptions applies opts over the package defaults and reports the
// result. It deliberately performs none of New()'s validation clamps: it answers
// "what did these options ask for", which is the question a configuration
// mapper needs answered. New() remains the only place policy is enforced.
func SettingsFromOptions(opts ...Option) CompressionSettings {
	cfg := defaultManagerConfig()
	for _, o := range opts {
		o(&cfg)
	}
	return CompressionSettings{
		MinPercent:         cfg.minPercent,
		NormalPercent:      cfg.normalPercent,
		SafetyPercent:      cfg.safetyPercent,
		TargetPercent:      cfg.targetPercent,
		MessageThreshold:   cfg.messageThreshold,
		TriggerDays:        cfg.triggerDays,
		RetainTokenPercent: cfg.retainTokenPercent,
		RetainMaxTokens:    cfg.retainMaxTokens,
		RetainMaxAgeDays:   cfg.retainMaxAgeDays,
		RetainMinMessages:  cfg.retainMinMessages,
		ContextWindow:      cfg.contextWindow,
		OverheadTokens:     cfg.overheadTokens,
		CharsPerToken:      cfg.charsPerToken,
		TokenSafetyMargin:  cfg.tokenSafetyMargin,
	}
}
