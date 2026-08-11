package inference

// ConverseOption configures a Converse call.
type ConverseOption func(*ConverseOptions)

// ConverseOptions holds per-call overrides.
type ConverseOptions struct {
	MaxTokens int32 // 0 means use the client default
	// ThinkingBudget > 0 requests extended thinking with that budget. Zero
	// expresses no preference and leaves the model's own default in place,
	// which on current models means thinking is ON. Use ThinkingDisabled to
	// turn it off.
	ThinkingBudget int32
	// ThinkingDisabled explicitly turns extended thinking off rather than
	// deferring to the model default. Reasoning tokens are billed against
	// MaxTokens, so a call with a tight MaxTokens must say so to avoid
	// spending its whole output budget before emitting any text.
	ThinkingDisabled bool
}

// DefaultMaxTokens is the default output token limit when not overridden.
const DefaultMaxTokens = 4096

// DefaultThinkingBudget is the standard extended thinking budget used across
// compose and ask. Providers map this to their native mechanism (Anthropic
// extended thinking, Bedrock thinking config, OpenAI reasoning effort).
const DefaultThinkingBudget = 16000

// Apply returns the merged options.
func Apply(opts []ConverseOption) ConverseOptions {
	var o ConverseOptions
	for _, fn := range opts {
		fn(&o)
	}
	return o
}

// WithMaxTokens caps the output token count for a single call.
func WithMaxTokens(n int32) ConverseOption {
	return func(o *ConverseOptions) { o.MaxTokens = n }
}

// WithThinking enables extended thinking with the given budget.
// MaxTokens is automatically increased to accommodate the thinking budget
// on top of the text output allocation, matching AI SDK behavior.
func WithThinking(budgetTokens int32) ConverseOption {
	return func(o *ConverseOptions) { o.ThinkingBudget = budgetTokens }
}

// WithoutThinking disables extended thinking for a single call instead of
// inheriting the model default. Use it for mechanical calls that gain nothing
// from reasoning, especially alongside a small WithMaxTokens, since providers
// bill reasoning tokens against the output limit.
func WithoutThinking() ConverseOption {
	return func(o *ConverseOptions) { o.ThinkingDisabled = true }
}

// StreamDelta is a chunk of streamed output from the model.
type StreamDelta struct {
	Text     string
	Thinking bool // true for reasoning tokens, false for response tokens
}

// StreamFunc receives streaming deltas as they arrive from the model.
type StreamFunc func(StreamDelta)

// Pricing holds per-token costs for a model.
type Pricing struct {
	InputPerToken  float64
	OutputPerToken float64
}

// ComputeCost returns the dollar cost for the given token counts.
func (p Pricing) ComputeCost(inputTokens, outputTokens int) float64 {
	return float64(inputTokens)*p.InputPerToken + float64(outputTokens)*p.OutputPerToken
}

// Usage tracks token consumption and cost from an LLM call.
type Usage struct {
	InputTokens  int
	OutputTokens int
	cost         float64 // accumulated dollar cost
}

// NewUsage creates a Usage with the given token counts and cost.
func NewUsage(inputTokens, outputTokens int, cost float64) Usage {
	return Usage{InputTokens: inputTokens, OutputTokens: outputTokens, cost: cost}
}

// Cost returns the estimated dollar cost for this usage.
func (u Usage) Cost() float64 {
	return u.cost
}

// Add combines two Usage values.
func (u Usage) Add(other Usage) Usage {
	return Usage{
		InputTokens:  u.InputTokens + other.InputTokens,
		OutputTokens: u.OutputTokens + other.OutputTokens,
		cost:         u.cost + other.cost,
	}
}
