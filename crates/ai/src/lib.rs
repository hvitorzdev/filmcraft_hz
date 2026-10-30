//! FilmCraft's AI assistant core.
//!
//! The assistant takes a request in plain language ("remove the long pauses", "add TikTok-style
//! captions", "duck the music under my voice") and carries it out with FilmCraft's own engine
//! commands, the same ones the menus, the CLI, the control channel and MCP use. This crate is the
//! part that talks to the language model; it knows nothing about the editor:
//!
//! - [`Settings`]: which provider, endpoint, model and API key to use. Any OpenAI-compatible chat
//!   completions server works (OpenAI, OpenRouter, Groq, a local Ollama or LM Studio…), and so does
//!   the Anthropic Messages API.
//! - [`Message`], [`ToolCall`], [`ToolDef`]: a provider-neutral conversation with tool calling.
//! - [`chat`]: one request to the model through a [`Transport`] (HTTPS with [`HttpTransport`],
//!   or a fake in tests).
//! - [`run_turn`]: the agent loop. It sends the conversation, runs every tool the model asks for
//!   through a caller-supplied executor (the UI runs them on its own thread, so every edit is a
//!   normal undoable command), feeds the results back and stops when the model answers in text or
//!   the step budget runs out.
//!
//! Nothing here panics on hostile input (`AGENTS.md` §0): model replies are untrusted JSON, read
//! with `get`, and every size is capped.

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable))]

mod agent;
mod anthropic;
mod openai;
mod settings;
mod transport;

pub use agent::{Event, run_turn};
pub use settings::{Preset, Provider, Settings};
pub use transport::{HttpTransport, Transport};

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Longest text kept from one tool result before it goes back to the model (characters).
pub const MAX_TOOL_RESULT_CHARS: usize = 24_000;

/// One tool call the model asked for.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// The provider's id for the call; the result must carry it back.
    pub id: String,
    pub name: String,
    /// The arguments as JSON (an object; `{}` when the model sent none or unreadable ones).
    pub arguments: Value,
}

/// A tool the model may call.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolDef {
    pub name: &'static str,
    pub description: &'static str,
    /// JSON Schema of the arguments object.
    pub schema: Value,
}

/// One entry of the conversation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "camelCase")]
pub enum Message {
    User { text: String },
    Assistant { text: String, tool_calls: Vec<ToolCall> },
    Tool { call_id: String, name: String, content: String },
}

/// What went wrong talking to the model.
#[derive(Clone, Debug, PartialEq)]
pub enum AiError {
    /// The settings can't be used (no model, no key…). The text says what to fix.
    Config(String),
    /// The request didn't get an answer (network, TLS, timeout).
    Http(String),
    /// The server answered with an error status; `message` is the provider's explanation.
    Status { host: String, code: u16, message: String },
    /// The answer wasn't what the provider's API promises.
    BadReply(String),
    /// The user stopped the request.
    Cancelled,
}

impl std::fmt::Display for AiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AiError::Config(m) => write!(f, "{m}"),
            AiError::Http(m) => write!(f, "{m}"),
            AiError::Status { host, code, message } => write!(f, "{host} answered HTTP {code}: {message}"),
            AiError::BadReply(m) => write!(f, "unexpected reply from the model: {m}"),
            AiError::Cancelled => write!(f, "stopped"),
        }
    }
}

impl std::error::Error for AiError {}

impl AiError {
    /// A temporary condition worth retrying: rate limits, overload, a gateway hiccup, a timeout.
    pub fn is_transient(&self) -> bool {
        match self {
            AiError::Status { code, .. } => matches!(code, 408 | 429 | 500 | 502 | 503 | 504 | 529),
            AiError::Http(m) => m.contains("timed out") || m.contains("timeout"),
            _ => false,
        }
    }
}

/// Waits before each retry of a transient error (seconds); after the last one the error is returned.
pub const RETRY_WAITS: [u64; 3] = [3, 8, 20];

/// One request to the model: the system prompt, the conversation so far and the tools. Returns the
/// assistant's reply (text, tool calls or both). No retries; see [`chat_retrying`].
pub fn chat(settings: &Settings, transport: &dyn Transport, system: &str, messages: &[Message], tools: &[ToolDef]) -> Result<Message, AiError> {
    let model = settings.model.trim();
    if model.is_empty() {
        return Err(AiError::Config("choose a model in the assistant settings".into()));
    }
    let key = settings.resolved_key();
    if key.is_empty() && settings.needs_key() {
        let env = settings.preset().and_then(|p| p.key_env).or_else(|| settings.provider.key_env()).unwrap_or("the provider's key variable");
        return Err(AiError::Config(format!("add an API key in the assistant settings (or set {env} in the environment)")));
    }
    match settings.provider {
        Provider::OpenAiCompatible => openai::chat(settings, &key, transport, system, messages, tools),
        Provider::Anthropic => anthropic::chat(settings, &key, transport, system, messages, tools),
    }
}

/// [`chat`], retrying transient errors (busy or rate-limited servers) after the [`RETRY_WAITS`].
/// `wait(attempt, seconds, error)` is called before each retry and does the waiting; returning
/// false gives up (the user stopped) with [`AiError::Cancelled`].
pub fn chat_retrying(
    settings: &Settings,
    transport: &dyn Transport,
    system: &str,
    messages: &[Message],
    tools: &[ToolDef],
    wait: &mut dyn FnMut(usize, u64, &AiError) -> bool,
) -> Result<Message, AiError> {
    let mut attempt = 0usize;
    loop {
        match chat(settings, transport, system, messages, tools) {
            Err(e) if e.is_transient() => {
                let Some(&secs) = RETRY_WAITS.get(attempt) else { return Err(e) };
                attempt += 1;
                if !wait(attempt, secs, &e) {
                    return Err(AiError::Cancelled);
                }
            }
            other => return other,
        }
    }
}

/// `text` cut to at most `max` characters (on a character boundary), with a note when it was cut.
pub fn truncate(text: &str, max: usize) -> String {
    match text.char_indices().nth(max) {
        None => text.to_string(),
        Some((at, _)) => {
            let total = text.chars().count();
            format!("{}… [truncated: {max} of {total} characters shown]", text.get(..at).unwrap_or(""))
        }
    }
}

/// Arguments as a JSON object: a model sometimes sends a string with JSON in it, or nothing.
pub(crate) fn arguments_object(v: Option<&Value>) -> Value {
    match v {
        Some(Value::Object(_)) => v.cloned().unwrap_or(Value::Null),
        Some(Value::String(s)) if s.trim().is_empty() => Value::Object(Default::default()),
        Some(Value::String(s)) => match serde_json::from_str::<Value>(s) {
            Ok(o @ Value::Object(_)) => o,
            _ => serde_json::json!({ "_unparsed": s }),
        },
        _ => Value::Object(Default::default()),
    }
}

#[cfg(test)]
mod tests;
