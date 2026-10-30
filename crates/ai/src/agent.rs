//! The agent loop: ask the model, run the tools it wants, repeat until it answers.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::{AiError, MAX_TOOL_RESULT_CHARS, Message, Settings, ToolCall, ToolDef, Transport, chat, truncate};

/// Progress of a turn, for the UI.
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    /// The model said something (possibly alongside tool calls).
    Text(String),
    /// A tool is about to run.
    ToolStarted(ToolCall),
    /// A tool finished; `ok` is false when it reported an error.
    ToolFinished { call: ToolCall, ok: bool, summary: String },
    /// The step budget ran out before the model answered.
    StepLimit(u32),
}

/// Run one user request to completion.
///
/// `history` holds the conversation so far and receives every new message, so the next request
/// can refer back ("now make it shorter"). `execute` runs one tool and returns `(ok, result)`.
/// `cancel` is polled between steps. On error the history keeps what happened before it.
#[allow(clippy::too_many_arguments)]
pub fn run_turn(
    settings: &Settings,
    transport: &dyn Transport,
    system: &str,
    tools: &[ToolDef],
    history: &mut Vec<Message>,
    user_text: &str,
    execute: &mut dyn FnMut(&ToolCall) -> (bool, String),
    on_event: &mut dyn FnMut(Event),
    cancel: &AtomicBool,
) -> Result<(), AiError> {
    history.push(Message::User { text: user_text.to_string() });
    for _ in 0..settings.max_steps.max(1) {
        if cancel.load(Ordering::Relaxed) {
            return Err(AiError::Cancelled);
        }
        let reply = chat(settings, transport, system, history, tools)?;
        let Message::Assistant { text, tool_calls } = &reply else {
            return Err(AiError::BadReply("not an assistant message".into()));
        };
        let (text, tool_calls) = (text.clone(), tool_calls.clone());
        history.push(reply);
        if !text.trim().is_empty() {
            on_event(Event::Text(text.trim().to_string()));
        }
        if tool_calls.is_empty() {
            return Ok(());
        }
        for call in tool_calls {
            if cancel.load(Ordering::Relaxed) {
                // every call needs a result, or the provider rejects the next request
                history.push(Message::Tool { call_id: call.id.clone(), name: call.name.clone(), content: "error: stopped by the user".into() });
                continue;
            }
            on_event(Event::ToolStarted(call.clone()));
            let (ok, result) = execute(&call);
            let content = truncate(&result, MAX_TOOL_RESULT_CHARS);
            on_event(Event::ToolFinished { call: call.clone(), ok, summary: truncate(&result, 160) });
            history.push(Message::Tool { call_id: call.id, name: call.name, content: if ok { content } else { format!("error: {content}") } });
        }
    }
    on_event(Event::StepLimit(settings.max_steps));
    Ok(())
}
