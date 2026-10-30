//! The Anthropic Messages API with tools (`/v1/messages`).

use serde_json::{Value, json};

use crate::{AiError, Message, Settings, ToolCall, ToolDef, Transport, arguments_object};

const VERSION: &str = "2023-06-01";

pub(crate) fn request_body(settings: &Settings, system: &str, messages: &[Message], tools: &[ToolDef]) -> Value {
    // Anthropic wants alternating user/assistant turns; consecutive tool results join one user turn.
    let mut out: Vec<Value> = Vec::new();
    for m in messages {
        let (role, block) = match m {
            Message::User { text } => ("user", vec![json!({"type": "text", "text": text})]),
            Message::Assistant { text, tool_calls } => {
                let mut blocks = Vec::new();
                if !text.is_empty() {
                    blocks.push(json!({"type": "text", "text": text}));
                }
                for c in tool_calls {
                    blocks.push(json!({"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}));
                }
                if blocks.is_empty() {
                    blocks.push(json!({"type": "text", "text": "…"}));
                }
                ("assistant", blocks)
            }
            Message::Tool { call_id, content, .. } => ("user", vec![json!({"type": "tool_result", "tool_use_id": call_id, "content": content})]),
        };
        match out.last_mut() {
            Some(last) if last.get("role").and_then(Value::as_str) == Some(role) => {
                if let Some(arr) = last.get_mut("content").and_then(Value::as_array_mut) {
                    arr.extend(block);
                }
            }
            _ => out.push(json!({"role": role, "content": block})),
        }
    }
    let mut body = json!({"model": settings.model.trim(), "system": system, "messages": out, "max_tokens": settings.max_tokens});
    if !tools.is_empty() {
        body["tools"] = tools.iter().map(|t| json!({"name": t.name, "description": t.description, "input_schema": t.schema})).collect();
    }
    body
}

pub(crate) fn parse_reply(v: &Value) -> Result<Message, AiError> {
    let Some(blocks) = v.get("content").and_then(Value::as_array) else {
        let detail = v.pointer("/error/message").and_then(Value::as_str).unwrap_or("no `content`");
        return Err(AiError::BadReply(detail.to_string()));
    };
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    for (i, b) in blocks.iter().enumerate() {
        match b.get("type").and_then(Value::as_str) {
            Some("text") => text.push_str(b.get("text").and_then(Value::as_str).unwrap_or("")),
            Some("tool_use") => {
                let Some(name) = b.get("name").and_then(Value::as_str) else { continue };
                let id = b.get("id").and_then(Value::as_str).map_or_else(|| format!("toolu_{i}"), str::to_string);
                tool_calls.push(ToolCall { id, name: name.to_string(), arguments: arguments_object(b.get("input")) });
            }
            _ => {}
        }
    }
    Ok(Message::Assistant { text, tool_calls })
}

pub(crate) fn chat(
    settings: &Settings,
    key: &str,
    transport: &dyn Transport,
    system: &str,
    messages: &[Message],
    tools: &[ToolDef],
) -> Result<Message, AiError> {
    let base = settings.base_url.trim().trim_end_matches('/');
    let base = base.strip_suffix("/v1").unwrap_or(base);
    let url = format!("{base}/v1/messages");
    let headers = [("x-api-key", key.to_string()), ("anthropic-version", VERSION.to_string())];
    let reply = transport.post_json(&url, &headers, &request_body(settings, system, messages, tools), settings.timeout_secs)?;
    parse_reply(&reply)
}
