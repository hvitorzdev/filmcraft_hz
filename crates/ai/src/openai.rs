//! OpenAI-compatible chat completions with tools (`/chat/completions`).

use serde_json::{Value, json};

use crate::{AiError, Message, Settings, ToolCall, ToolDef, Transport, arguments_object};

pub(crate) fn request_body(settings: &Settings, system: &str, messages: &[Message], tools: &[ToolDef]) -> Value {
    let mut out = vec![json!({"role": "system", "content": system})];
    for m in messages {
        out.push(match m {
            Message::User { text } => json!({"role": "user", "content": text}),
            Message::Assistant { text, tool_calls } => {
                let mut a = json!({"role": "assistant", "content": if text.is_empty() && !tool_calls.is_empty() { Value::Null } else { json!(text) }});
                if !tool_calls.is_empty() {
                    a["tool_calls"] = tool_calls
                        .iter()
                        .map(|c| {
                            let mut call = json!({"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments.to_string()}});
                            // Gemini needs its thought signature back exactly as it sent it
                            if let Some(extra) = &c.extra {
                                call["extra_content"] = extra.clone();
                            }
                            call
                        })
                        .collect();
                }
                a
            }
            Message::Tool { call_id, content, .. } => json!({"role": "tool", "tool_call_id": call_id, "content": content}),
        });
    }
    let mut body = json!({"model": settings.model.trim(), "messages": out, "max_tokens": settings.max_tokens});
    if !tools.is_empty() {
        body["tools"] =
            tools.iter().map(|t| json!({"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.schema}})).collect();
    }
    body
}

pub(crate) fn parse_reply(v: &Value) -> Result<Message, AiError> {
    let Some(msg) = v.pointer("/choices/0/message") else {
        let detail = v.pointer("/error/message").and_then(Value::as_str).unwrap_or("no `choices[0].message`");
        return Err(AiError::BadReply(detail.to_string()));
    };
    let text = match msg.get("content") {
        Some(Value::String(s)) => s.clone(),
        // some servers send content parts
        Some(Value::Array(parts)) => parts.iter().filter_map(|p| p.get("text").and_then(Value::as_str)).collect::<Vec<_>>().join(""),
        _ => String::new(),
    };
    let mut tool_calls = Vec::new();
    for (i, c) in msg.get("tool_calls").and_then(Value::as_array).into_iter().flatten().enumerate() {
        let Some(name) = c.pointer("/function/name").and_then(Value::as_str) else { continue };
        let id = c.get("id").and_then(Value::as_str).map_or_else(|| format!("call_{i}"), str::to_string);
        tool_calls.push(ToolCall {
            id,
            name: name.to_string(),
            arguments: arguments_object(c.pointer("/function/arguments")),
            extra: c.get("extra_content").filter(|v| v.is_object()).cloned(),
        });
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
    let url = format!("{base}/chat/completions");
    let mut headers = Vec::new();
    if !key.is_empty() {
        headers.push(("Authorization", format!("Bearer {key}")));
    }
    if base.contains("openrouter.ai") {
        headers.push(("X-Title", "FilmCraft".to_string()));
    }
    let reply = transport.post_json(&url, &headers, &request_body(settings, system, messages, tools), settings.timeout_secs)?;
    parse_reply(&reply)
}
