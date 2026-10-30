use std::sync::Mutex;
use std::sync::atomic::AtomicBool;

use serde_json::{Value, json};

use super::*;

/// Replies with canned JSON in order and records every request.
struct Fake {
    replies: Mutex<Vec<Value>>,
    seen: Mutex<Vec<(String, Vec<(String, String)>, Value)>>,
}

impl Fake {
    fn new(replies: Vec<Value>) -> Self {
        Fake { replies: Mutex::new(replies.into_iter().rev().collect()), seen: Mutex::new(Vec::new()) }
    }
}

impl Transport for Fake {
    fn post_json(&self, url: &str, headers: &[(&str, String)], body: &Value, _t: u64) -> Result<Value, AiError> {
        self.seen.lock().unwrap().push((url.to_string(), headers.iter().map(|(k, v)| (k.to_string(), v.clone())).collect(), body.clone()));
        self.replies.lock().unwrap().pop().ok_or_else(|| AiError::Http("no more replies".into()))
    }
}

fn tools() -> Vec<ToolDef> {
    vec![ToolDef { name: "run_command", description: "Run a command", schema: json!({"type": "object", "properties": {"id": {"type": "string"}}}) }]
}

fn openai_settings() -> Settings {
    Settings {
        provider: Provider::OpenAiCompatible,
        base_url: "https://api.example.com/v1/".into(),
        model: "m".into(),
        api_key: "k".into(),
        ..Settings::default()
    }
}

#[test]
fn openai_turn_runs_tools_and_feeds_results_back() {
    let fake = Fake::new(vec![
        json!({"choices": [{"message": {"content": null, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "run_command", "arguments": "{\"id\":\"transcript.removePauses\"}"}}]}}]}),
        json!({"choices": [{"message": {"content": "Pronto: removi as pausas."}}]}),
    ]);
    let mut history = Vec::new();
    let mut ran = Vec::new();
    let mut events = Vec::new();
    run_turn(
        &openai_settings(),
        &fake,
        "sys",
        &tools(),
        &mut history,
        "remova as pausas",
        &mut |c| {
            ran.push(c.arguments["id"].as_str().unwrap_or("").to_string());
            (true, "{\"removed\":3}".into())
        },
        &mut |e| events.push(e),
        &AtomicBool::new(false),
    )
    .unwrap();
    assert_eq!(ran, ["transcript.removePauses"]);
    assert_eq!(history.len(), 4, "{history:?}");
    assert!(matches!(events.last(), Some(Event::Text(t)) if t.contains("Pronto")));
    let seen = fake.seen.lock().unwrap();
    assert_eq!(seen[0].0, "https://api.example.com/v1/chat/completions");
    assert!(seen[0].1.iter().any(|(k, v)| k == "Authorization" && v == "Bearer k"));
    let second = &seen[1].2["messages"];
    assert_eq!(second[2]["tool_calls"][0]["function"]["name"], "run_command");
    assert_eq!(second[3]["role"], "tool");
    assert_eq!(second[3]["tool_call_id"], "c1");
    assert_eq!(seen[0].2["tools"][0]["function"]["name"], "run_command");
}

#[test]
fn anthropic_merges_tool_results_into_one_user_turn() {
    let s = Settings {
        provider: Provider::Anthropic,
        base_url: "https://api.anthropic.com/v1".into(),
        model: "claude".into(),
        api_key: "k".into(),
        ..Settings::default()
    };
    let fake = Fake::new(vec![
        json!({"content": [{"type": "text", "text": "Vou fazer."}, {"type": "tool_use", "id": "t1", "name": "run_command", "input": {"id": "a"}}, {"type": "tool_use", "id": "t2", "name": "run_command", "input": {"id": "b"}}]}),
        json!({"content": [{"type": "text", "text": "Feito."}]}),
    ]);
    let mut history = Vec::new();
    run_turn(&s, &fake, "sys", &tools(), &mut history, "x", &mut |_| (false, "nope".into()), &mut |_| {}, &AtomicBool::new(false)).unwrap();
    let seen = fake.seen.lock().unwrap();
    assert_eq!(seen[0].0, "https://api.anthropic.com/v1/messages", "a /v1 suffix isn't doubled");
    assert!(seen[0].1.iter().any(|(k, _)| k == "anthropic-version"));
    let msgs = seen[1].2["messages"].as_array().unwrap();
    assert_eq!(msgs.len(), 3, "user, assistant, one user turn with both results: {msgs:?}");
    assert_eq!(msgs[2]["content"].as_array().unwrap().len(), 2);
    assert_eq!(msgs[2]["content"][0]["content"], "error: nope");
    assert_eq!(seen[1].2["system"], "sys");
}

#[test]
fn hostile_replies_are_errors_not_panics() {
    for v in [json!(null), json!([]), json!({"choices": []}), json!({"choices": [{"message": 5}]}), json!({"error": {"message": "bad key"}})] {
        let _ = openai::parse_reply(&v);
        let _ = anthropic::parse_reply(&v);
    }
    assert_eq!(openai::parse_reply(&json!({"error": {"message": "bad key"}})), Err(AiError::BadReply("bad key".into())));
    // arguments that aren't JSON stay visible to the tool instead of vanishing
    let m = openai::parse_reply(&json!({"choices": [{"message": {"tool_calls": [{"function": {"name": "x", "arguments": "{oops"}}]}}]})).unwrap();
    let Message::Assistant { tool_calls, .. } = m else { panic!() };
    assert_eq!(tool_calls[0].arguments["_unparsed"], "{oops");
    assert_eq!(tool_calls[0].id, "call_0");
}

#[test]
fn step_limit_and_cancel_stop_the_loop() {
    let call = json!({"choices": [{"message": {"tool_calls": [{"id": "c", "function": {"name": "run_command", "arguments": "{}"}}]}}]});
    let fake = Fake::new(vec![call.clone(), call.clone(), call]);
    let s = Settings { max_steps: 2, ..openai_settings() };
    let mut events = Vec::new();
    run_turn(&s, &fake, "", &tools(), &mut Vec::new(), "x", &mut |_| (true, String::new()), &mut |e| events.push(e), &AtomicBool::new(false)).unwrap();
    assert_eq!(events.last(), Some(&Event::StepLimit(2)));
    let r = run_turn(&s, &fake, "", &tools(), &mut Vec::new(), "x", &mut |_| (true, String::new()), &mut |_| {}, &AtomicBool::new(true));
    assert_eq!(r, Err(AiError::Cancelled));
}

#[test]
fn config_errors_say_what_to_fix() {
    let fake = Fake::new(vec![]);
    let no_model = Settings { model: " ".into(), ..openai_settings() };
    assert!(matches!(chat(&no_model, &fake, "", &[], &tools()), Err(AiError::Config(m)) if m.contains("model")));
    let local = Settings { base_url: "http://localhost:11434/v1".into(), api_key: String::new(), ..openai_settings() };
    assert!(local.is_local() && !local.needs_key());
    assert!(!Settings { base_url: "https://localhost.evil.example/v1".into(), ..openai_settings() }.is_local());
}

#[test]
fn truncate_keeps_char_boundaries() {
    assert_eq!(truncate("ação", 10), "ação");
    assert!(truncate("ááááá", 2).starts_with("áá…"));
}

#[test]
fn settings_round_trip_and_clamp() {
    let dir = std::env::temp_dir().join(format!("filmcraft-ai-test-{}", std::process::id()));
    let path = dir.join("ai-assistant.json");
    let s = Settings { model: "x".into(), max_steps: 7, ..Settings::default() };
    s.save(&path).unwrap();
    assert_eq!(Settings::load(&path), s);
    std::fs::write(&path, r#"{"maxSteps": 100000, "maxTokens": 1}"#).unwrap();
    let c = Settings::load(&path);
    assert_eq!((c.max_steps, c.max_tokens), (100, 256));
    std::fs::write(&path, "not json").unwrap();
    assert_eq!(Settings::load(&path), Settings::default());
    let _ = std::fs::remove_dir_all(dir);
}

/// The real HTTP client against a one-shot local server: the request arrives as JSON with the
/// headers, a 200 reply is parsed, and an error status carries the server's message.
#[cfg(not(target_arch = "wasm32"))]
#[test]
fn http_transport_posts_json_and_reports_http_errors() {
    use std::io::{BufRead, BufReader, Read, Write};
    fn serve(status: &str, body: &str) -> (String, std::thread::JoinHandle<String>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/v1", listener.local_addr().unwrap());
        let (status, body) = (status.to_string(), body.to_string());
        let h = std::thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            let mut r = BufReader::new(s.try_clone().unwrap());
            let mut head = String::new();
            let mut len = 0usize;
            loop {
                let mut line = String::new();
                r.read_line(&mut line).unwrap();
                if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                    len = v.trim().parse().unwrap();
                }
                head.push_str(&line);
                if line == "\r\n" {
                    break;
                }
            }
            let mut req = vec![0u8; len];
            r.read_exact(&mut req).unwrap();
            write!(s, "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
            format!("{head}{}", String::from_utf8(req).unwrap())
        });
        (url, h)
    }
    let (url, h) = serve("200 OK", r#"{"choices":[{"message":{"content":"olá"}}]}"#);
    let s = Settings { provider: Provider::OpenAiCompatible, base_url: url, model: "m".into(), api_key: "sk-test".into(), ..Settings::default() };
    let reply = chat(&s, &HttpTransport, "sys", &[Message::User { text: "oi".into() }], &[]).unwrap();
    assert_eq!(reply, Message::Assistant { text: "olá".into(), tool_calls: vec![] });
    let req = h.join().unwrap();
    assert!(req.starts_with("POST /v1/chat/completions"), "{req}");
    assert!(req.to_ascii_lowercase().contains("authorization: bearer sk-test"), "{req}");
    assert!(req.contains(r#""content":"oi""#), "{req}");

    let (url, h) = serve("401 Unauthorized", r#"{"error":{"message":"Incorrect API key"}}"#);
    let s = Settings { base_url: url, ..s };
    let err = chat(&s, &HttpTransport, "", &[], &[]).unwrap_err();
    h.join().unwrap();
    assert!(matches!(&err, AiError::Status { code: 401, message, .. } if message == "Incorrect API key"), "{err:?}");
    assert!(!err.to_string().contains("sk-test") && err.to_string().contains("401"));
}

#[test]
fn busy_servers_are_retried_then_reported() {
    let busy = || AiError::Status { host: "h".into(), code: 503, message: "high demand".into() };
    struct Flaky(Mutex<Vec<Result<Value, AiError>>>);
    impl Transport for Flaky {
        fn post_json(&self, _: &str, _: &[(&str, String)], _: &Value, _: u64) -> Result<Value, AiError> {
            self.0.lock().unwrap().pop().unwrap_or_else(|| Err(AiError::Http("empty".into())))
        }
    }
    let ok = json!({"choices": [{"message": {"content": "oi"}}]});
    // two busy answers, then success: the turn succeeds and reports both retries
    let t = Flaky(Mutex::new(vec![Ok(ok), Err(busy()), Err(busy())]));
    let mut events = Vec::new();
    run_turn(&openai_settings(), &t, "", &[], &mut Vec::new(), "x", &mut |_| (true, String::new()), &mut |e| events.push(e), &AtomicBool::new(false)).unwrap();
    let retries: Vec<_> = events.iter().filter(|e| matches!(e, Event::Retrying { .. })).collect();
    assert_eq!(retries.len(), 2, "{events:?}");
    assert!(matches!(retries[0], Event::Retrying { attempt: 1, seconds: 3, reason } if reason.contains("high demand")));
    // always busy: gives up after the last wait with the server's error
    let t = Flaky(Mutex::new(vec![Err(busy()), Err(busy()), Err(busy()), Err(busy())]));
    let r = run_turn(&openai_settings(), &t, "", &[], &mut Vec::new(), "x", &mut |_| (true, String::new()), &mut |_| {}, &AtomicBool::new(false));
    assert_eq!(r, Err(busy()));
    // a bad key is not retried
    let bad = AiError::Status { host: "h".into(), code: 401, message: "no".into() };
    let t = Flaky(Mutex::new(vec![Ok(json!({})), Err(bad.clone())]));
    assert_eq!(run_turn(&openai_settings(), &t, "", &[], &mut Vec::new(), "x", &mut |_| (true, String::new()), &mut |_| {}, &AtomicBool::new(false)), Err(bad));
}

#[test]
fn error_bodies_become_one_line_messages() {
    let google = r#"[{
  "error": {
    "code": 503,
    "message": "This model is currently experiencing high demand.\nPlease try again later.",
    "status": "UNAVAILABLE"
  }
}]"#;
    assert_eq!(transport::error_message(google), "This model is currently experiencing high demand. Please try again later.");
    assert_eq!(transport::error_message(r#"{"error":{"message":"bad key"}}"#), "bad key");
    assert_eq!(transport::error_message(r#"{"detail":"nope"}"#), "nope");
    assert_eq!(transport::error_message("<html>\n Bad Gateway </html>"), "<html> Bad Gateway </html>");
}

/// Gemini 3 (OpenAI-compatible) sends a thought signature with each tool call and rejects the next
/// request unless it comes back unchanged on that call.
#[test]
fn gemini_thought_signatures_go_back_with_their_calls() {
    let sig = json!({"google": {"thought_signature": "CvcQAdHN2OekY10ClPFkYA=="}});
    let fake = Fake::new(vec![
        json!({"choices": [{"message": {"tool_calls": [{"id": "a", "type": "function", "function": {"name": "run_command", "arguments": "{}"}, "extra_content": sig}]}}]}),
        json!({"choices": [{"message": {"content": "ok"}}]}),
    ]);
    let mut history = Vec::new();
    run_turn(&openai_settings(), &fake, "", &tools(), &mut history, "x", &mut |_| (true, "done".into()), &mut |_| {}, &AtomicBool::new(false)).unwrap();
    let seen = fake.seen.lock().unwrap();
    let call = &seen[1].2["messages"][2]["tool_calls"][0];
    assert_eq!(call["extra_content"], sig, "{call}");
    // providers without it get no extra field
    assert!(seen[0].2["messages"].as_array().unwrap().iter().all(|m| m.get("extra_content").is_none()));
}
