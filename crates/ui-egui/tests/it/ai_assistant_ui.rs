//! Headless test of the AI Assistant panel with a scripted model: Window ▸ AI Assistant opens it, a
//! request typed into `ai.input` and sent runs the tools the model asks for as ordinary undoable
//! commands on the UI thread, a blocked command is refused, and the answer lands in the log.

use std::sync::mpsc::{Sender, channel};
use std::sync::{Arc, Mutex};

use egui_kittest::Harness;
use filmcraft_ai::{AiError, Provider, Settings, Transport};
use filmcraft_engine::Session;
use filmcraft_ui_egui::FilmcraftApp;
use filmcraft_ui_egui::control::ControlRequest;
use filmcraft_ui_egui::panels::ai::LogEntry;
use serde_json::{Value, json};

/// Replies in order, recording each request body.
struct Script {
    replies: Mutex<Vec<Value>>,
    bodies: Mutex<Vec<Value>>,
}

impl Transport for Script {
    fn post_json(&self, _url: &str, _h: &[(&str, String)], body: &Value, _t: u64) -> Result<Value, AiError> {
        self.bodies.lock().unwrap().push(body.clone());
        self.replies.lock().unwrap().pop().ok_or_else(|| AiError::Http("script ended".into()))
    }
}

fn call(id: &str, name: &str, args: Value) -> Value {
    json!({"id": id, "type": "function", "function": {"name": name, "arguments": args.to_string()}})
}

struct Driver {
    harness: Harness<'static, FilmcraftApp>,
    tx: Sender<ControlRequest>,
}

impl Driver {
    fn demo() -> Self {
        let mut session = Session::default();
        session.execute("file.openDemoProject", json!({})).expect("demo project");
        let (tx, rx) = channel();
        let app = FilmcraftApp::new(session).with_control(rx);
        let harness = Harness::builder().with_size(egui::vec2(1600.0, 980.0)).with_max_steps(10_000).build_eframe(move |_cc| app);
        let mut d = Driver { harness, tx };
        d.frames(4);
        d
    }

    fn frames(&mut self, n: usize) {
        for _ in 0..n {
            let ctx = self.harness.ctx.clone();
            let mut raw = std::mem::take(self.harness.input_mut());
            eframe::App::raw_input_hook(self.harness.state_mut(), &ctx, &mut raw);
            *self.harness.input_mut() = raw;
            self.harness.step();
        }
    }

    fn ok(&mut self, method: &str, params: Value) -> Value {
        let (req, reply) = ControlRequest::new(method, params.clone());
        self.tx.send(req).unwrap();
        for _ in 0..600 {
            self.frames(1);
            if let Ok(v) = reply.try_recv() {
                assert_eq!(v["ok"], json!(true), "{method} {params} failed: {v}");
                return v["result"].clone();
            }
        }
        panic!("no reply to {method} {params}");
    }

    fn app(&mut self) -> &mut FilmcraftApp {
        self.harness.state_mut()
    }
}

#[test]
fn assistant_runs_model_tool_calls_as_undoable_commands() {
    let mut d = Driver::demo();
    d.ok("ui.menu.invoke", json!({"id": "window.panel.AiAssistant"}));
    d.frames(3);
    assert!(!d.ok("ui.elements", json!({"prefix": "ai.input"})).as_array().unwrap().is_empty(), "the panel shows its input");

    // the scripted model: look, refuse-worthy call, an edit, then the answer
    let replies = vec![
        json!({"choices": [{"message": {"content": null, "tool_calls": [call("c1", "inspect_sequence", json!({})), call("c2", "list_commands", json!({"filter": "notes"}))]}}]}),
        json!({"choices": [{"message": {"content": "", "tool_calls": [call("c3", "run_command", json!({"id": "app.quit"})), call("c4", "run_command", json!({"id": "project.setNotes", "params": {"text": "Cortes feitos pela IA"}}))]}}]}),
        json!({"choices": [{"message": {"content": "Pronto: anotei nas notas do projeto."}}]}),
    ];
    let script = Arc::new(Script { replies: Mutex::new(replies.into_iter().rev().collect()), bodies: Mutex::new(Vec::new()) });
    let undo0 = d.app().session.history.undo.len();
    {
        let app = d.app();
        app.ai.set_transport(script.clone());
        app.ai.set_settings(Settings {
            provider: Provider::OpenAiCompatible,
            base_url: "http://localhost:9/v1".into(),
            model: "fake".into(),
            ..Settings::default()
        });
    }
    d.ok("ui.click", json!({"id": "ai.input"}));
    d.ok("ui.type", json!({"text": "anote nas notas"}));
    d.frames(2);
    d.ok("ui.click", json!({"id": "ai.send"}));

    let t0 = std::time::Instant::now();
    while (d.app().ai.running() || d.app().ai.log().is_empty()) && t0.elapsed().as_secs() < 60 {
        std::thread::sleep(std::time::Duration::from_millis(5));
        d.frames(1);
    }
    assert!(!d.app().ai.running(), "the request finished");
    assert_eq!(d.app().session.project.notes, "Cortes feitos pela IA");
    assert_eq!(d.app().session.history.undo.len(), undo0 + 1, "one undoable edit");

    let log = d.app().ai.log().to_vec();
    assert!(matches!(log.first(), Some(LogEntry::User(t)) if t == "anote nas notas"), "{log:?}");
    assert!(log.iter().any(|e| matches!(e, LogEntry::Tool { label, ok: Some(false), .. } if label == "app.quit")), "quit is refused: {log:?}");
    assert!(log.iter().any(|e| matches!(e, LogEntry::Tool { label, ok: Some(true), .. } if label == "project.setNotes")), "{log:?}");
    assert!(matches!(log.last(), Some(LogEntry::Assistant(t)) if t.starts_with("Pronto")), "{log:?}");

    // the model saw real results: the sequence JSON and the matching command line
    let bodies = script.bodies.lock().unwrap().clone();
    let second = bodies[1]["messages"].as_array().unwrap();
    let results: Vec<&str> = second.iter().filter(|m| m["role"] == "tool").filter_map(|m| m["content"].as_str()).collect();
    assert!(results[0].contains("\"video\"") && results[0].contains("\"Ocean_Sunset.mp4\""), "{}", results[0]);
    assert!(!results[0].contains("vignette_amount"), "effect params are left out unless asked: {}", results[0]);
    assert!(results[1].contains("project.setNotes |"), "{}", results[1]);
    let third = bodies[2]["messages"].as_array().unwrap();
    assert!(third.iter().any(|m| m["role"] == "tool" && m["content"].as_str().is_some_and(|c| c.starts_with("error:") && c.contains("app.quit"))));

    d.ok("engine.execute", json!({"command": "edit.undo", "params": {}}));
    d.frames(2);
    assert_eq!(d.app().session.project.notes, "");
}
