//! AI Assistant panel: ask for an edit in plain language ("remove the long pauses", "add captions",
//! "duck the music under the voice") and a language model carries it out with FilmCraft's own
//! engine commands. Every edit is an ordinary command run on the UI thread, so it shows in History
//! and Edit ▸ Undo takes it back, step by step.
//!
//! The model runs on a worker thread (`filmcraft-ai`): it asks for tools, the worker hands each tool
//! call to the UI through a channel, [`pump`] runs it between frames (also while the panel is
//! hidden) and sends the result back. Any OpenAI-compatible server (OpenAI, OpenRouter, Ollama, LM
//! Studio…) or the Anthropic API works; the settings live in `ai-assistant.json` beside the
//! preferences.
//!
//! Automation ids: `ai.input`, `ai.send`, `ai.stop`, `ai.clear`, `ai.settingsToggle`,
//! `ai.suggestion.<n>`, `ai.preset`, `ai.baseUrl`, `ai.model`, `ai.apiKey`, `ai.save`.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::Duration;

use egui::{Rect, RichText, vec2};
use filmcraft_ai::{AiError, Event, HttpTransport, Message, Preset, Settings, ToolCall, ToolDef, Transport};
use serde_json::{Value, json};

use crate::FilmcraftApp;

/// The settings file, beside `preferences.json` in the data directory.
pub const SETTINGS_FILE: &str = "ai-assistant.json";
/// Longest wait for one UI-side tool result before the worker gives up on it.
const TOOL_TIMEOUT: Duration = Duration::from_secs(120);
/// Longest `wait_for_jobs` wait.
const MAX_JOB_WAIT_SECS: u64 = 600;
/// Commands the assistant may not run: they end the session or leave the editor.
const BLOCKED: &[&str] = &["app.quit", "file.exit", "file.close", "file.closeProject", "file.closeAll", "app.restart"];
/// Lines kept in the panel log.
const MAX_LOG: usize = 400;

/// One line of the panel's conversation view.
#[derive(Clone, Debug, PartialEq)]
pub enum LogEntry {
    User(String),
    Assistant(String),
    Tool { label: String, ok: Option<bool>, summary: String },
    Error(String),
    Info(String),
}

enum WorkerMsg {
    Event(Event),
    Tool(ToolCall, Sender<(bool, String)>),
    Done(Vec<Message>, Option<AiError>),
}

struct Worker {
    rx: Receiver<WorkerMsg>,
    cancel: Arc<AtomicBool>,
}

/// The panel's state (kept on the app, so a run continues while the panel is closed).
#[derive(Default)]
pub struct AiPanel {
    settings: Option<Settings>,
    /// The form being edited; saved with Save.
    draft: Option<Settings>,
    show_settings: bool,
    input: String,
    log: Vec<LogEntry>,
    /// The conversation sent to the model (moved to the worker while it runs).
    history: Vec<Message>,
    worker: Option<Worker>,
    /// Replaces the network (tests, or a host that brings its own client).
    transport: Option<Arc<dyn Transport>>,
}

impl AiPanel {
    pub fn running(&self) -> bool {
        self.worker.is_some()
    }

    /// Talk to the model through `t` instead of HTTPS.
    pub fn set_transport(&mut self, t: Arc<dyn Transport>) {
        self.transport = Some(t);
    }

    /// Use `s` for the next requests (not saved).
    pub fn set_settings(&mut self, s: Settings) {
        self.settings = Some(s.clamped());
    }

    /// The conversation view.
    pub fn log(&self) -> &[LogEntry] {
        &self.log
    }

    fn push(&mut self, e: LogEntry) {
        self.log.push(e);
        if self.log.len() > MAX_LOG {
            let extra = self.log.len().saturating_sub(MAX_LOG);
            self.log.drain(..extra);
        }
    }
}

fn settings_path(app: &FilmcraftApp) -> Option<std::path::PathBuf> {
    app.session.prefs_path.as_ref().and_then(|p| p.parent()).map(|d| d.join(SETTINGS_FILE))
}

fn settings(app: &mut FilmcraftApp) -> Settings {
    if app.ai.settings.is_none() {
        let s = settings_path(app).map(|p| Settings::load(&p)).unwrap_or_default();
        app.ai.settings = Some(s);
    }
    app.ai.settings.clone().unwrap_or_default()
}

// ------------------------------------------------------------------ tools

/// The tools the model gets. Everything goes through engine commands.
pub fn tools() -> Vec<ToolDef> {
    vec![
        ToolDef {
            name: "list_commands",
            description: "Search FilmCraft's engine commands. Returns lines `id | label | params | (disabled: why)`. Use words from the id or label, e.g. `transcript`, `caption`, `ducking`, `keyframe`, `transition`, `noise`. Call it before using a command whose parameters you don't know.",
            schema: json!({"type": "object", "properties": {
                "filter": {"type": "string", "description": "words that must all appear in the id or label (case-insensitive); empty lists everything"},
                "enabled_only": {"type": "boolean", "description": "only commands that can run now"}
            }}),
        },
        ToolDef {
            name: "run_command",
            description: "Run one FilmCraft engine command by id with its JSON params (see list_commands for the params format). Edits are undoable. Times are in ticks: 254016000000 ticks = 1 second. Returns the command's JSON result or an error.",
            schema: json!({"type": "object", "properties": {
                "id": {"type": "string"},
                "params": {"type": "object", "description": "the command's parameters ({} when it takes none)"}
            }, "required": ["id"]}),
        },
        ToolDef {
            name: "inspect_sequence",
            description: "The active sequence: video/audio tracks, clips with ids, start/end and source in/out (ticks and frames), speed, gain, effect names, transitions, markers, playhead, selection and settings (frame rate). Pass detail=true to also get every effect parameter value.",
            schema: json!({"type": "object", "properties": {"detail": {"type": "boolean", "description": "include effect parameters (large)"}}}),
        },
        ToolDef {
            name: "inspect_project",
            description: "The project: bins and media items with ids, types and durations, and the active sequence.",
            schema: json!({"type": "object", "properties": {}}),
        },
        ToolDef {
            name: "inspect_transcript",
            description: "The sequence transcript (after transcript.generate): words with indexes, sequence times, speakers, paragraphs.",
            schema: json!({"type": "object", "properties": {}}),
        },
        ToolDef {
            name: "wait_for_jobs",
            description: "Wait until background jobs (transcription, downloads, scene detection, exports, renders) finish, then return their results. Use it after a command that starts a job.",
            schema: json!({"type": "object", "properties": {"max_seconds": {"type": "integer", "description": "give up after this many seconds (default 300)"}}}),
        },
    ]
}

/// The instructions for the model.
pub const SYSTEM_PROMPT: &str = r#"You are the AI editing assistant built into FilmCraft, a non-linear video editor modelled on Premiere Pro. The user asks for edits in plain language (often in Portuguese); you carry them out by calling FilmCraft's engine commands with your tools, then reply briefly in the user's language saying what you changed.

How to work:
- Look before you edit: call inspect_sequence (and inspect_project when you need media ids) to get clip ids, tracks and times. Never invent ids.
- Find commands with list_commands; run them with run_command. Check each result; if a command fails, read the error, fix the params or try another approach. Don't repeat a failing call unchanged.
- Time is in ticks: 1 second = 254016000000 ticks. Convert seconds to ticks exactly (integers). Frames: see the sequence frame rate in inspect_sequence.
- Every edit is undoable (Ctrl+Z / Edit ▸ Undo). Do what was asked and nothing more; for large destructive changes on unclear requests, ask first in one short question.
- If FilmCraft can't do something (generative video, object removal, recolouring a specific object, AI-generated B-roll), say so plainly and offer the closest thing it can do. Never claim an edit you didn't make.

Recipes (verify params with list_commands when unsure):
- Speech-based editing needs a transcript: run transcript.generate {"language":"auto"} (it may start a job: then wait_for_jobs). Models: transcript.models / transcript.downloadModel.
- Remove silences / long pauses: transcript.removePauses {"minSeconds":0.7,"keepSeconds":0.15}.
- Remove filler words (um, uh, "tipo", "né", "éé"): transcript.removeFillers {"fillers":[...]} — include Portuguese fillers when the speech is Portuguese.
- Cut a sentence or mistakes/retakes: inspect_transcript, find the word range (or transcript.search {"query":...}), then transcript.extract {"from":i,"to":j} (ripple). For retakes, keep the last complete take of a repeated sentence.
- Captions / subtitles: transcript.createCaptions {"maxChars":..,"lines":1|2}; short 1-line captions (maxChars ~16-20) for TikTok/Reels style. Caption commands: captions.* (list them).
- Reorder clips: timeline.move {"moves":[{"clip":id,"track":"V1","time":ticks}],"insert":true}.
- Cut at a time: timeline.razor {"time":ticks,"track":"V1"}; ripple-delete ranges: mark in/out then sequence.extract.
- Exact duration: trim clips, change speed (clip.speedDuration), or for music clip.remix.
- Slow zoom-in: effects.setKeyframe on the clip's "motion" effect, param "scale" (100 = original size), e.g. 100 at the clip's sourceIn and 115 at its sourceOut (mediaTime is in source ticks; inspect_sequence with detail=true shows effect params).
- Transitions: sequence.applyVideoTransition {"clip":id,"effect":"cross_dissolve","edge":"in"|"out"}; list effects with effects.list {"kind":"VideoTransition"}.
- Effects: effects.list then effects.apply {"clips":[id],"effect":name}; effects.setParam to change values.
- Audio: lower music under the voice with the ducking.* commands (Essential Sound; tag dialogue/music first if required), noise removal with repair.noise.*, voice clarity with clarity.*, loudness with loudness.*, gain with clip.audioGain {"clips":[id],"db":-20,"mode":"set"}.
- Scene cuts: clip.sceneEditDetection.
Keep answers short: what you did, and anything you could not do."#;

/// Run one tool on the UI thread.
fn execute_tool(app: &mut FilmcraftApp, ctx: &egui::Context, call: &ToolCall) -> (bool, String) {
    let a = &call.arguments;
    let r: Result<Value, String> = match call.name.as_str() {
        "list_commands" => {
            list_commands(app, a.get("filter").and_then(Value::as_str).unwrap_or(""), a.get("enabled_only").and_then(Value::as_bool).unwrap_or(false))
        }
        "run_command" => {
            let Some(id) = a.get("id").and_then(Value::as_str) else { return (false, "missing `id`".into()) };
            if BLOCKED.contains(&id) || id.starts_with("app.language.") {
                return (false, format!("`{id}` is not available to the assistant"));
            }
            let params = match a.get("params") {
                None | Some(Value::Null) => json!({}),
                Some(Value::String(s)) => serde_json::from_str(s).unwrap_or_else(|_| json!({})),
                Some(p) => p.clone(),
            };
            crate::menus::invoke(app, ctx, id, params)
        }
        "inspect_sequence" => app.session.execute("sequence.inspect", json!({})).map_err(|e| e.to_string()).map(|mut v| {
            if a.get("detail").and_then(Value::as_bool) != Some(true) {
                compact_effects(&mut v, 0);
            }
            v
        }),
        "inspect_project" => app.session.execute("project.inspect", json!({})).map_err(|e| e.to_string()),
        "inspect_transcript" => app.session.execute("transcript.inspect", json!({})).map_err(|e| e.to_string()),
        // the worker polls this one; here it is the single poll
        "jobs_poll" => app.session.execute("jobs.list", json!({})).map_err(|e| e.to_string()),
        other => Err(format!("unknown tool `{other}`")),
    };
    match r {
        Ok(Value::Null) => (true, "ok".into()),
        Ok(v) => (true, v.to_string()),
        Err(e) => (false, e),
    }
}

/// Replace every clip's `effects` list with the effect names (and how many keyframes each has), so
/// the sequence fits the model's context; `detail` asks for the full values.
fn compact_effects(v: &mut Value, depth: usize) {
    if depth > 16 {
        return;
    }
    match v {
        Value::Object(o) => {
            for (k, child) in o.iter_mut() {
                if k == "effects"
                    && let Value::Array(list) = child
                {
                    let names: Vec<Value> = list
                        .iter()
                        .map(|e| {
                            let name = e.get("effect").and_then(Value::as_str).unwrap_or("?");
                            let keys: u64 = e
                                .get("params")
                                .and_then(Value::as_object)
                                .map(|p| p.values().filter_map(|x| x.get("keyframes").and_then(Value::as_u64)).fold(0u64, u64::saturating_add))
                                .unwrap_or(0);
                            if keys > 0 { json!(format!("{name} ({keys} keyframes)")) } else { json!(name) }
                        })
                        .collect();
                    *child = Value::Array(names);
                } else {
                    compact_effects(child, depth + 1);
                }
            }
        }
        Value::Array(list) => list.iter_mut().for_each(|x| compact_effects(x, depth + 1)),
        _ => {}
    }
}

fn list_commands(app: &mut FilmcraftApp, filter: &str, enabled_only: bool) -> Result<Value, String> {
    let all = app.session.execute("command.list", json!({})).map_err(|e| e.to_string())?;
    let words: Vec<String> = filter.split_whitespace().map(str::to_lowercase).collect();
    let mut lines = Vec::new();
    let mut total = 0usize;
    for c in all.as_array().into_iter().flatten() {
        let id = c.get("id").and_then(Value::as_str).unwrap_or("");
        let label = c.get("label").and_then(Value::as_str).unwrap_or("");
        let enabled = c.get("enabled").and_then(Value::as_bool).unwrap_or(false);
        let hay = format!("{id} {label}").to_lowercase();
        if (enabled_only && !enabled) || !words.iter().all(|w| hay.contains(w.as_str())) || BLOCKED.contains(&id) {
            continue;
        }
        total += 1;
        if lines.len() < 120 {
            let params = c.get("params").and_then(Value::as_str).unwrap_or("{}");
            lines.push(format!("{id} | {label} | {params}{}", if enabled { "" } else { " | (disabled now)" }));
        }
    }
    if lines.is_empty() {
        return Ok(json!("no command matches; try fewer or other words"));
    }
    let more = if total > lines.len() { format!("\n… {} more; narrow the filter", total.saturating_sub(lines.len())) } else { String::new() };
    Ok(Value::String(format!("{}{more}", lines.join("\n"))))
}

// ------------------------------------------------------------------ worker

fn start(app: &mut FilmcraftApp, ctx: &egui::Context, text: String) {
    if app.ai.running() || text.trim().is_empty() {
        return;
    }
    let s = settings(app);
    app.ai.push(LogEntry::User(text.clone()));
    let (tx, rx) = channel::<WorkerMsg>();
    let cancel = Arc::new(AtomicBool::new(false));
    let mut history = std::mem::take(&mut app.ai.history);
    let transport: Arc<dyn Transport> = app.ai.transport.clone().unwrap_or_else(|| Arc::new(HttpTransport));
    let (cancel2, ctx2) = (Arc::clone(&cancel), ctx.clone());
    let spawned = std::thread::Builder::new().name("ai-assistant".into()).spawn(move || {
        let tools = tools();
        let wake = || ctx2.request_repaint();
        let ask_ui = |call: &ToolCall| -> (bool, String) {
            let (rtx, rrx) = channel();
            if tx.send(WorkerMsg::Tool(call.clone(), rtx)).is_err() {
                return (false, "the editor closed".into());
            }
            wake();
            rrx.recv_timeout(TOOL_TIMEOUT).unwrap_or_else(|_| (false, "the editor didn't answer in time".into()))
        };
        let mut execute = |call: &ToolCall| -> (bool, String) {
            if call.name != "wait_for_jobs" {
                return ask_ui(call);
            }
            let max = call.arguments.get("max_seconds").and_then(Value::as_u64).unwrap_or(300).clamp(1, MAX_JOB_WAIT_SECS);
            let poll = ToolCall { id: call.id.clone(), name: "jobs_poll".into(), arguments: json!({}), extra: None };
            let started = std::time::Instant::now();
            loop {
                let (ok, text) = ask_ui(&poll);
                if !ok {
                    return (false, text);
                }
                let jobs: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
                let running = jobs.as_array().into_iter().flatten().any(|j| j.get("finished").and_then(Value::as_bool) == Some(false));
                if !running {
                    return (true, text);
                }
                if cancel2.load(Ordering::Relaxed) {
                    return (false, "stopped by the user".into());
                }
                if started.elapsed().as_secs() >= max {
                    return (true, format!("still running after {max} s: {text}"));
                }
                std::thread::sleep(Duration::from_millis(750));
            }
        };
        let mut on_event = |e: Event| {
            let _ = tx.send(WorkerMsg::Event(e));
            wake();
        };
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            filmcraft_ai::run_turn(&s, transport.as_ref(), SYSTEM_PROMPT, &tools, &mut history, &text, &mut execute, &mut on_event, &cancel2)
        }));
        let err = match r {
            Ok(Ok(())) => None,
            Ok(Err(e)) => Some(e),
            Err(_) => Some(AiError::BadReply("the assistant stopped unexpectedly".into())),
        };
        let _ = tx.send(WorkerMsg::Done(history, err));
        wake();
    });
    match spawned {
        Ok(_) => app.ai.worker = Some(Worker { rx, cancel }),
        Err(e) => app.ai.push(LogEntry::Error(format!("could not start the assistant: {e}"))),
    }
}

fn tool_label(call: &ToolCall) -> String {
    match call.name.as_str() {
        "run_command" => call.arguments.get("id").and_then(Value::as_str).unwrap_or("?").to_string(),
        other => other.to_string(),
    }
}

/// Run queued tool calls and collect the worker's progress. Called every frame by the app.
pub fn pump(app: &mut FilmcraftApp, ctx: &egui::Context) {
    let msgs: Vec<WorkerMsg> = match app.ai.worker.as_ref() {
        Some(w) => w.rx.try_iter().collect(),
        None => return,
    };
    for m in msgs {
        match m {
            WorkerMsg::Tool(call, reply) => {
                let out = execute_tool(app, ctx, &call);
                let _ = reply.send(out);
            }
            WorkerMsg::Event(Event::Text(t)) => app.ai.push(LogEntry::Assistant(t)),
            WorkerMsg::Event(Event::ToolStarted(call)) => {
                if call.name != "jobs_poll" {
                    app.ai.push(LogEntry::Tool { label: tool_label(&call), ok: None, summary: String::new() });
                }
            }
            WorkerMsg::Event(Event::ToolFinished { call, ok, summary }) => {
                let label = tool_label(&call);
                if let Some(LogEntry::Tool { label: l, ok: o, summary: s }) = app.ai.log.iter_mut().rev().find(|e| matches!(e, LogEntry::Tool { ok: None, .. }))
                    && *l == label
                {
                    *o = Some(ok);
                    *s = summary;
                }
            }
            WorkerMsg::Event(Event::Retrying { attempt, seconds, reason }) => {
                let n = filmcraft_ai::RETRY_WAITS.len();
                app.ai.push(LogEntry::Info(tlf!(
                    "The server is busy ({reason}). Trying again in {s} s ({a} of {n})…",
                    reason = reason,
                    s = seconds,
                    a = attempt,
                    n = n
                )));
            }
            WorkerMsg::Event(Event::StepLimit(n)) => app.ai.push(LogEntry::Info(tlf!("Stopped after {n} steps; ask again to continue.", n = n))),
            WorkerMsg::Done(history, err) => {
                app.ai.history = history;
                app.ai.worker = None;
                match err {
                    None => {}
                    Some(AiError::Cancelled) => app.ai.push(LogEntry::Info(tl!("Stopped.").to_string())),
                    Some(e) => app.ai.push(LogEntry::Error(e.to_string())),
                }
                // a long chat costs tokens on every request; keep the recent part
                trim_history(&mut app.ai.history);
            }
        }
    }
    if app.ai.running() {
        ctx.request_repaint_after(Duration::from_millis(200));
    }
}

/// Keep at most the last 12 user requests (cut at a user message, so tool calls keep their results).
fn trim_history(h: &mut Vec<Message>) {
    let users: Vec<usize> = h.iter().enumerate().filter(|(_, m)| matches!(m, Message::User { .. })).map(|(i, _)| i).collect();
    if users.len() > 12
        && let Some(&cut) = users.get(users.len().saturating_sub(12))
    {
        h.drain(..cut);
    }
}

// ------------------------------------------------------------------ UI

const SUGGESTIONS: [&str; 6] = [
    "Remove the silences and long pauses",
    "Remove filler words and stutters",
    "Generate short captions, TikTok style",
    "Lower the music to -20 dB while I'm talking",
    "Clean the background noise from my voice",
    "Add a slow zoom-in to the selected clip",
];

pub fn show(app: &mut FilmcraftApp, ui: &mut egui::Ui, rect: Rect) {
    let ctx = ui.ctx().clone();
    pump(app, &ctx);
    let t = app.tokens;
    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(rect.shrink2(vec2(10.0, 8.0))));
    let ui = &mut child;
    let s = settings(app);

    // header
    ui.horizontal(|ui| {
        let model = if s.model.trim().is_empty() { tl!("no model chosen").to_string() } else { s.model.clone() };
        let source = s.preset().map_or(s.base_url.as_str(), |p| p.name);
        ui.label(RichText::new(format!("{model} · {source}")).color(t.text_dim).small());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let r = ui.selectable_label(app.ai.show_settings, tl!("Settings"));
            if r.clicked() {
                app.ai.show_settings = !app.ai.show_settings;
                app.ai.draft = app.ai.show_settings.then(|| s.clone());
            }
            app.auto.add("ai.settingsToggle", r.rect, tl!("Settings"));
            let r = ui.add_enabled(!app.ai.running() && !app.ai.log.is_empty(), egui::Button::new(tl!("New chat")));
            if r.clicked() {
                app.ai.log.clear();
                app.ai.history.clear();
            }
            app.auto.add("ai.clear", r.rect, tl!("New chat"));
        });
    });
    ui.separator();

    if app.ai.show_settings {
        settings_form(app, ui);
        ui.separator();
    }

    // composer at the bottom, log above it
    let composer_h = 86.0;
    let log_h = (ui.available_height() - composer_h).max(40.0);
    egui::ScrollArea::vertical().id_salt("ai-log").max_height(log_h).auto_shrink([false, false]).stick_to_bottom(true).show(ui, |ui| {
        if app.ai.log.is_empty() {
            ui.label(
                RichText::new(tl!("Ask for an edit in your own words. The assistant uses FilmCraft's commands, and every change can be undone."))
                    .color(t.text_dim),
            );
            ui.add_space(6.0);
            for (n, sug) in SUGGESTIONS.iter().enumerate() {
                let label = crate::i18n::t(sug);
                let r = ui.add_enabled(!app.ai.running(), egui::Button::new(label).wrap());
                if r.clicked() {
                    app.ai.input = label.to_string();
                }
                app.auto.add(&format!("ai.suggestion.{n}"), r.rect, label);
            }
        }
        for e in &app.ai.log {
            match e {
                LogEntry::User(x) => {
                    ui.add_space(4.0);
                    ui.label(RichText::new(x).strong().color(t.text));
                }
                LogEntry::Assistant(x) => {
                    ui.label(RichText::new(x).color(t.text));
                }
                LogEntry::Tool { label, ok, summary } => {
                    let (mark, color) = match ok {
                        None => ("…", t.text_faint),
                        Some(true) => ("✔", t.text_dim),
                        Some(false) => ("✖", egui::Color32::from_rgb(0xd9, 0x6a, 0x5f)),
                    };
                    let r = ui.label(RichText::new(format!("{mark} {label}")).monospace().small().color(color));
                    if !summary.is_empty() {
                        r.on_hover_text(summary);
                    }
                }
                LogEntry::Error(x) => {
                    ui.label(RichText::new(format!("✖ {x}")).color(egui::Color32::from_rgb(0xd9, 0x6a, 0x5f)));
                }
                LogEntry::Info(x) => {
                    ui.label(RichText::new(x).italics().color(t.text_dim));
                }
            }
        }
        if app.ai.running() {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(RichText::new(tl!("Working…")).color(t.text_dim));
            });
        }
    });

    ui.add_space(4.0);
    let running = app.ai.running();
    let r = ui.add_enabled(
        !running,
        egui::TextEdit::multiline(&mut app.ai.input)
            .id_salt("ai-input")
            .hint_text(tl!("What should I do? (Enter sends, Shift+Enter adds a line)"))
            .desired_rows(2)
            .desired_width(f32::INFINITY),
    );
    let label: String = app.ai.input.chars().take(200).collect();
    app.auto.add("ai.input", r.rect, &label);
    let enter = r.has_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter) && !i.modifiers.shift);
    ui.horizontal(|ui| {
        if running {
            let r = ui.button(tl!("Stop"));
            if r.clicked()
                && let Some(w) = app.ai.worker.as_ref()
            {
                w.cancel.store(true, Ordering::Relaxed);
            }
            app.auto.add("ai.stop", r.rect, tl!("Stop"));
        } else {
            let r = ui.add_enabled(!app.ai.input.trim().is_empty(), egui::Button::new(tl!("Send")));
            app.auto.add("ai.send", r.rect, tl!("Send"));
            if r.clicked() || enter {
                // Enter inserted a newline before we saw it
                let text = app.ai.input.trim().to_string();
                app.ai.input.clear();
                start(app, &ctx, text);
            }
        }
    });
}

fn settings_form(app: &mut FilmcraftApp, ui: &mut egui::Ui) {
    let t = app.tokens;
    let mut d = app.ai.draft.clone().unwrap_or_else(|| settings(app));
    egui::Grid::new("ai-settings").num_columns(2).spacing(vec2(8.0, 6.0)).show(ui, |ui| {
        ui.label(tl!("Provider"));
        let current = d.preset().map_or(tl!("Custom"), |p| p.name);
        let r = craft_ui::ComboBox::from_id_salt("ai-preset")
            .selected_text(current)
            .show_ui(ui, |ui| {
                for p in Preset::ALL.iter() {
                    if ui.selectable_label(d.preset() == Some(p), p.name).clicked() {
                        d.apply(p);
                    }
                }
            })
            .response;
        app.auto.add("ai.preset", r.rect, current);
        ui.end_row();

        ui.label(tl!("Protocol"));
        ui.horizontal(|ui| {
            ui.radio_value(&mut d.provider, filmcraft_ai::Provider::OpenAiCompatible, tl!("OpenAI-compatible"));
            ui.radio_value(&mut d.provider, filmcraft_ai::Provider::Anthropic, "Anthropic");
        });
        ui.end_row();

        ui.label(tl!("Server address"));
        let r = ui.add(egui::TextEdit::singleline(&mut d.base_url).desired_width(f32::INFINITY));
        app.auto.add("ai.baseUrl", r.rect, &d.base_url);
        ui.end_row();

        ui.label(tl!("Model"));
        let r = ui.add(egui::TextEdit::singleline(&mut d.model).hint_text(tl!("e.g. a model id from your provider")).desired_width(f32::INFINITY));
        app.auto.add("ai.model", r.rect, &d.model);
        ui.end_row();

        ui.label(tl!("API key"));
        let env = d.preset().and_then(|p| p.key_env).or_else(|| d.provider.key_env()).unwrap_or("");
        let hint = if d.is_local() { tl!("not needed for a server on this computer").to_string() } else { tlf!("or set {var} in the environment", var = env) };
        let r = ui.add(egui::TextEdit::singleline(&mut d.api_key).password(true).hint_text(hint).desired_width(f32::INFINITY));
        app.auto.add("ai.apiKey", r.rect, if d.api_key.is_empty() { "" } else { "••••" });
        ui.end_row();

        ui.label(tl!("Steps per request"));
        ui.add(egui::DragValue::new(&mut d.max_steps).range(1..=100));
        ui.end_row();
    });
    ui.label(RichText::new(tl!("The key is saved in plain text in FilmCraft's settings folder on this computer. Requests send your instructions and the project structure (names, clip times, transcript) to the provider you choose; media files are never uploaded.")).small().color(t.text_faint));
    ui.horizontal(|ui| {
        let r = ui.button(tl!("Save"));
        app.auto.add("ai.save", r.rect, tl!("Save"));
        if r.clicked() {
            let s = d.clone().clamped();
            if let Some(Err(e)) = settings_path(app).map(|p| s.save(&p)) {
                app.ui.status = tlf!("Couldn't save the assistant settings: {error}", error = e);
            }
            app.ai.settings = Some(s);
            app.ai.show_settings = false;
            app.ai.draft = None;
            return;
        }
        if ui.button(tl!("Cancel")).clicked() {
            app.ai.show_settings = false;
            app.ai.draft = None;
            return;
        }
        app.ai.draft = Some(d);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn history_is_trimmed_at_a_user_message() {
        let mut h = Vec::new();
        for i in 0..20 {
            h.push(Message::User { text: i.to_string() });
            h.push(Message::Assistant {
                text: String::new(),
                tool_calls: vec![ToolCall { id: "c".into(), name: "x".into(), arguments: json!({}), extra: None }],
            });
            h.push(Message::Tool { call_id: "c".into(), name: "x".into(), content: String::new() });
        }
        trim_history(&mut h);
        assert_eq!(h.len(), 36);
        assert_eq!(h.first(), Some(&Message::User { text: "8".into() }));
    }

    #[test]
    fn tool_schemas_are_objects_and_names_unique() {
        let t = tools();
        let mut names: Vec<_> = t.iter().map(|d| d.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), t.len());
        assert!(t.iter().all(|d| d.schema["type"] == "object"));
    }
}
