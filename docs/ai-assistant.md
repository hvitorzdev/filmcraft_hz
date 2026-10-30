# AI Assistant

**Window ▸ AI Assistant** opens a panel where you ask for edits in plain language, in any language
the model understands:

- "Remove the silences and long pauses"
- "Cut the filler words and stutters"
- "Generate short captions, TikTok style"
- "Lower the music to -20 dB while I'm talking"
- "Put clip 3 before clip 1"
- "Add a slow zoom-in to the selected clip"

A language model reads the request, inspects the project and runs FilmCraft's own engine commands
(the same ones the menus, the CLI, the control channel and MCP use). Every change is an ordinary
command: it shows in History and **Edit ▸ Undo** takes it back step by step. The panel log shows
each command the assistant ran (✔ worked, ✖ failed; hover for the result).

## Choosing a model

Click **Settings** in the panel:

| Provider | Server address | Key |
|---|---|---|
| Anthropic (Claude) | `https://api.anthropic.com` | `ANTHROPIC_API_KEY` or the key field |
| OpenAI | `https://api.openai.com/v1` | `OPENAI_API_KEY` or the key field |
| OpenRouter | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` or the key field |
| Ollama (local, free) | `http://localhost:11434/v1` | none |
| LM Studio (local, free) | `http://localhost:1234/v1` | none |

Any other server that speaks the OpenAI chat completions API with tool calling works too: pick
**OpenAI-compatible** and type its address. Choose a model that supports tool (function) calling;
larger models plan multi-step edits much better than small local ones.

Settings are saved in `ai-assistant.json` beside FilmCraft's preferences. **The key is stored in
plain text there**; leave the field empty to read it from the environment variable instead.

## Privacy

Each request sends your instructions plus what the assistant asks to look at: the project
structure (item and clip names, ids, times, effects), the command list and the transcript text.
Media files are never uploaded. With a local server (Ollama, LM Studio) nothing leaves the
computer. The assistant can't quit the app, close the project or change the interface language.

## What it can and can't do

It can do anything FilmCraft has a command for: text-based editing from the transcript (remove
pauses and filler words, cut sentences, captions; transcription runs locally with Whisper in builds
with the `whisper` feature), cuts, moves, trims, speed, transitions, effects and keyframes, audio
gain, ducking, noise reduction, clarity and loudness, scene-edit detection, music remix to a
duration.

It can't do what FilmCraft has no engine for yet: generative video (object removal, recolouring a
specific object, AI-generated B-roll), beat detection for cutting to music. The assistant says so
instead of pretending.

## How it works

- `crates/ai` (`filmcraft-ai`, L2) is the model client: `Settings`, a provider-neutral
  conversation with tool calls, `chat` for OpenAI-compatible and Anthropic endpoints over a
  `Transport` (rustls HTTPS; unavailable on the web build), and `run_turn`, the agent loop.
- `crates/ui-egui/src/panels/ai.rs` is the panel. A request runs on a worker thread; every tool
  call is handed to the UI thread, where `panels::ai::pump` (called each frame, also while the panel
  is closed) runs it and sends the result back. The tools are `list_commands`, `run_command`,
  `inspect_sequence`, `inspect_project`, `inspect_transcript` and `wait_for_jobs`.
- The system prompt (`SYSTEM_PROMPT`) holds recipes that map common requests to commands. Adding a
  recipe there, or a new engine command, is how the assistant learns a new trick.

Automation ids: `ai.input`, `ai.send`, `ai.stop`, `ai.clear`, `ai.settingsToggle`,
`ai.suggestion.<n>`, `ai.preset`, `ai.baseUrl`, `ai.model`, `ai.apiKey`, `ai.save`.
