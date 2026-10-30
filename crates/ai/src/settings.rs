//! Provider, endpoint, model and key.

use serde::{Deserialize, Serialize};

/// The wire protocol.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Provider {
    /// `POST {base}/chat/completions` with `tools`: OpenAI, OpenRouter, Groq, Ollama, LM Studio…
    #[default]
    OpenAiCompatible,
    /// `POST {base}/v1/messages`: Anthropic's Claude models.
    Anthropic,
}

impl Provider {
    /// The environment variable read when no key is saved, for the provider's default endpoint.
    pub fn key_env(self) -> Option<&'static str> {
        match self {
            Provider::OpenAiCompatible => Some("OPENAI_API_KEY"),
            Provider::Anthropic => Some("ANTHROPIC_API_KEY"),
        }
    }
}

/// A ready-made endpoint for the settings form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Preset {
    pub name: &'static str,
    pub provider: Provider,
    pub base_url: &'static str,
    /// A model to start from (the user can type any other the provider offers).
    pub model: &'static str,
    pub key_env: Option<&'static str>,
}

impl Preset {
    pub const ALL: [Preset; 7] = [
        Preset {
            name: "Google Gemini (free tier)",
            provider: Provider::OpenAiCompatible,
            base_url: "https://generativelanguage.googleapis.com/v1beta/openai",
            model: "gemini-3.8-flash",
            key_env: Some("GEMINI_API_KEY"),
        },
        Preset {
            name: "Groq (free tier)",
            provider: Provider::OpenAiCompatible,
            base_url: "https://api.groq.com/openai/v1",
            model: "openai/gpt-oss-120b",
            key_env: Some("GROQ_API_KEY"),
        },
        Preset {
            name: "Anthropic (Claude)",
            provider: Provider::Anthropic,
            base_url: "https://api.anthropic.com",
            model: "claude-sonnet-5-5",
            key_env: Some("ANTHROPIC_API_KEY"),
        },
        Preset { name: "OpenAI", provider: Provider::OpenAiCompatible, base_url: "https://api.openai.com/v1", model: "", key_env: Some("OPENAI_API_KEY") },
        Preset {
            name: "OpenRouter",
            provider: Provider::OpenAiCompatible,
            base_url: "https://openrouter.ai/api/v1",
            model: "",
            key_env: Some("OPENROUTER_API_KEY"),
        },
        Preset { name: "Ollama (local)", provider: Provider::OpenAiCompatible, base_url: "http://localhost:11434/v1", model: "qwen3", key_env: None },
        Preset { name: "LM Studio (local)", provider: Provider::OpenAiCompatible, base_url: "http://localhost:1234/v1", model: "", key_env: None },
    ];
}

/// The assistant's settings, saved as `ai-assistant.json` beside the preferences.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Settings {
    pub provider: Provider,
    pub base_url: String,
    pub model: String,
    /// Saved in plain text in the user's data folder; empty = read the provider's environment variable.
    pub api_key: String,
    /// Model requests per user message (each one may run several tools).
    pub max_steps: u32,
    /// Reply tokens per request.
    pub max_tokens: u32,
    /// Seconds to wait for one reply.
    pub timeout_secs: u64,
}

impl Default for Settings {
    fn default() -> Self {
        let p = Preset::ALL[0];
        Settings {
            provider: p.provider,
            base_url: p.base_url.into(),
            model: p.model.into(),
            api_key: String::new(),
            max_steps: 24,
            max_tokens: 4096,
            timeout_secs: 180,
        }
    }
}

impl Settings {
    /// Use `preset`'s protocol, endpoint and starting model (the saved key stays).
    pub fn apply(&mut self, preset: &Preset) {
        self.provider = preset.provider;
        self.base_url = preset.base_url.into();
        self.model = preset.model.into();
    }

    /// The preset these settings use, if they match one.
    pub fn preset(&self) -> Option<&'static Preset> {
        Preset::ALL.iter().find(|p| p.provider == self.provider && p.base_url.trim_end_matches('/') == self.base_url.trim().trim_end_matches('/'))
    }

    /// The endpoint is on this computer (a local model server, which needs no key).
    pub fn is_local(&self) -> bool {
        let rest = self.base_url.trim().split_once("://").map_or(self.base_url.trim(), |(_, r)| r);
        let host = rest.split(['/', '?']).next().unwrap_or("");
        let host = host.rsplit_once(':').map_or(host, |(h, port)| if port.chars().all(|c| c.is_ascii_digit()) { h } else { host });
        matches!(host, "localhost" | "127.0.0.1" | "[::1]")
    }

    pub fn needs_key(&self) -> bool {
        !self.is_local()
    }

    /// The saved key, else the matching environment variable.
    pub fn resolved_key(&self) -> String {
        let saved = self.api_key.trim();
        if !saved.is_empty() {
            return saved.to_string();
        }
        let env = self.preset().and_then(|p| p.key_env).or_else(|| self.provider.key_env());
        env.and_then(|v| std::env::var(v).ok()).map(|k| k.trim().to_string()).unwrap_or_default()
    }

    /// Bounds a hand-edited file can't break: at least one step, at most 100; sane token and time limits.
    pub fn clamped(mut self) -> Self {
        self.max_steps = self.max_steps.clamp(1, 100);
        self.max_tokens = self.max_tokens.clamp(256, 64_000);
        self.timeout_secs = self.timeout_secs.clamp(10, 1_800);
        self
    }

    /// Read from `path`; missing or unreadable files give the defaults.
    pub fn load(path: &std::path::Path) -> Self {
        use std::io::Read as _;
        const MAX_BYTES: u64 = 64 * 1024;
        let Ok(file) = std::fs::File::open(path) else { return Self::default() };
        let mut bytes = Vec::new();
        if file.take(MAX_BYTES + 1).read_to_end(&mut bytes).is_err() || bytes.len() as u64 > MAX_BYTES {
            return Self::default();
        }
        serde_json::from_slice::<Settings>(&bytes).map(Settings::clamped).unwrap_or_default()
    }

    /// Write to `path` (through a temporary file, so a crash never leaves half a file).
    pub fn save(&self, path: &std::path::Path) -> Result<(), String> {
        let json = serde_json::to_vec_pretty(self).map_err(|e| e.to_string())?;
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, json).map_err(|e| e.to_string())?;
        std::fs::rename(&tmp, path).map_err(|e| e.to_string())
    }
}
