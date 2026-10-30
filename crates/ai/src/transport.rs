//! Sending one JSON request.

use serde_json::Value;

use crate::AiError;

/// Posts a JSON body and returns the JSON reply. The real one is [`HttpTransport`]; tests use fakes.
pub trait Transport: Send + Sync {
    fn post_json(&self, url: &str, headers: &[(&str, String)], body: &Value, timeout_secs: u64) -> Result<Value, AiError>;
}

/// Largest reply accepted (bytes).
#[cfg(not(target_arch = "wasm32"))]
const MAX_REPLY: u64 = 8 * 1024 * 1024;

/// HTTPS (and plain HTTP for a model server on this computer) with rustls.
#[derive(Clone, Copy, Debug, Default)]
pub struct HttpTransport;

#[cfg(not(target_arch = "wasm32"))]
impl Transport for HttpTransport {
    fn post_json(&self, url: &str, headers: &[(&str, String)], body: &Value, timeout_secs: u64) -> Result<Value, AiError> {
        use std::io::Read as _;
        use std::time::Duration;

        let provider = std::sync::Arc::new(rustls_rustcrypto::provider());
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .tls_config(
                ureq::tls::TlsConfig::builder()
                    .provider(ureq::tls::TlsProvider::Rustls)
                    .root_certs(ureq::tls::RootCerts::PlatformVerifier)
                    .unversioned_rustls_crypto_provider(provider)
                    .build(),
            )
            .timeout_connect(Some(Duration::from_secs(20)))
            .timeout_global(Some(Duration::from_secs(timeout_secs)))
            // read error bodies too: they say what's wrong (bad key, unknown model…)
            .http_status_as_error(false)
            .build()
            .into();
        let bytes = serde_json::to_vec(body).map_err(|e| AiError::BadReply(e.to_string()))?;
        let mut req = agent.post(url).header("Content-Type", "application/json");
        for (k, v) in headers {
            req = req.header(*k, v.as_str());
        }
        let resp = req.send(&bytes[..]).map_err(|e| AiError::Http(format!("couldn't reach {} ({e})", host(url))))?;
        let status = resp.status().as_u16();
        let mut text = String::new();
        resp.into_body()
            .into_reader()
            .take(MAX_REPLY)
            .read_to_string(&mut text)
            .map_err(|e| AiError::Http(format!("reading the reply from {} failed ({e})", host(url))))?;
        if !(200..300).contains(&status) {
            return Err(AiError::Status { host: host(url).to_string(), code: status, message: error_message(&text) });
        }
        serde_json::from_str(&text).map_err(|e| AiError::BadReply(format!("not JSON ({e})")))
    }
}

#[cfg(target_arch = "wasm32")]
impl Transport for HttpTransport {
    fn post_json(&self, _url: &str, _headers: &[(&str, String)], _body: &Value, _timeout_secs: u64) -> Result<Value, AiError> {
        Err(AiError::Config("the AI assistant is only available in the desktop app".into()))
    }
}

/// The provider's explanation from an error body: `{"error":{"message"}}`, the same inside a
/// list (Google), `{"message"}` or `{"detail"}`; else the start of the body. One line.
pub(crate) fn error_message(body: &str) -> String {
    let v = serde_json::from_str::<Value>(body).unwrap_or(Value::Null);
    let found =
        ["/error/message", "/0/error/message", "/message", "/detail", "/error"].iter().find_map(|p| v.pointer(p).and_then(Value::as_str).map(str::to_string));
    let text = found.unwrap_or_else(|| body.trim().to_string());
    crate::truncate(&text.split_whitespace().collect::<Vec<_>>().join(" "), 400)
}

/// The host of `url`, for messages (never the path or query, which might hold a key).
pub(crate) fn host(url: &str) -> &str {
    let rest = url.split_once("://").map_or(url, |(_, r)| r);
    rest.split(['/', '?', '#']).next().unwrap_or(rest)
}
