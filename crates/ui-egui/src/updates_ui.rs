//! Software updates, notice only.
//!
//! This build is a fork (it adds the AI Assistant). The official releases it would update to come
//! from the original repository and don't have the fork's changes, so FilmCraft never downloads or
//! installs them: the check at startup and Help ▸ Check for Updates only tell the user that a newer
//! official version exists, with a link to its release notes, so its changes can be brought into
//! the fork later. Nothing is installed and no prompt asks to install.

use serde_json::Value;

use crate::FilmcraftApp;

#[cfg(not(target_arch = "wasm32"))]
type CheckResult = Result<Option<craft_update::Release>, String>;

#[derive(Default)]
pub(crate) struct Runtime {
    /// The desktop app turns the check at start on; tests and the web build leave it off.
    #[cfg(not(target_arch = "wasm32"))]
    startup_check: bool,
    #[cfg(not(target_arch = "wasm32"))]
    started: bool,
    #[cfg(not(target_arch = "wasm32"))]
    check_requested: bool,
    /// The running check, and whether the user asked for it (Help ▸ Check for Updates).
    #[cfg(not(target_arch = "wasm32"))]
    pending: Option<(std::sync::mpsc::Receiver<CheckResult>, bool)>,
    /// The context of the last frame, to wake the UI when a check is queued from a menu callback.
    #[cfg(not(target_arch = "wasm32"))]
    repaint: Option<egui::Context>,
    pub(crate) manual_window_open: bool,
    pub(crate) status_message: Option<String>,
    /// The newer official release's page, when one was found.
    pub(crate) notice_url: Option<String>,
}

impl FilmcraftApp {
    /// Whether the app looks for a newer official release when it starts (off until the desktop app
    /// turns it on). It only shows a notice; nothing is installed.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn set_startup_update_check(&mut self, on: bool) {
        self.updates_rt.startup_check = on;
    }
}

pub(crate) fn check(app: &mut FilmcraftApp) -> Result<Value, String> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        app.updates_rt.check_requested = true;
        if let Some(ctx) = &app.updates_rt.repaint {
            ctx.request_repaint();
        }
        Ok(Value::Null)
    }
    #[cfg(target_arch = "wasm32")]
    {
        app.updates_rt.manual_window_open = true;
        app.updates_rt.status_message = Some("Update check is not supported in the web browser.".into());
        Ok(Value::Null)
    }
}

/// The notice for a newer official release (`None`: this version is current).
pub(crate) fn notice_text(found: Option<(&str, &str)>) -> String {
    match found {
        Some((name, version)) => tlf!(
            "A newer official version is available: {name} ({version}). It was not installed, because this FilmCraft is your fork with the AI Assistant and the official version doesn't have it. Bring its changes into the fork when you need them.",
            name = name,
            version = version
        ),
        None => tl!("FilmCraft {version} is currently the newest version available.").replace("{version}", env!("CARGO_PKG_VERSION")),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn start_check(rt: &mut Runtime, ctx: &egui::Context, manual: bool) {
    if rt.pending.is_some() {
        if manual && let Some((_, m)) = rt.pending.as_mut() {
            *m = true;
        }
        return;
    }
    let (tx, rx) = std::sync::mpsc::channel::<CheckResult>();
    let ctx = ctx.clone();
    let spawned = std::thread::Builder::new().name("update-notice".into()).spawn(move || {
        let r = std::panic::catch_unwind(|| {
            let app = &craft_update::suite::FILMCRAFT;
            craft_update::Http::new(&format!("{}/Update", app.name)).and_then(|http| craft_update::check_for_update(app, &http, env!("CARGO_PKG_VERSION")))
        })
        .unwrap_or_else(|_| Err("the update check failed unexpectedly".into()));
        let _ = tx.send(r);
        ctx.request_repaint();
    });
    if spawned.is_ok() {
        rt.pending = Some((rx, manual));
        if manual {
            rt.manual_window_open = true;
            rt.status_message = Some(tl!("Checking for updates…").to_string());
            rt.notice_url = None;
        }
    }
}

pub(crate) fn tick(app: &mut FilmcraftApp, ctx: &egui::Context) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let rt = &mut app.updates_rt;
        rt.repaint = Some(ctx.clone());
        if std::mem::take(&mut rt.check_requested) {
            start_check(rt, ctx, true);
        } else if rt.startup_check && !rt.started {
            rt.started = true;
            start_check(rt, ctx, false);
        }
        let done = rt.pending.as_ref().and_then(|(rx, manual)| rx.try_recv().ok().map(|r| (r, *manual)));
        if let Some((result, manual)) = done {
            rt.pending = None;
            match result {
                Ok(Some(release)) => {
                    let version = release.version.as_ref().map(|v| v.to_string()).unwrap_or_else(|| release.tag.clone());
                    let text = notice_text(Some((&release.name, &version)));
                    app.ui.status = tlf!("Official FilmCraft {version} is available (not installed).", version = version);
                    let rt = &mut app.updates_rt;
                    rt.status_message = Some(text);
                    rt.notice_url = Some(release.html_url).filter(|u| u.starts_with("https://"));
                    rt.manual_window_open = true;
                }
                Ok(None) if manual => {
                    rt.status_message = Some(notice_text(None));
                    rt.notice_url = None;
                    rt.manual_window_open = true;
                }
                Err(e) if manual => {
                    rt.status_message = Some(tl!("Could not check for updates:\n{error}").replace("{error}", &e));
                    rt.notice_url = None;
                    rt.manual_window_open = true;
                }
                // a quiet check at start says nothing when there's no news or no network
                Ok(None) | Err(_) => {}
            }
        }
    }
    #[cfg(target_arch = "wasm32")]
    let _ = (app, ctx);
}

pub(crate) fn show(app: &mut FilmcraftApp, ctx: &egui::Context) {
    if !app.updates_rt.manual_window_open {
        return;
    }
    let mut open = true;
    let mut visit: Option<String> = None;
    egui::Window::new(crate::i18n::t("FilmCraft Updates"))
        .open(&mut open)
        .collapsible(false)
        .resizable(false)
        .pivot(egui::Align2::CENTER_CENTER)
        .default_pos(ctx.content_rect().center())
        .default_width(420.0)
        .show(ctx, |ui| {
            if let Some(msg) = &app.updates_rt.status_message {
                ui.label(msg);
            }
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if let Some(url) = &app.updates_rt.notice_url
                    && ui.button(tl!("See what's new")).clicked()
                {
                    visit = Some(url.clone());
                }
                if ui.button(crate::i18n::t("OK")).clicked() {
                    app.updates_rt.manual_window_open = false;
                }
            });
        });
    if !open {
        app.updates_rt.manual_window_open = false;
    }
    if let Some(url) = visit {
        crate::links::open(app, ctx, &url);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_runtime_is_idle() {
        let rt = Runtime::default();
        assert!(!rt.manual_window_open);
        assert!(rt.notice_url.is_none());
        #[cfg(not(target_arch = "wasm32"))]
        {
            assert!(!rt.startup_check);
            assert!(rt.pending.is_none());
        }
    }

    #[test]
    fn a_newer_official_release_is_announced_not_installed() {
        let t = notice_text(Some(("FilmCraft 99.0.0", "99.0.0")));
        assert!(t.contains("99.0.0") && t.contains("not installed"), "{t}");
        assert!(notice_text(None).contains(env!("CARGO_PKG_VERSION")));
    }

    #[test]
    fn the_notice_window_offers_the_release_page_and_no_install() {
        let mut app = FilmcraftApp::new(filmcraft_engine::Session::default());
        app.updates_rt.manual_window_open = true;
        app.updates_rt.status_message = Some(notice_text(Some(("FilmCraft 99.0.0", "99.0.0"))));
        app.updates_rt.notice_url = Some("https://github.com/storytold/filmcraft/releases/tag/v99.0.0".into());
        let ctx = egui::Context::default();
        let mut output = ctx.run_ui(egui::RawInput::default(), |ui| show(&mut app, ui.ctx()));
        output.textures_delta.clear();
        assert!(app.updates_rt.manual_window_open, "the notice stays until OK");
    }
}
