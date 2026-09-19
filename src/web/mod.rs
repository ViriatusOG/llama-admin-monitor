pub mod api;
pub mod static_assets;
pub mod ws;

use std::sync::Arc;
use warp::Filter;

use crate::config::AppConfig;
use crate::state::AppState;

pub fn build_routes(
    state: AppState,
    app_config: Arc<AppConfig>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    let ws = ws::ws_route(state.clone());
    let api = api::api_routes(state, app_config);
    let static_files = static_routes();

    ws.or(api).or(static_files)
}

/// The dashboard HTML with the build's version and track filled in. Both
/// are compile-time constants, so this is rendered once and reused.
fn rendered_index() -> &'static str {
    static INDEX: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    INDEX.get_or_init(|| {
        let track = crate::update::current_track();
        let badge = match track {
            "beta" => r#"<span class="badge badge-accent track-badge">BETA</span>"#,
            "dev" => r#"<span class="badge badge-dim track-badge">DEV</span>"#,
            _ => "",
        };
        static_assets::INDEX_HTML
            .replace("{{VERSION}}", &crate::update::current_version())
            .replace("{{BETA_BADGE}}", badge)
    })
}

fn static_routes() -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    let index = warp::path::end().map(|| asset(rendered_index(), "text/html; charset=utf-8"));
    let tokens = warp::path("tokens.css")
        .and(warp::get())
        .map(|| asset(static_assets::TOKENS_CSS, "text/css"));
    let css = warp::path("style.css")
        .and(warp::get())
        .map(|| asset(static_assets::STYLE_CSS, "text/css"));
    let theme_js = warp::path("theme.js")
        .and(warp::get())
        .map(|| asset(static_assets::THEME_JS, "application/javascript"));
    let js = warp::path("app.js")
        .and(warp::get())
        .map(|| asset(static_assets::APP_JS, "application/javascript"));
    let manifest = warp::path("manifest.json")
        .and(warp::get())
        .map(|| asset(static_assets::MANIFEST_JSON, "application/manifest+json"));
    let sw = warp::path("sw.js")
        .and(warp::get())
        .map(|| asset(static_assets::SW_JS, "application/javascript"));
    let icon = warp::path("icon.svg")
        .and(warp::get())
        .map(|| asset(static_assets::ICON_SVG, "image/svg+xml"));

    index
        .or(tokens)
        .or(css)
        .or(theme_js)
        .or(js)
        .or(manifest)
        .or(sw)
        .or(icon)
}

/// Serves an embedded asset. `no-cache` makes browsers revalidate on every
/// load, so an in-app update is never paired with a stale script or
/// stylesheet from the previous build. The index also stamps asset URLs
/// with the build version for the same reason.
fn asset(body: &'static str, content_type: &'static str) -> impl warp::Reply {
    warp::reply::with_header(
        warp::reply::with_header(body, "content-type", content_type),
        "cache-control",
        "no-cache",
    )
}
