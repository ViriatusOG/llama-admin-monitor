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
    let index = warp::path::end().map(|| warp::reply::html(rendered_index()));

    let tokens = warp::path("tokens.css")
        .and(warp::get())
        .map(|| warp::reply::with_header(static_assets::TOKENS_CSS, "content-type", "text/css"));

    let css = warp::path("style.css")
        .and(warp::get())
        .map(|| warp::reply::with_header(static_assets::STYLE_CSS, "content-type", "text/css"));

    let theme_js = warp::path("theme.js").and(warp::get()).map(|| {
        warp::reply::with_header(
            static_assets::THEME_JS,
            "content-type",
            "application/javascript",
        )
    });

    let js = warp::path("app.js").and(warp::get()).map(|| {
        warp::reply::with_header(
            static_assets::APP_JS,
            "content-type",
            "application/javascript",
        )
    });

    let manifest = warp::path("manifest.json").and(warp::get()).map(|| {
        warp::reply::with_header(
            static_assets::MANIFEST_JSON,
            "content-type",
            "application/manifest+json",
        )
    });

    let sw = warp::path("sw.js").and(warp::get()).map(|| {
        warp::reply::with_header(
            static_assets::SW_JS,
            "content-type",
            "application/javascript",
        )
    });

    let icon = warp::path("icon.svg")
        .and(warp::get())
        .map(|| warp::reply::with_header(static_assets::ICON_SVG, "content-type", "image/svg+xml"));

    index
        .or(tokens)
        .or(css)
        .or(theme_js)
        .or(js)
        .or(manifest)
        .or(sw)
        .or(icon)
}
