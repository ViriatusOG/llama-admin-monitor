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

fn static_routes() -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    let index = warp::path::end().map(|| {
        let branch = crate::update::get_current_branch();
        let badge = if branch == "beta" {
            r#"<span class="badge badge-accent" style="margin-left: 4px; padding: 0 4px; font-size: 8px; line-height: 1.2;">BETA</span>"#
        } else {
            ""
        };
        let html = static_assets::INDEX_HTML
            .replace("{{VERSION}}", env!("CARGO_PKG_VERSION"))
            .replace("{{BETA_BADGE}}", badge);
        warp::reply::html(html)
    });

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
