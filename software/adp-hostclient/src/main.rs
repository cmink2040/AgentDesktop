mod input;
mod screen;

use axum::{
    routing::{get, post},
    Router,
    Json,
    extract::State,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::net::TcpListener;

use crate::input::InputController;

#[derive(Clone)]
struct AppState {
    input: Arc<InputController>,
}

#[tokio::main]
async fn main() {
    // Initialize input controller
    let input_controller = Arc::new(InputController::new());
    let state = AppState { input: input_controller };

    // Build our application with a route
    let app = Router::new()
        .route("/", get(root))
        .route("/mouse/move", post(move_mouse))
        .route("/mouse/click", post(click_mouse))
        .route("/keyboard/type", post(type_text))
        .route("/keyboard/press", post(press_key))
        .route("/screen", get(get_screen))
        .with_state(state);

    // Run it
    let listener = TcpListener::bind("0.0.0.0:3000").await.unwrap();
    println!("Listening on {}", listener.local_addr().unwrap());
    axum::serve(listener, app).await.unwrap();
}

async fn root() -> &'static str {
    "ADP Host Client Running"
}

#[derive(Deserialize)]
struct MoveMousePayload {
    x: i32,
    y: i32,
}

async fn move_mouse(
    State(state): State<AppState>,
    Json(payload): Json<MoveMousePayload>,
) -> Json<String> {
    state.input.move_mouse(payload.x, payload.y);
    Json("ok".to_string())
}

#[derive(Deserialize)]
struct ClickMousePayload {
    button: String,
}

async fn click_mouse(
    State(state): State<AppState>,
    Json(payload): Json<ClickMousePayload>,
) -> Json<String> {
    state.input.click_mouse(&payload.button);
    Json("ok".to_string())
}

#[derive(Deserialize)]
struct TypeTextPayload {
    text: String,
}

async fn type_text(
    State(state): State<AppState>,
    Json(payload): Json<TypeTextPayload>,
) -> Json<String> {
    state.input.type_text(&payload.text);
    Json("ok".to_string())
}

#[derive(Deserialize)]
struct PressKeyPayload {
    key: String,
}

async fn press_key(
    State(state): State<AppState>,
    Json(payload): Json<PressKeyPayload>,
) -> Json<String> {
    state.input.press_key(&payload.key);
    Json("ok".to_string())
}

#[derive(Serialize)]
struct ScreenResponse {
    image_base64: String,
}

async fn get_screen() -> Json<ScreenResponse> {
    match screen::capture_screen() {
        Ok(base64) => Json(ScreenResponse { image_base64: base64 }),
        Err(e) => Json(ScreenResponse { image_base64: format!("Error: {}", e) }),
    }
}
