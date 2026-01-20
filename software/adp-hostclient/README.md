# ADP Host Client

This is a Rust-based mouse/keyboard emulator and screen recorder with an HTTP API.

## Prerequisites

- Rust (cargo)
- macOS: Accessibility permissions are required for input simulation and screen recording.

## Running

```bash
cargo run
```

The server listens on `0.0.0.0:3000`.

## API Endpoints

### Mouse

- **Move Mouse**
  - `POST /mouse/move`
  - Body: `{"x": 100, "y": 200}`

- **Click Mouse**
  - `POST /mouse/click`
  - Body: `{"button": "left"}` (or "right", "middle")

### Keyboard

- **Type Text**
  - `POST /keyboard/type`
  - Body: `{"text": "Hello World"}`

- **Press Key**
  - `POST /keyboard/press`
  - Body: `{"key": "enter"}` (or "space", "backspace", etc.)

### Screen

- **Capture Screen**
  - `GET /screen`
  - Returns: `{"image_base64": "..."}` (PNG format)
