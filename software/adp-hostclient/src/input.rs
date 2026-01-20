use rdev::{simulate, EventType, Button, Key};
use std::thread;
use std::time::Duration;

#[derive(Clone)]
pub struct InputController;

impl InputController {
    pub fn new() -> Self {
        Self
    }

    pub fn move_mouse(&self, x: i32, y: i32) {
        // rdev uses f64 for coordinates
        let _ = simulate(&EventType::MouseMove { x: x as f64, y: y as f64 });
    }

    pub fn click_mouse(&self, button: &str) {
        let btn = match button {
            "left" => Button::Left,
            "right" => Button::Right,
            "middle" => Button::Middle,
            _ => return,
        };
        let _ = simulate(&EventType::ButtonPress(btn));
        thread::sleep(Duration::from_millis(50)); // Short delay
        let _ = simulate(&EventType::ButtonRelease(btn));
    }

    pub fn type_text(&self, text: &str) {
        for char in text.chars() {
            // This is a simplification. rdev doesn't have a direct "type string" function
            // We would need to map chars to keys.
            // For now, let's just support basic ASCII or ignore complex mapping.
            // Actually, rdev has Key::Unknown(code) but mapping char to Key is hard.
            // We can try to use `EventType::KeyPress(Key)` but we need to know which Key.
            
            // A better way for text is to use a crate that maps char to keycode, 
            // or just implement a basic mapper.
            
            // For this demo, let's just support a few keys or use a helper if available.
            // rdev doesn't have a char to key mapper built-in easily.
            
            // Let's try to map some common chars.
            let key = char_to_key(char);
            if let Some(k) = key {
                let _ = simulate(&EventType::KeyPress(k));
                let _ = simulate(&EventType::KeyRelease(k));
            }
        }
    }
    
    pub fn press_key(&self, key: &str) {
        if let Some(k) = map_key_str(key) {
            let _ = simulate(&EventType::KeyPress(k));
            let _ = simulate(&EventType::KeyRelease(k));
        }
    }
}

fn char_to_key(c: char) -> Option<Key> {
    match c {
        'a' => Some(Key::KeyA),
        'b' => Some(Key::KeyB),
        'c' => Some(Key::KeyC),
        'd' => Some(Key::KeyD),
        'e' => Some(Key::KeyE),
        'f' => Some(Key::KeyF),
        'g' => Some(Key::KeyG),
        'h' => Some(Key::KeyH),
        'i' => Some(Key::KeyI),
        'j' => Some(Key::KeyJ),
        'k' => Some(Key::KeyK),
        'l' => Some(Key::KeyL),
        'm' => Some(Key::KeyM),
        'n' => Some(Key::KeyN),
        'o' => Some(Key::KeyO),
        'p' => Some(Key::KeyP),
        'q' => Some(Key::KeyQ),
        'r' => Some(Key::KeyR),
        's' => Some(Key::KeyS),
        't' => Some(Key::KeyT),
        'u' => Some(Key::KeyU),
        'v' => Some(Key::KeyV),
        'w' => Some(Key::KeyW),
        'x' => Some(Key::KeyX),
        'y' => Some(Key::KeyY),
        'z' => Some(Key::KeyZ),
        ' ' => Some(Key::Space),
        '\n' => Some(Key::Return),
        _ => None,
    }
}

fn map_key_str(key: &str) -> Option<Key> {
    match key.to_lowercase().as_str() {
        "enter" => Some(Key::Return),
        "space" => Some(Key::Space),
        "backspace" => Some(Key::Backspace),
        "tab" => Some(Key::Tab),
        "escape" => Some(Key::Escape),
        "shift" => Some(Key::ShiftLeft),
        "control" => Some(Key::ControlLeft),
        "alt" => Some(Key::Alt),
        "meta" => Some(Key::MetaLeft),
        _ => None, 
    }
}
