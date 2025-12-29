use screenshots::Screen;
use std::io::Cursor;
use image::{ImageFormat, DynamicImage};
use base64::{Engine as _, engine::general_purpose};

pub fn capture_screen() -> Result<String, String> {
    let screens = Screen::all().map_err(|e| e.to_string())?;
    
    // Try to get the first screen (primary usually)
    let screen = screens.first().ok_or("No screen found")?;
    
    let image_buffer = screen.capture().map_err(|e| e.to_string())?;
    
    // Convert to DynamicImage to use write_to
    let img = DynamicImage::ImageRgba8(image_buffer);
    
    // Convert to PNG in memory
    let mut bytes: Vec<u8> = Vec::new();
    img.write_to(&mut Cursor::new(&mut bytes), ImageFormat::Png)
        .map_err(|e| e.to_string())?;
        
    // Encode to base64
    let base64_string = general_purpose::STANDARD.encode(&bytes);
    
    Ok(base64_string)
}
