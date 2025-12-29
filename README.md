# VDesktop
VDesktop is a virtualized, proof-of-concept application designed for pure GUI raster-based AI interaction. This playground is made to avoid OS display automation complexity and breaking of personal devices.

![image](https://github.com/user-attachments/assets/9a809202-6267-4678-af13-3d1e8583a6dc)

While VDesktop is extremely simple, it will progressively get more and more complex till it represents a real application, in which case a real world OS implementation may be made.

## Usage
To run, setup Gemini API keys with ```echo "API_KEY=yourAPIKEY" > .env```. Then install the reqs.txt. 

Run ```python -m adt.main```.

## Design & Implementation
![image](https://github.com/user-attachments/assets/774616ef-8a03-4136-8a48-da197bcdf71b)

VDesktop utilizes tk to render a graphics window. There are various buttons of different colors. The goal is to make a input field so that a user can click very basic buttons simply by telling a language model with text and image support to do it. 

### Location Awareness
![image](https://github.com/user-attachments/assets/98a3a8c9-9d8a-4f38-aeb8-da1d1e279551)

A grid is placed on top of a screenshot to help the language model evaluate how far objects are from each other. Each square represents a X-by-X pixel location. The model uses this to understand how far it should move the cursors.
