from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import torch
from torchvision import transforms
from PIL import Image, ImageChops, ImageEnhance
import io
import os
from model_arch import TamperingDetectionCNN
from logger import setup_logger
logger = setup_logger(name="main",log_file="logs/main.log")
# ELA Conversion 
def convert_to_ela_image(image: Image.Image, quality=90):
    """
    Convert a PIL image to its Error Level Analysis (ELA) representation.
    Saves the image temporarily to JPEG format at the given quality, then
    computes the difference between the original and re-saved image.
    """
    temp_filename = "temp_ela.jpg"
    image.save(temp_filename, 'JPEG', quality=quality)
    resaved = Image.open(temp_filename)
    ela_image = ImageChops.difference(image, resaved)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    os.remove(temp_filename)
    return ela_image


app = FastAPI()

# Mount static files and configure templates (if you use a frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# setting up the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TamperingDetectionCNN().to(device)
model_path = os.path.join("models", "tampering_detection.pth")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found in '{model_path}'.")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  

# Define the transformation pipeline (must match training preprocessing)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

#  Routes
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        logger.error("Error occurred because of incorrect file type.")
        raise HTTPException(status_code=400, detail="Only .png, .jpg, and .jpeg files are supported.")

    # Read image bytes
    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        logger.debug("Image loaded")
    except Exception:
        logger.error("Unexpected error occcurred while processing the image.")
        raise HTTPException(status_code=400, detail="Could not process the image.")
    
    ela_image = convert_to_ela_image(image, quality=90)
    logger.debug("Converted the image to ELA")
    # Apply transformation (resize, to tensor, etc.)
    input_tensor = transform(ela_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
    prediction_idx = torch.argmax(outputs, dim=1).item()
    
    # Map prediction index to class name
    class_names = {0: "Original", 1: "Tampered"}
    prediction = class_names.get(prediction_idx, "Unknown")
    logger.debug("Prediction: ", prediction)
    return JSONResponse({"filename": file.filename, "prediction": prediction})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
