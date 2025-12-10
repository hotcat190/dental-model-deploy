import io
import tensorflow as tf
import numpy as np
import trimesh
import uvicorn
from skimage import measure
from fastapi import FastAPI, UploadFile, File, Response, HTTPException
from typing import List
from contextlib import asynccontextmanager
from model import build_model

# Global variable to hold the model
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing TensorFlow...")
    pretrain_weight = "model/shapenet_pretrained.weights.h5"
    finetuned_weight = "model/teethnet_finetuned.weights.h5"
    
    # Load the model here so it stays in RAM
    strategy = tf.distribute.get_strategy()
    with strategy.scope():
        ml_models["dental_model"] = build_model(4, 128, False)
    ml_models["dental_model"].load_weights(pretrain_weight)
    
    print("Model loaded successfully. Ready for inference.")
    
    yield  # <--- The application runs here while this yield is active
    
    print("Shutting down...")
    ml_models.clear()

# Initialize App
app = FastAPI(lifespan=lifespan)

@app.get("/")
def health_check():
    return {"status": "running", "model_loaded": model is not None}

@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    global model
    if len(files) != 4:
        raise HTTPException(status_code=400, detail="Endpoint requires exactly 4 PNG images.")
    
    if "dental_model" not in ml_models:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    model = ml_models["dental_model"]
    processed_images = []

    for file in files:
        content = await file.read()
        img = tf.io.decode_png(content, channels=3)
        img = tf.image.resize(img, [224, 224])        
        processed_images.append(img)

    # stack -> (4, 224, 224, 3)
    input_tensor = tf.stack(processed_images) 
    # expand_dims -> (1, 4, 224, 224, 3)
    input_tensor = tf.expand_dims(input_tensor, axis=0) 

    # Output Shape: (1, 128, 128, 128, 1)
    preds = model(input_tensor, training=False)

    # 5. Post-Processing
    # Sigmoid to get probabilities [0, 1]
    preds = tf.math.sigmoid(preds)
    
    # Remove batch and channel dims -> (128, 128, 128)
    vol_data = preds.numpy()[0, :, :, :, 0]

    # Threshold to binary (as requested)
    binary_vol = vol_data > 0.5

    # 6. Generate STL (Marching Cubes)
    try:
        # Use marching cubes to find the surface where value is 0.5
        # step_size=1 preserves full resolution. Increase to 2 to lower file size/quality.
        verts, faces, normals, values = measure.marching_cubes(binary_vol, level=0.5)
        
        # Create a Mesh object
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        
        # Write to in-memory buffer
        file_stream = io.BytesIO()
        mesh.export(file_stream, file_type='stl')
        file_stream.seek(0) # Reset pointer to start of file
        
        filename = "dental_model.stl"
        
    except ValueError:
        # This happens if the model predicts empty space (no teeth found)
        raise HTTPException(status_code=400, detail="No 3D object detected in inference.")

    # 7. Return the file directly
    return Response(
        content=file_stream.getvalue(),
        media_type="application/octet-stream", # Standard for binary downloads
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

if __name__ == "__main__":
    # Cloud Run expects the app to listen on the PORT env var
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)