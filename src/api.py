import json
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from recognition import load_model
from decode import decode_base64_to_image_file
from inference import run_inference, convert_output_to_json

# FastAPI setup 
app = FastAPI(title="Math OCR Service")

class PredictRequest(BaseModel):
    image_base64: str

# Load model once
model = load_model()  #can pass weights_path="path/to/your.pth" if needed

@app.post("/predict")
async def predict(req: PredictRequest):
    b64 = req.image_base64
    if not b64:
        raise HTTPException(status_code=400, detail="Empty image_base64")

    try:
        # decode Base64 → temp image file
        image_path = decode_base64_to_image_file(b64)

        # run full inference pipeline
        final_results, fractions, subexp_info = run_inference(
            model, image_path, visualize=False
        )

        # convert to json string
        json_str = convert_output_to_json(final_results, fractions, subexp_info)

        # return as parsed JSON
        return json.loads(json_str)

    except Exception as e:

        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Use Uvicorn to run the app
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
