from fastapi import FastAPI, File, UploadFile
from src.detector import SleepDetector
import cv2
import numpy as np
from io import BytesIO

app = FastAPI()

detector = SleepDetector()

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.fromstring(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        results = detector.detect(frame)
        return results
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)