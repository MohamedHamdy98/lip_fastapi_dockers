from fastapi import FastAPI, HTTPException, Depends, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import gdown
import os
from tqdm import tqdm  
import uvicorn
from pydantic import BaseModel
from model_setup import setup_environment
from pathlib import Path
from typing import Optional

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

setup_environment()

output_path = Path('/lip_sync/outputs/output_lip_sync.mp4')

# Pydantic BaseModel for request data validation 
class LipSyncRequest(BaseModel):
    face_url: str
    audio_url: str
    pads: list[int] = [0, 10, 0, 0]  # Default pads, if not provided

# Helper function to download a file from Google Drive
def download_from_google_drive(url, output_path):
    try:
        file_id = url.split("/d/")[1].split("/view")[0]
        download_url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(download_url, output_path, quiet=False)
        print(f"File downloaded successfully to {output_path}")
    except Exception as e:
        print(f"Failed to download file from {url}. Error: {str(e)}")
        
async def form_or_json(
    face_url: Optional[str] = Form(None),
    audio_url: Optional[str] = Form(None),
    request: LipSyncRequest = Depends()
):
    # Prioritize form data if provided, otherwise use JSON
    if not face_url:
        face_url = request.face_url
    if not audio_url:
        audio_url = request.audio_url

    # Raise an error if either face_url or audio_url is missing
    if not face_url or not audio_url:
        raise HTTPException(status_code=400, detail="Both face_url and audio_url are required.")
    
    return {"face_url": face_url, "audio_url": audio_url}

@app.post('/lip_sync')
async def lip_sync(data: LipSyncRequest = Depends(form_or_json)):
    try:
        face_url = data.face_url
        audio_url = data.audio_url
        pads = data.pads
        output_path = '/lip_sync/outputs/output_lip_sync.mp4'  # Default output path

        face_path = Path('/lip_sync/data_from_user/videos/face_video.mp4')
        audio_path = Path('/lip_sync/data_from_user/audios/audio.wav')

        # Ensure directories exist
        face_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Download the files from Google Drive with progress bar
        print("Downloading target video...")
        download_from_google_drive(face_url, face_path)
        print("Downloading source audio...")
        download_from_google_drive(audio_url, audio_path)

        if not os.path.exists(face_path) or not os.path.exists(audio_path):
            raise HTTPException(status_code=400, detail='Face and audio file paths are required')

        # Change the working directory to Video_retalking
        os.chdir('Video_retalking')
        # Construct the command
        command = [
            'python3', 'inference.py',
            '--face', face_path,
            '--audio', audio_path,
            '--pads'] + list(map(str, pads)) + ['--outfile', output_path]
        
         # Execute the command with tqdm progress bar
        with tqdm(total=100, desc="Lip Sync", unit="%") as pbar:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            pbar.update(100)  # Assuming 100% progress after subprocess finishes

        # If the command was successful, return the output path
        if result.returncode == 0:
            return JSONResponse(content={'message': 'Processing completed successfully', 'output': str(output_path), 'logs': result.stdout}, status_code=200)

    except subprocess.CalledProcessError as e:
        # Handle errors during command execution
        return JSONResponse(content={'error': f'Error during processing: {str(e)}', 'logs': e.stderr}, status_code=500)
    except Exception as e:
        # Handle any other unexpected errors
        return JSONResponse(content={'error': f'An error occurred: {str(e)}'}, status_code=500)

@app.get('/get_path_lip_sync')
async def get_path_face_swap():
    try:
        # The path to the output video from the face swap
        OUTPUT_VIDEO_PATH = Path('/lip_sync/outputs/output_lip_sync.mp4')

        # Check if the output file exists
        if OUTPUT_VIDEO_PATH.exists():
            return JSONResponse(content={
                'status': 'success',
                'message': 'Output file path retrieved successfully',
                'output_path': str(OUTPUT_VIDEO_PATH)
            })
        else:
            return JSONResponse(content={
                'status': 'error',
                'message': 'Output file not found'
            })
    except Exception as e:
        return JSONResponse(content={
            'status': 'error',
            'message': str(e)
        })

@app.get('/')
async def index():
    """
    Test route to check if the API is running.
    """
    return "Lip Sync API is running!"

def main():
    # Run the FastAPI application with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5004)

if __name__ == "__main__":
    main()
