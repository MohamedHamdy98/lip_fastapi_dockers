from flask import Flask, request, jsonify
import subprocess
import gdown
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


output_path = '/tmp/uploaded_data/outputs/output_lip_sync.mp4'

# Helper function to download a file from Google Drive
def download_from_google_drive(url, output_path):
    try:
        file_id = url.split("/d/")[1].split("/view")[0]
        download_url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(download_url, output_path, quiet=False)
        print(f"File downloaded successfully to {output_path}")
    except Exception as e:
        print(f"Failed to download file from {url}. Error: {str(e)}")


@app.route('/lip_sync', methods=['POST'])
def lip_sync():
    try:

        data = request.json
        face_url = data.get('face_url')
        audio_url = data.get('audio_url')
        pads = data.get('pads', [0, 10, 0, 0])
        output_path = '/tmp/uploaded_data/outputs/output_lip_sync.mp4'  # Default output path

        face_path = '/tmp/uploaded_data/video/face_video.mp4'
        audio_path = '/tmp/uploaded_data/audio/audio.wav'

        # Ensure directories exist
        os.makedirs(os.path.dirname(face_path), exist_ok=True)
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Download the files from Google Drive with progress bar
        print("Downloading target video...")
        download_from_google_drive(face_url, face_path)
        print("Downloading source image...")
        download_from_google_drive(audio_url, audio_path)

        if not face_path or not audio_path:
            return jsonify({'error': 'Face and audio file paths are required'}), 400
        
        # Change the working directory to Video_retalking
        os.chdir('Video_retalking')
        # Construct the command
        command = [
            'python3', 'inference.py',
            '--face', face_path,
            '--audio', audio_path,
            '--pads'] + list(map(str, pads)) + ['--outfile', output_path]
        
        # Execute the command
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return jsonify({'message': 'Processing completed successfully', 'output': output_path, 'logs': result.stdout}), 200
    
    except subprocess.CalledProcessError as e:
        return jsonify({'error': f'Error during processing: {e}', 'logs': e.stderr}), 500

@app.route('/get_path_lip_sync', methods=['GET'])
def get_path_face_swap():
    try:
        # The path to the output video from the face swap
        OUTPUT_VIDEO_PATH = '/tmp/uploaded_data/outputs/output_lip_sync.mp4'

        # Check if the output file exists
        if os.path.exists(OUTPUT_VIDEO_PATH):
            return jsonify({
                'status': 'success',
                'message': 'Output file path retrieved successfully',
                'output_path': OUTPUT_VIDEO_PATH
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Output file not found'
            })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })
