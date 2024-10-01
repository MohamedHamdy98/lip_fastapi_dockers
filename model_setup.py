import subprocess
import os
import sys
from pathlib import Path
from tqdm import tqdm
import requests

def is_package_installed(package_name):
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def install_package(package_name):
    subprocess.run([sys.executable, "-m", "pip", "install", package_name], check=True)

def download_file(url, destination):
    """Download a file with progress reporting."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kilobyte
    t = tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"Downloading {os.path.basename(destination)}")
    with open(destination, 'wb') as f:
        for data in response.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()
    if total_size != 0 and t.n != total_size:
        print(f"ERROR: Download of {os.path.basename(destination)} failed.")

def check_model_exists(model_path):
    """Check if a model file exists."""
    if Path(model_path).exists():
        print(f"{model_path} already exists. Skipping download.")
        return True
    else:
        return False
    
def setup_environment():
    # Check if required packages are installed, and install if necessary
    packages = ["flask", "gdown"]
    for package in packages:
        if not is_package_installed(package):
            print(f"{package} not found. Installing...")
            install_package(package)
        else:
            print(f"{package} is already installed.")

    # Clone the repository
    repo_url = "https://github.com/vinthony/video-retalking.git"
    repo_dir = Path("video-retalking")
    if not repo_dir.exists():
        print(f"Cloning the repository from {repo_url}...")
        subprocess.run(["git", "clone", repo_url], check=True)
    os.chdir(repo_dir)  # Change to the video-retalking directory

    # Install necessary packages
    subprocess.run([sys.executable, "-m", "pip", "install", "flask", "gdown"], check=True)
    # subprocess.run(["apt-get", "update"], check=True)
    # subprocess.run(["apt", "install", "ffmpeg", "-y"], check=True)

    # Install the dependencies for video-retalking
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)

    # Create checkpoints directory
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Download pre-trained models
    urls = [
        ("https://github.com/vinthony/video-retalking/releases/download/v0.0.1/30_net_gen.pth", checkpoints_dir / "30_net_gen.pth"),
        ("https://github.com/vinthony/video-retalking/releases/download/v0.0.1/BFM.zip", checkpoints_dir / "BFM.zip"),
        ("https://github.com/vinthony/video-retalking/releases/download/v0.0.1/DNet.pt", checkpoints_dir / "DNet.pt"),
        ("https://github.com/vinthony/video-retalking/releases/download/v0.0.1/ENet.pth", checkpoints_dir / "ENet.pth"),
        ("https://github.com/vinthony/video-retalking/releases/download/v0.0.1/expression.mat", checkpoints_dir / "expression.mat"),
        ("https://github.com/vinthony/video-retalking/releases/download/v0.0.1/face3d_pretrain_epoch_20.pth", checkpoints_dir / "face3d_pretrain_epoch_20.pth"),
        ("https://github.com/vinthony/video-retalking/releases/download/v0.0.1/GFPGANv1.3.pth", checkpoints_dir / "GFPGANv1.3.pth"),
        ("https://carimage-1253226081.cos.ap-beijing.myqcloud.com/gpen/GPEN-BFR-1024.pth", checkpoints_dir / "GPEN-BFR-1024.pth"),
        ("https://github.com/vinthony/video-retalking/releases/download/v0.0.1/GPEN-BFR-512.pth", checkpoints_dir / "GPEN-BFR-512.pth"),
        ("https://github.com/vinthony/video-retalking/releases/download/v0.0.1/LNet.pth", checkpoints_dir / "LNet.pth"),
        ("https://github.com/vinthony/video-retalking/releases/download/v0.0.1/ParseNet-latest.pth", checkpoints_dir / "ParseNet-latest.pth"),
        ("https://github.com/vinthony/video-retalking/releases/download/v0.0.1/RetinaFace-R50.pth", checkpoints_dir / "RetinaFace-R50.pth"),
        ("https://github.com/vinthony/video-retalking/releases/download/v0.0.1/shape_predictor_68_face_landmarks.dat", checkpoints_dir / "shape_predictor_68_face_landmarks.dat"),
    ]

    # Download models if they don't exist
    for url, output in urls:
        if not check_model_exists(output):
            print(f"Downloading {os.path.basename(output)}...")
            download_file(url, output)

    # Unzip BFM.zip
    subprocess.run(["unzip", "-d", str(checkpoints_dir / "BFM"), str(checkpoints_dir / "BFM.zip")], check=True)

def main():
    """Main function to run the setup process."""
    print("Starting the environment setup...")
    try:
        setup_environment()
        print("Setup completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during setup: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
