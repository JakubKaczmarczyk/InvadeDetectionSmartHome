import kagglehub
import shutil
import os

# Download latest version
# path = kagglehub.dataset_download("ministerjohn/detecting-anomalies-in-smart-home-iot-devices")
path = kagglehub.dataset_download("bobaaayoung/dataset-invade")
print("Path to dataset files:", path)

# Get the filename(s) in the downloaded path
files = os.listdir(path)

# Copy each file to current directory
for file in files:
    src_path = os.path.join(path, file)
    dst_path = os.path.join(os.getcwd(), file)
    shutil.copy2(src_path, dst_path)
    print(f"Copied {file} to current directory")