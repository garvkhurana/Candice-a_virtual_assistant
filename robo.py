import os
from roboflow import Roboflow

# Initialize Roboflow with your API key
rf = Roboflow(api_key="nnVaKEa8lZd2gtAkqsBo")

# Get the project reference
workspaceId = "garv-quaoe"
projectId = "custom-workflow-object-detection-a0v8p"
project = rf.workspace(workspaceId).project(projectId)

# Path to your folder containing images
folder_path = r"C:\Users\garvk\OneDrive - Bhagwan Parshuram Institute of Technology\Desktop\advance_projects\visual_assistant\dataset"

# Loop through all files in the folder and upload them
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    
    # Check if it's a valid image file
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        try:
            project.upload(
                image_path=file_path,
                batch_name="batch_01",   # You can give any batch name
                split="train",           # Options: train / valid / test
                num_retry_uploads=3
            )
            print(f"Uploaded: {filename}")
        except Exception as e:
            print(f"Failed to upload {filename}: {e}")

print("✅ All images uploaded successfully!")
