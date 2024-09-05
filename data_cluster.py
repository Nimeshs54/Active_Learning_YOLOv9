import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torchvision import transforms
from models.experimental import attempt_load  # Adjust according to your YOLOv9 implementation

# Function to load images and their labels
def load_images_and_labels(image_folder, label_folder, image_size=(640, 640)):
    images = []
    labels = []
    
    for label_file in os.listdir(label_folder):
        if label_file.endswith('.txt'):
            label_path = os.path.join(label_folder, label_file)
            image_path = os.path.join(image_folder, label_file.replace('.txt', '.png'))  # Assuming images are in .jpg format
            
            if os.path.exists(image_path):
                # Read the image and resize
                image = cv2.imread(image_path)
                image = cv2.resize(image, image_size)
                
                # Read the label
                with open(label_path, 'r') as file:
                    lines = file.readlines()
                    if lines:  # Check if the label file is not empty
                        # Extract the class label (first item) from the first line
                        label = int(lines[0].split()[0])
                        images.append(image)
                        labels.append(label)
                    else:
                        print(f"Warning: Label file {label_file} is empty.")
            else:
                print(f"Warning: Image file {image_path} does not exist.")
    
    return np.array(images), np.array(labels)

# Function to extract features using the YOLOv9 model
def extract_features(images, model, device):
    model.eval()
    features = []
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize if required
    ])
    
    with torch.no_grad():
        for i, image in enumerate(images):
            image_tensor = transform(image).unsqueeze(0).to(device)
            outputs = model(image_tensor)  # The model returns a tuple
            
            # Inspect the outputs
            print(f"Output: {outputs}")  # Print the full output
            print(f"Output type: {type(outputs)}, Output length: {len(outputs)}")
            
            # Extract features from the appropriate part of the output
            if isinstance(outputs, tuple) and len(outputs) > 0:
                feature_list = outputs[0]  # Assuming the first element contains the feature maps
                if isinstance(feature_list, list) and len(feature_list) > 0:
                    for feature in feature_list:
                        if isinstance(feature, torch.Tensor):
                            feature = feature.flatten(start_dim=1).cpu().numpy()  # Flatten while preserving the feature dimension
                            features.append(feature)
                        else:
                            print(f"Warning: Unexpected feature format in the list. Type: {type(feature)}")
                else:
                    print(f"Warning: Unexpected feature format in the tuple. Type: {type(feature_list)}")
            else:
                print(f"Warning: Unexpected model output format. Type: {type(outputs)}")
    
    if len(features) == 0:
        print("No features extracted. Ensure the model is properly configured.")
        return np.array([])  # Return an empty array if no features were extracted
    
    return np.vstack(features)  # Combine all features into a single array

# Load images and labels
image_folder = 'dataset/test_demo/images'  # Replace with your image folder path
label_folder = 'dataset/test_demo/labels'  # Replace with your label folder path
images, labels = load_images_and_labels(image_folder, label_folder)

# Ensure images and labels are correctly paired
if len(images) != len(labels):
    print(f"Error: The number of images ({len(images)}) and labels ({len(labels)}) do not match.")
else:
    # Load the YOLOv9 model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the YOLOv9 model
    model = attempt_load('best.pt', device=device)  # Replace with your model path

    # Extract features
    features = extract_features(images, model, device)

    # Debugging: Check the shape of features array and labels
    print(f"Features shape: {features.shape}")
    print(f"Labels length: {len(labels)}")

    if features.shape[0] == 0:
        print("No features extracted. Ensure the model is properly configured.")
    else:
        # Check the consistency of feature and label counts
        if features.shape[0] != len(labels):
            print(f"Error: The number of features ({features.shape[0]}) does not match the number of labels ({len(labels)}).")
        else:
            # Reduce dimensions to 2D
            tsne = TSNE(n_components=2, random_state=42)
            reduced_features = tsne.fit_transform(features)
            
            # Debugging: Check the length of reduced features
            print(f"Reduced features shape: {reduced_features.shape}")

            # Plot the reduced features
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', marker='o')
            plt.colorbar(scatter, ticks=range(8), label='Classes')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.title('2D visualization of image clusters')
            plt.show()
