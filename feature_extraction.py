import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os

model = models.googlenet(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model(image)
    return features

# Extract features for a video
video_folder = "data/video_frames"
features = []
for frame in sorted(os.listdir(video_folder)):
    feature = extract_features(os.path.join(video_folder, frame))
    features.append(feature)
