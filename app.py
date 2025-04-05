import os
from flask import Flask, render_template, request
from torchvision import models, transforms
from PIL import Image
import torch
import torch.nn as nn

# Flask app initialization
app = Flask(__name__)

# Upload folder configuration
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)  # Binary classification: cat or dog
model.load_state_dict(torch.load("cats_vs_dogs_resnet50.pth", map_location=device))
model.eval()
model.to(device)

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Image prediction function
def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = torch.max(outputs, 1)
    return "Dog" if pred.item() == 1 else "Cat"

# Home route
@app.route("/", methods=["GET", "POST"])
def index():
    label = None
    image_filename = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            image_filename = file.filename

            # 🔥 Clean the uploads folder before saving new image
            for old_file in os.listdir(app.config["UPLOAD_FOLDER"]):
                os.remove(os.path.join(app.config["UPLOAD_FOLDER"], old_file))

            # Save the uploaded image
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], image_filename)
            file.save(filepath)

            # Make prediction
            label = predict_image(filepath)
            return render_template("index.html", label=label, image=image_filename)

    return render_template("index.html", label=label, image=image_filename)

# Run the app
if __name__ == "__main__":
    app.run()
