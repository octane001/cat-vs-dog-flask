🐶 Cat vs 🐱 Dog Classifier Web App
A simple deep learning-based web application that classifies images as either Cat or Dog using a pretrained ResNet50 model, built with Flask.

🔍 How It Works
Upload an image via the web interface.

The image is preprocessed and passed through a fine-tuned ResNet50 model.

The model returns a prediction: Cat or Dog.

Result is displayed instantly along with the uploaded image.

🚀 Tech Stack
PyTorch – for model and inference

Torchvision – for ResNet50 and image transforms

Flask – lightweight Python web framework

HTML – minimal frontend

Gunicorn – production WSGI server (for deployment)
