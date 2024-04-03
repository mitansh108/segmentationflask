from flask import Flask, render_template, request
from PIL import Image
import torch
import torchvision.transforms as transforms
from segmentation_models_pytorch import Unet
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Check if CUDA is available and set device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the segmentation model
model = Unet('resnet50', encoder_weights=None, classes=1).to(device)
model.load_state_dict(torch.load('/Users/mitanshpatel/Downloads/micronet_resnet50_steel_dataset.pth', map_location=device))
model.eval()  # Set model to evaluation mode

# Define preprocessing transformation
def preprocess_image(image):
    # Convert image to RGB (if RGBA)
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    # Apply transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize image to expected input size
        transforms.ToTensor(),           # Convert PIL image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet statistics
    ])

    # Apply transformations and add batch dimension
    image_tensor = transform(image).unsqueeze(0)

    return image_tensor

# Define function for segmentation
def segment_image(image, model):
    # Preprocess the image
    image_tensor = preprocess_image(image).to(device)

    # Perform prediction
    with torch.no_grad():
        output = model(image_tensor)

    # Convert output to numpy array
    segmented_image = output.squeeze().detach().cpu().numpy()

    return segmented_image

# Function to apply custom colormap 'PuRd_r'
def apply_PuRd_r_colormap(segmented_image):
    # Apply PuRd_r colormap to the segmented image
    cmap = plt.get_cmap('PuRd_r')
    colored_image = cmap(segmented_image)[:, :, :3]  # Ignore alpha channel if present
    colored_image *= 255  # Scale colormap values to 0-255 range
    colored_image = colored_image.astype(np.uint8)
    return colored_image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/segment', methods=['POST'])
@app.route('/segment', methods=['POST'])
def segment():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        # Read image file
        img = Image.open(io.BytesIO(uploaded_file.read()))
        # Perform segmentation
        segmented_image = segment_image(img, model)  # Assuming you have defined 'segment_image' function
        # Apply PuRd_r colormap to segmented image
        colored_image = apply_PuRd_r_colormap(segmented_image)  # Assuming you have defined 'apply_PuRd_r_colormap' function
        # Convert images to base64 format for display
        original_img_io = io.BytesIO()
        img.save(original_img_io, 'PNG')
        original_img_io.seek(0)
        original_img_base64 = base64.b64encode(original_img_io.getvalue()).decode()

        segmented_img_io = io.BytesIO()
        Image.fromarray(colored_image).save(segmented_img_io, 'PNG')
        segmented_img_io.seek(0)
        segmented_img_base64 = base64.b64encode(segmented_img_io.getvalue()).decode()

        return render_template('result.html', original_image=f'data:image/png;base64,{original_img_base64}', segmented_image=f'data:image/png;base64,{segmented_img_base64}')
    else:
        return 'No file uploaded'

if __name__ == '__main__':
    app.run(debug=True)





