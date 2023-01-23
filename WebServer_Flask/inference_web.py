    
from flask import Flask, request,render_template,send_file
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from model import UNET
import albumentations as A
from albumentations.pytorch import ToTensorV2
import io
import base64
import json
import cv2
from skimage import color as skc
import time 
app = Flask(__name__)

# Load the model
#path_checkpoint='C:\\Users\\User\\Documents\\projects\\flask_inference\\my_checkpoint.pth'
def predict_image(
        image, model, device="cpu"
):
    model.eval()
    #print(model)
    image = np.array(image.convert("RGB"))
    tfms = A.Compose(
        [
            A.Resize(height=256, width=256),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    # unsqueeze provides the batch dimension
    x = tfms(image=image)["image"].to(device=device).unsqueeze(0)        # [[1], [2, [3]].squeeze(0)= [1,2,3]
    #(x)
    
    with torch.no_grad():
        preds = torch.sigmoid(model(x))
        preds = (preds > 0.5).float()
    print(preds)
    preds = (preds.squeeze().cpu().numpy().round())
    #print(preds)
    return preds
def change_frames_color(image, model, device, color):
    
    model.eval()
    image = np.array(image.convert("RGB"))
    H, W, _ = image.shape
    tfms = A.Compose(
        [
            A.Resize(height=H, width=W),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    # unsqueeze provides the batch dimension
    x = tfms(image=image)["image"].to(device=device).unsqueeze(0)  # [[1], [2], [3]].squeeze(0)= [1,2,3]

    with torch.no_grad():
        preds = torch.sigmoid(model(x))
        # preds = (preds > 0.5).float()
        preds = preds.float()
        mask = preds.squeeze().cpu().numpy().round()

    mask = cv2.resize(mask, (W, H))  # resize from 256x256  to WxH
    img_hsv = skc.rgb2hsv(image)  # transformation de l'image en hsv
    mask = np.expand_dims(mask, axis=-1)  # ajout d'une autre dimension pour calculer combine_frame sans erreur
    img_hsv[:, :, 0] = color[0]

    img_rgb = skc.hsv2rgb(img_hsv)
    combine_frame = image * (1 - mask) + img_rgb * mask * 255.  # changement de la couleur suivant le mask
    combine_frame = combine_frame.astype(np.uint8)
    return combine_frame
def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

from flask import Flask, Response


app = Flask(__name__)

@app.route('/')
def main_page():
  return render_template('main2.html')    
@app.route('/predict', methods=['POST'])
def predict():
    image_b64 = request.json['image']
    image_data = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_data))
    preds = predict_image(image, app.model,device="cpu")
    preds = (preds*225).astype(np.uint8)
    preds_img = Image.fromarray(preds)
    temp = io.BytesIO()
    preds_img.save(temp, format="PNG")
    temp.seek(0)
    
    return send_file(temp, mimetype='image/png')

@app.route('/change_color', methods=['POST'])
def change_color():
    color = request.json['color']
    print(color)
    image_b64 = request.json['image']
    image_data = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_data))
    updated_image = change_frames_color(image, app.model,device="cpu", color=color)
    img = Image.fromarray(updated_image)    
    temp = io.BytesIO()
    img.save(temp, format="PNG")
    temp.seek(0)
    return send_file(temp, mimetype='image/png')
'''
@app.route('/change_color_video', methods=['POST'])
def change_color_video():
    image_b64 = request.json['image']
    image_data = base64.b64decode(image_b64.split(',')[1].encode())
    image = Image.open(io.BytesIO(image_data))
    # process image
    updated_image = predict_image(image, app.model,device="cpu")
    img = Image.fromarray(updated_image)    
    temp = io.BytesIO()
    img.save(temp, format="PNG")
    temp.seek(0)
    return send_file(temp, mimetype='image/png')

'''


if __name__ == '__main__':
    
    with app.app_context():
        DEVICE="cpu"
        app.model = UNET(in_channels=3, out_channels=1).to(DEVICE)
        load_checkpoint(torch.load("my_checkpoint_updated.pth.tar", map_location=torch.device('cpu')), app.model)
        #state_dict = checkpoint['state_dict']
        #model.load_state_dict(state_dict)
        app.model.eval()
        # Set up the image transform
        app.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    app.run(port=5002)
