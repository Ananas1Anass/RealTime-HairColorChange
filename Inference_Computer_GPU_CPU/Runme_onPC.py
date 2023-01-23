import time

import torch
from torchvision import transforms
from model import UNET
import cv2
from skimage import color as skc
import numpy as np
from wheel_color import choose_color

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_checkpoint(checkpoint, model,):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"],strict=False)
    print('loaded!')


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 256)
cap.set(cv2.CAP_PROP_FPS, 24)

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.ToTensor()
])

net = UNET(in_channels=3, out_channels=1)
load_checkpoint(torch.load("my_checkpoint_updated.pth.tar", map_location=torch.device(DEVICE)), net)

started = time.time()
last_logged = time.time()
frame_count = 0
color= [0.33333333, 1., 255.]

while True:
    with torch.no_grad():
        # read frame
        ret, image = cap.read()
        if not ret:
            raise RuntimeError("failed to read frame")

        # convert opencv output from BGR to RGB
        image = image[:, :, [2, 1, 0]]
        permuted = image

        # preprocess
        input_tensor = preprocess(image)

        # create a mini-batch as expected by the model
        input_batch = input_tensor.unsqueeze(0)
        H, W, _ = image.shape
        # run model
        output = (net(input_batch) > 0.985).float().squeeze().cpu().numpy()
        output = cv2.resize(output, (W, H))  # resize from 256x256  to WxH
        img_hsv = skc.rgb2hsv(image)  # transformation de l'image en hsv
        output = np.expand_dims(output, axis=-1)  # ajout d'une autre dimension pour calculer combine_frame sans erreur
        img_hsv[:, :, 0] = color[0]
        # assign the modified hue channel to hsv image
        img_rgb = skc.hsv2rgb(img_hsv)
        combine_frame = image * (1-output) + img_rgb * output * 255.
        combine_frame = combine_frame[:,:,[2,1,0]]
        # changement de la couleur suivant le mask
        combine_frame = combine_frame.astype(np.uint8)
        cv2.namedWindow('output', cv2.WINDOW_NORMAL)
        cv2.imshow('output', combine_frame)
        # log model performance
        frame_count += 1
        now = time.time()
        if now - last_logged > 1:
            print(f"{frame_count / (now - last_logged)} fps")
            last_logged = now
            frame_count = 0
        # Press 'q' to exit
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyWindow('Result')
            break
def change_color_frame(model,device,transform,color,frame_count):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 256)
    cap.set(cv2.CAP_PROP_FPS, 24)
    started = time.time()
    last_logged = time.time()

    while True:
        with torch.no_grad():
            # read frame
            ret, image = cap.read()
            if not ret:
                raise RuntimeError("failed to read frame")

            # convert opencv output from BGR to RGB
            image = image[:, :, [2, 1, 0]]

            # preprocess
            input_tensor = preprocess(image)

            # create a mini-batch as expected by the model
            input_batch = input_tensor.unsqueeze(0)
            H, W, _ = image.shape
            # run model
            output = (net(input_batch) > 0.985).float().squeeze().cpu().numpy()
            output = cv2.resize(output, (W, H))  # resize from 256x256  to WxH
            img_hsv = skc.rgb2hsv(image)  # transformation de l'image en hsv
            output = np.expand_dims(output, axis=-1)  # ajout d'une autre dimension pour calculer combine_frame sans erreur
            img_hsv[:, :, 0] = color[0]
            # assign the modified hue channel to hsv image
            img_rgb = skc.hsv2rgb(img_hsv)
            combine_frame = image * (1-output) + img_rgb * output * 255.
            combine_frame = combine_frame[:,:,[2,1,0]]
            # changement de la couleur suivant le mask
            combine_frame = combine_frame.astype(np.uint8)
            cv2.namedWindow('output', cv2.WINDOW_NORMAL)
            cv2.imshow('output', combine_frame)
            # log model performance
            frame_count += 1
            now = time.time()
            if now - last_logged > 1:
                print(f"{frame_count / (now - last_logged)} fps")
                last_logged = now
                frame_count = 0
            # Press 'q' to exit
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyWindow('Result')
                break
    