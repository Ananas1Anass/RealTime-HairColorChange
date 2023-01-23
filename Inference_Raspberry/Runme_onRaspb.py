#!/usr/bin/python3

############## model = Lightest One/ model2 = Middle One/ model3 = Most accurate ##########
############## for better performances it is preconized to work with model ################

import time
from scipy import ndimage
import cv2
import numpy as np
import torch
#import torchvision
from torchvision import transforms, utils
from picamera2 import MappedArray, Picamera2, Preview
from model import UNET 
from picamera2.outputs import FfmpegOutput



def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    print('loaded!')

# Define the transform
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485,0.456,0.406],std = [0.229,0.224,0.225]),
    
])

net = UNET(in_channels=3, out_channels=1)
load_checkpoint(torch.load("my_checkpoint.pth.tar", map_location=torch.device('cpu')), net)
# Create a random tensor with the same shape as the input tensor expected by your model


def draw_masks(request):
    with MappedArray(request, "main") as m:
        for mask in masks:
            print('helloworld!')



def max_region(image):
        
        labeled_array, num_regions = ndimage.label(image)

        #find the size of each region
        region_sizes = ndimage.sum(image, labeled_array, range(num_regions + 1))

        #find the index of the largest region
        largest_region_index = region_sizes.argmax()

        #create a new array that only contains the largest region
        largest_region_array = np.where(labeled_array == largest_region_index, image, 0)      
        
        return largest_region_array 

# input
print("Please type desired color, format RGB: 0-255,0-255,0-255")
input1 = input()
color  = input1.split(",")
#user selected color is stored here
float_clr = [float(i) for i in color]




picam2 = Picamera2()
picam2.start_preview(Preview.QTGL)
config = picam2.create_preview_configuration(main={"size": (256, 256)},)
picam2.configure(config)

(w0, h0) = picam2.stream_configuration("main")["size"]

picam2.start()

count=0
start_time = time.monotonic()
# Run for 10 seconds so that we can include this example in the test suite.

#only need to create once and not in the loop
overlay = np.zeros((256, 256, 4), dtype=np.uint8)


with torch.no_grad():
        
    while True:

        buffer = picam2.capture_buffer("main")
        image = np.frombuffer(buffer, dtype=np.uint8)
        image = image.reshape((256, 256, 4))[:, :, :3]
        # Preprocess the image
        input_tensor = preprocess(image)
        # Create a mini-batch as expected by the model
        input_batch = input_tensor.unsqueeze(0)
        # Run the model
        preds = net(input_batch)
        # the preds threshold could go higher but works fine with keeping the largest mask by area 
        output = np.asarray((preds>0.89).squeeze(), dtype=np.uint8) 
        #finding largest region & applying eroding
        output = max_region(output)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 3)) # this shape bc we want to get rid of thin vertical regions detected as hair
        output = cv2.erode(output, kernel)
        #utils.save_image((preds>0.90).float(), "prediction.png")
        # the color values in the form of [R, G, B] should multiply each channel of the overlay
        overlay[:, :, 0] = output*float_clr[0]
        overlay[:, :, 1] = output*float_clr[1]
        overlay[:, :, 2] = output*float_clr[2]
        overlay[:, :, 3] = np.full(output.shape, 95) 
        # Set the overlay image
        picam2.set_overlay(overlay)
    #output = FfmpegOutput('test.mp4', audio = False)
        
        

# try to get it working  -> wheel color for the communication with the user
"""
#%%
import numpy as np
import math
import colorsys
from matplotlib import pyplot as plt


plt.rcParams["figure.figsize"] = [3, 3]
fig = plt.figure()

#in polar coordinates, generates a meshgrid with len(a)*len(b) combinations
ax = fig.add_subplot(projection='polar')

rho = np.linspace(0,1,100) # radius of 1, distance from center to outer edge
phi = np.linspace(0, math.pi*2.,1000 ) # in radians, one full circle

RHO, PHI = np.meshgrid(rho,phi) # get every combination of rho and phi

h = (PHI-PHI.min()) / (PHI.max()-PHI.min()) # use angle to determine hue, normalized from 0-1
h = np.flip(h)        
s = RHO               # saturation is set as a function of radias
v = np.ones_like(RHO) # value is constant for our case

# convert the np arrays to lists
h,s,v = h.flatten().tolist(), s.flatten().tolist(), v.flatten().tolist()
c = [colorsys.hsv_to_rgb(*x) for x in zip(h,s,v)]
c = np.array(c)

ax.scatter(PHI, RHO, c=c)
_ = ax.axis('off')
    
colors = []
final_color = [] #the final color chosen by user is there 

def onclick(event):
    if ax.contains(event)[0]:
        phi_index = (np.abs(phi - event.xdata)).argmin()
        rho_index = (np.abs(rho - event.ydata)).argmin()
        h_index = phi_index*len(rho) + rho_index
        hsv_color = np.array([h[h_index], s[h_index], v[h_index]])
        print("HSV color:", hsv_color)
        rgb_color = np.array(colorsys.hsv_to_rgb(*hsv_color)) 
        #rgb_color = rgb_color[:,None]
        print("RGB color :", rgb_color)
        
        ax.scatter(event.xdata, event.ydata, c=rgb_color.reshape((1, 3)), edgecolors='black')
        colors.append(rgb_color)
        fig.canvas.draw()

def key_event(event):
    if event.key == 'enter':
        print("RGB color selected:",colors[-1])
        fig.canvas.mpl_disconnect(cid)
        final_color.append(colors[-1])
        pltcd .close(fig=None)[source]
        return
        
cid = fig.canvas.mpl_connect('key_press_event', key_event)
cid = fig.canvas.mpl_connect('button_press_event', onclick)

ax.annotate('Click to select a color, press Enter to confirm', xy=(10, 1.05),  xycoords='data',
            xytext=(0.5, 1.08), textcoords='axes fraction',
            horizontalalignment='center', verticalalignment='top',size=8
            )

plt.show()

print(final_color)
"""
