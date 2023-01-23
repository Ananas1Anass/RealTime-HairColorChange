# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 00:22:19 2023

@author: artem
"""
#%%
import numpy as np
import math
import colorsys
from matplotlib import pyplot as plt


def choose_color():
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
            
    cid = fig.canvas.mpl_connect('key_press_event', key_event)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    
    ax.annotate('Click to select a color, press Enter to confirm', xy=(10, 1.05),  xycoords='data',
                xytext=(0.5, 1.08), textcoords='axes fraction',
                horizontalalignment='center', verticalalignment='top',size=8
                )
    
    plt.show()
    return final_color
