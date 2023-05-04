import math

import numpy as np
import cv2
import matplotlib.pyplot as plot
from matplotlib.colors import hsv_to_rgb


def gaussderiv(img, sigma):
    x = np.array(list(range(math.floor(-3.0 * sigma + 0.5), math.floor(3.0 * sigma + 0.5) + 1)))
    G = np.exp(-x**2 / (2 * sigma**2))
    G = G / np.sum(G)
    
    D = -2 * (x * np.exp(-x**2 / (2 * sigma**2))) / (np.sqrt(2 * math.pi) * sigma**3)
    D = D / (np.sum(np.abs(D)) / 2)
    
    Dx = cv2.sepFilter2D(img, -1, D, G)
    Dy = cv2.sepFilter2D(img, -1, G, D)

    return Dx, Dy

def gausssmooth(img, sigma):
    x = np.array(list(range(math.floor(-3.0 * sigma + 0.5), math.floor(3.0 * sigma + 0.5) + 1)))
    G = np.exp(-x**2 / (2 * sigma**2))
    G = G / np.sum(G)
    return cv2.sepFilter2D(img, -1, G, G)
    
def show_flow(U, V, ax, type='field', set_aspect=False):
    if type == 'field':
        scaling = 0.1
        u = cv2.resize(gausssmooth(U, 1.5), (0, 0), fx=scaling, fy=scaling)
        v = cv2.resize(gausssmooth(V, 1.5), (0, 0), fx=scaling, fy=scaling)
        
        x_ = (np.array(list(range(1, u.shape[1] + 1))) - 0.5) / scaling
        y_ = -(np.array(list(range(1, u.shape[0] + 1))) - 0.5) / scaling
        x, y = np.meshgrid(x_, y_)
        
        ax.quiver(x, y, -u * 5, v * 5)
        if set_aspect:
            ax.set_aspect(1.)
    elif type == 'magnitude':
        magnitude = np.sqrt(U**2 + V**2)
        ax.imshow(np.minimum(1, magnitude))
    elif type == 'angle':
        angle = np.arctan2(V, U) + math.pi
        im_hsv = np.concatenate((np.expand_dims(angle / (2 * math.pi), -1),
                                np.expand_dims(np.ones(angle.shape, dtype=np.float32), -1),
                                np.expand_dims(np.ones(angle.shape, dtype=np.float32), -1)), axis=-1)
        ax.imshow(hsv_to_rgb(im_hsv))
    elif type == 'angle_magnitude':
        magnitude = np.sqrt(U**2 + V**2)
        angle = np.arctan2(V, U) + math.pi
        im_hsv = np.concatenate((np.expand_dims(angle / (2 * math.pi), -1),
                                np.expand_dims(np.minimum(1, magnitude), -1),
                                np.expand_dims(np.ones(angle.shape, dtype=np.float32), -1)), axis=-1)
        ax.imshow(hsv_to_rgb(im_hsv))
    else:
        print('Error: unknown optical flow visualization type.')
        exit(-1)

def rotate_image(img, angle):
    center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return rotated

#plot 2 images side by side
def plot_images(img1, img2, cmap=None):
    plot.subplot(1,2,1)
    plot.imshow(img1, cmap=cmap)
    plot.subplot(1,2,2)
    plot.imshow(img2, cmap=cmap)
    plot.show()

def calc_derivatives (im1 , im2): 
    #im1 - first image matrix (grayscale)
    #im2 - second image matrix (grayscale)

    #TODO if needed implement custom sigma values smoothing and derivates

    #from slides opticalFlow1: temporal derivative is aproximated by the difference between the 2 images 
    #slides 48 improvments:  apply a small Gaussian to improve the results
    it = gausssmooth((im2 - im1), sigma=1)

    #from slides opticalFlow1: spatial derivatives are aproximated by convolution 

    #ix, iy = gaussderiv(im1, 0.4)

    #slides 48 improvments: avrage spatila derivatev in frame t and t+1 !mathematically incorect but could help!

    #bigger sigma -> smoother edges but less details.
    #sigma 0.6 from experiments
    #ix, iy = gaussderiv(np.divide(im1+im2,2) , sigma=0.6) 
 
    #applying an additional gaussian smoothing to the derivatives to reduce noise
    ix, iy = gaussderiv(gausssmooth( np.divide(im1+im2,2), sigma=0.6) , sigma=1)

    #ix1, iy1 = gaussderiv(im1, sigma=0.6)
   #ix2, iy2 = gaussderiv(im2, sigma=0.6)
    #ix = (ix1 + ix2) / 2
    #iy = (iy1 + iy2) / 2
    

    return ix, iy, it

def plot_flow(u_vector, v_vector, img1, img2, kateri, filename='test', save=False ):
    #u_vector - x component of the flow vector
    #v_vector - y component of the flow vector
    #img1 - first image matrix (grayscale)
    #img2 - second image matrix (grayscale)
    #kateri - lucaskanade or hornschunck
    #filename - save the file as this name 
    #save - save file or dont save the file

    # plot a 2x2 plot where the top row displays img1 and img2 bottom left show_flow field, bottom right is show_flow angle
    fig, ax = plot.subplots(2, 2)

    ax[0][0].imshow(img1)
    ax[0][0].set_title("im(t)")
    ax[0][1].imshow(img2)
    ax[0][1].set_title("im(t+1)")

    show_flow(u_vector, v_vector, ax[1][0], type='field', set_aspect=True)
    show_flow(u_vector, v_vector, ax[1][1], type='angle', set_aspect=False)

    fig.tight_layout()
    dodatek = ''
    #kako poimenovati plot
    if kateri == 'lk':
        fig.suptitle ('Lucas-Kanade optical flow')
        dodatek = 'lk'
    elif kateri == 'hs':
        fig.suptitle ('Horn-Schunck optical flow')
        dodatek = 'hs'
    else:
        fig.suptitle ('Figure 1')
        dodatek = 'unknown'

    if save: 
        fig.savefig(f'./plots/{filename}-{dodatek}.png', bbox_inches='tight')
        
    plot.show()

def cmb_plot_show(img1, img2, u_lk, v_lk, u_hs, v_hs, filename):
    fig, ax = plot.subplots(3, 2)
    fig.suptitle ('Optical flow results')

    ax[0][0].imshow(img1)
    ax[0][0].set_title("im(t)")
    ax[0][1].imshow(img2)
    ax[0][1].set_title("im(t+1)")

    show_flow(u_lk, v_lk, ax[1][0], type='field', set_aspect=True)
    ax[1][0].set_title("Lucas-Kanade flow field")
    show_flow(u_lk, v_lk, ax[1][1], type='angle', set_aspect=False)
    ax[1][1].set_title("Lucas-Kanade flow angle")


    show_flow(u_hs, v_hs, ax[2][0], type='field', set_aspect=True)
    ax[2][0].set_title("Horn-Schunck flow field")
    show_flow(u_hs, v_hs, ax[2][1], type='angle', set_aspect=False)
    ax[2][1].set_title("Horn-Schunck flow angle")
    
    fig.tight_layout()

    fig.savefig(f'./plots/{filename}-oba.png')

    plot.show()

def cmb_plot_show_dense(img1, img2, u_lk, v_lk, u_hs, v_hs, filename):
    fig, ax = plot.subplots(2, 2)
    fig.suptitle ('Optical flow results')

    ax[0][0].imshow(img1)
    ax[0][0].set_title("im(t)")
    ax[0][1].imshow(img2)
    ax[0][1].set_title("im(t+1)")

    show_flow(u_lk, v_lk, ax[1][0], type='field', set_aspect=True)
    ax[1][0].set_title("Lucas-Kanade flow field")
    show_flow(u_hs, v_hs, ax[1][1], type='field', set_aspect=True)
    ax[1][1].set_title("horn-Schunck flow field") 

    fig.tight_layout()

    fig.savefig(f'./plots/{filename}-oba-dense.png')

    plot.show()

