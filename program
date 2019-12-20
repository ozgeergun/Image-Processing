from tkinter import *
import tkinter.ttk as ttk
from skimage import io,filters,color, transform, img_as_ubyte, util, morphology, exposure
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import tkinter
import tkinter.filedialog
from pathlib import Path

def selectImage():
        root = Tk()
        file = tkinter.filedialog.askopenfilename(initialdir='C:/Users/%s', title="Select An Image")
        directory = os.path.split(file)[0]+"/"+os.path.split(file)[1]
        root.destroy()
        global image
        image = io.imread(directory)
        return image

i = 1    
def save():
    global i
    path = "savedImage" +str(i) + ".png"  
    myImage = Path(path)
    while(myImage.exists()):
        i += 1  
        path = "savedImage" +str(i) + ".png"  
        myImage = Path(path) 
    io.imsave(path, edited) 
    i += 1 
    print("image is saved as "+path)

def sobel_edge():  
    im = color.rgb2gray(image)
    global edited
    edited = filters.sobel(im) 
    show(edited)
       
def laplacian_filter():
    global edited
    edited = filters.laplace(image)
    show(edited)
            
def threshold_local():
    im = color.rgb2gray(image)
    global edited
    edited = filters.threshold_local(im,21,mode = "nearest")
    show(edited)
            
def prewitt_filter():
    im = color.rgb2gray(image)
    global edited
    edited = filters.prewitt(im, mask= None)
    show(edited)
            
def gaussian_filter():
    global edited
    edited = filters.gaussian(image, sigma=1, output= None, mode="reflect", cval = 0, multichannel =False,preserve_range=False, truncate=4.0)
    show(edited)
            
def median_filter():
    im = color.rgb2gray(image)
    global edited
    edited = filters.median(im,selem=np.ones((10, 5)), mask = None, shift_x = False, shift_y = False, mode= "reflect", cval=0.35, behavior='ndimage')
    show(edited)
            
def roberts_filter():
    im = color.rgb2gray(image)
    global edited
    edited = filters.roberts(im, mask = None)
    show(edited)
            
def meijering_filter():
    global edited
    edited = filters.meijering(image, sigmas=range(1, 2, 10), alpha=None, black_ridges=False)
    show(edited)
    
def unsharp_mask():
    global edited
    edited = filters.unsharp_mask(image,radius=1.5, amount=2.0, multichannel=True, preserve_range=False)
    show(edited)
            
def hessian_filter():
     global edited
     edited = filters.hessian(image,sigmas=range(1, 2, 2), scale_range=None, scale_step=None, beta1=None, beta2=None, alpha=0.5, beta=1, gamma=15, black_ridges=False)
     show(edited)
       
def show(img):
    io.imshow(img)
    io.show()

def Filters():        
    pencere=Tk()
    pencere.title("Filters")
    pencere.geometry('200x355+350+200')
    yazi=ttk.Label(pencere)
    yazi.config(text="Select one of the filters")
    yazi.pack()
       
    dugme1=ttk.Button(pencere)
    dugme1.config(text="Sobel Edge Detector")
    dugme1.config(command=sobel_edge)
    dugme1.pack()
        
    dugme2=ttk.Button(pencere)
    dugme2.config(text="Laplacian Filter")
    dugme2.config(command=laplacian_filter)
    dugme2.pack()
    
    dugme3=ttk.Button(pencere)
    dugme3.config(text="Threshold Local Filter")
    dugme3.config(command=threshold_local)
    dugme3.pack()
    
    dugme4=ttk.Button(pencere)
    dugme4.config(text="Prewitt Filter")
    dugme4.config(command=prewitt_filter)
    dugme4.pack()
    
    dugme5=ttk.Button(pencere)
    dugme5.config(text="Gaussian Filter")
    dugme5.config(command=gaussian_filter)
    dugme5.pack()
    
    dugme6=ttk.Button(pencere)
    dugme6.config(text="Median Filter")
    dugme6.config(command=median_filter)
    dugme6.pack()
    
    dugme7=ttk.Button(pencere)
    dugme7.config(text="Roberts Filter")
    dugme7.config(command=threshold_local)
    dugme7.pack()
    
    dugme8=ttk.Button(pencere)
    dugme8.config(text="Meijering Filter")
    dugme8.config(command=meijering_filter)
    dugme8.pack()
    
    dugme9=ttk.Button(pencere)
    dugme9.config(text="Unsharp Mask")
    dugme9.config(command=unsharp_mask)
    dugme9.pack()
    
    dugme10=ttk.Button(pencere)
    dugme10.config(text="Hessian Filter")
    dugme10.config(command=hessian_filter)
    dugme10.pack()
    
    pencere.mainloop()
  
def histogram_process1(image):
    plt.figure(figsize=(7, 3))        
    plt.subplot(121)
    plt.imshow(image, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.subplot(122)
        
        
def histogram():
    plt.hist(image.ravel(), bins=256, range=(0.0, 256.0), fc='k', ec='k')
    plt.title("histogram")
    plt.show()
    
def equalize_histogram():
    histogram_process1(image)
    global edited
    edited = exposure.equalize_hist(image)
    plt.imshow(edited, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    hist, hist_centers = exposure.histogram(edited)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].imshow(image, cmap=plt.cm.gray)
    axes[0].axis('off')
    axes[1].plot(hist_centers, hist, lw=2)
    axes[1].set_xlabel("equalize histogram")
    plt.show()
        
def equalize_adapthist():
    histogram_process1(image)
    global edited
    edited = exposure.equalize_adapthist(image)
    plt.imshow(edited, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    hist, hist_centers = exposure.histogram(edited)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].imshow(image, cmap=plt.cm.gray)
    axes[0].axis('off')
    axes[1].plot(hist_centers, hist, lw=2)    
    axes[1].set_xlabel("equalize_adapthist")
    plt.show()
        
def cumulative_distribution():
    bins = 256
    img_cdf, bins = exposure.cumulative_distribution(image,bins)
    plt.plot(bins, img_cdf, 'b')
    plt.title("cumulative distribution")
    plt.show()
    
def adjust_gamma():
    histogram_process1(image)
    global edited
    edited = exposure.adjust_gamma(image)
    plt.imshow(edited, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    hist, hist_centers = exposure.histogram(edited)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].imshow(image, cmap=plt.cm.gray)
    axes[0].axis('off')
    axes[1].plot(hist_centers, hist, lw=2)
    axes[1].set_xlabel("adjust gamma")
    plt.show()
        
def adjust_sigmoid():
    histogram_process1(image)
    global edited
    edited = exposure.adjust_sigmoid(image)
    plt.imshow(edited, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    hist, hist_centers = exposure.histogram(edited)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].imshow(image, cmap=plt.cm.gray)
    axes[0].axis('off')
    axes[1].plot(hist_centers, hist, lw=2)
    axes[1].set_xlabel("adjust_sigmoid")
    plt.show()
        
def adjust_log():
    histogram_process1(image)
    global edited
    edited = exposure.adjust_log(image)
    plt.imshow(edited, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    hist, hist_centers = exposure.histogram(edited)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].imshow(image, cmap=plt.cm.gray)
    axes[0].axis('off')
    axes[1].plot(hist_centers, hist, lw=2)
    axes[1].set_xlabel("adjust_log")
    plt.show()
    
def is_low_contrast():
    edited = exposure.is_low_contrast(image)
    hist, hist_centers = exposure.histogram(edited)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].imshow(image, cmap=plt.cm.gray)
    axes[0].axis('off')
    axes[1].plot(hist_centers, hist, lw=2)
    axes[1].set_xlabel("is low contrast")
    plt.show()

def match_histograms():
        def find_nearest_above(my_array, target):
            diff = my_array - target
            mask = np.ma.less_equal(diff, -1)
            if np.all(mask):
                c = np.abs(diff).argmin()
                return c 
            masked_diff = np.ma.masked_array(diff, mask)
            return masked_diff.argmin()
        def hist_match(original, specified):
         
            oldshape = original.shape
            original = original.ravel()
            specified = specified.ravel()
            s_values, bin_idx, s_counts = np.unique(original, return_inverse=True,return_counts=True)
            t_values, t_counts = np.unique(specified, return_counts=True)
            s_quantiles = np.cumsum(s_counts).astype(np.float64)
            s_quantiles /= s_quantiles[-1]    
            t_quantiles = np.cumsum(t_counts).astype(np.float64)
            t_quantiles /= t_quantiles[-1]
            sour = np.around(s_quantiles*255)
            temp = np.around(t_quantiles*255)
            b=[]
            for mydata in sour[:]:
                b.append(find_nearest_above(temp,mydata))
            b= np.array(b,dtype='uint8')
            return b[bin_idx].reshape(oldshape)
        
        def show_hist():
            original = image
            global edited
            edited = hist_match(original,specified)
            io.imshow(edited)
            io.show()
        
        root=Tk()
        file = tkinter.filedialog.askopenfilename(initialdir='C:/Users/%s', title="Select An Image")
        directory = os.path.split(file)[0]+"/"+os.path.split(file)[1]
        root.destroy()
        global specified
        specified = io.imread(directory)  
        show_hist()
        
            
def Histogram():       
    pencere=Tk()
    pencere.title("Histogram")
    pencere.geometry('200x335+350+200')
    yazi=ttk.Label(pencere)
    yazi.config(text="Select one process")
    yazi.pack()
   
    dugme1=ttk.Button(pencere)
    dugme1.config(text="Histogram")
    dugme1.config(command=histogram)
    dugme1.pack()
    
    dugme2=ttk.Button(pencere)
    dugme2.config(text="Cumulative Distribution")
    dugme2.config(command=cumulative_distribution)
    dugme2.pack()
    
    dugme3=ttk.Button(pencere)
    dugme3.config(text="Equalize Histogram")
    dugme3.config(command=equalize_histogram)
    dugme3.pack()
    
    dugme4=ttk.Button(pencere)
    dugme4.config(text="Equalize Adapthist")
    dugme4.config(command=equalize_adapthist)
    dugme4.pack()
    
    dugme5=ttk.Button(pencere)
    dugme5.config(text="Adjust Gamma")
    dugme5.config(command=adjust_gamma)
    dugme5.pack()
    
    dugme6=ttk.Button(pencere)
    dugme6.config(text="Adjust Sigmoid")
    dugme6.config(command=adjust_sigmoid)
    dugme6.pack()
    
    dugme7=ttk.Button(pencere)
    dugme7.config(text="Adjust Log")
    dugme7.config(command=adjust_log)
    dugme7.pack()
    
    dugme8=ttk.Button(pencere)
    dugme8.config(text="Is Low Contrast")
    dugme8.config(command=is_low_contrast)
    dugme8.pack()
    
    dugme9=ttk.Button(pencere)
    dugme9.config(text="Match Histograms")
    dugme9.config(command=match_histograms)
    dugme9.pack()
    
    pencere.mainloop()

def resize():
    global edited
    edited= img_as_ubyte(transform.resize(image, (1000, 300),order=1, mode='reflect', cval=0, clip=True, preserve_range=False, anti_aliasing=True, anti_aliasing_sigma=None))
    show(edited)
def rotate():
    global edited
    edited= transform.rotate(image, 180)
    show(edited)
def radon():
    img = color.rgb2gray(image)
    img = transform.rescale(img, scale=0.4, mode='reflect', multichannel=False)
    edited = transform.radon(img, circle=True)
    fig, (ax1) = plt.subplots(1, figsize=(5, 6))
    ax1.set_title("Radon transform\n")
    ax1.set_xlabel("Projection angle")
    ax1.set_ylabel("Projection position")
    ax1.imshow(edited, cmap=plt.cm.Greys_r,
               extent=(0, 180, 0, edited.shape[0]), aspect='auto')
    
    fig.tight_layout()
    plt.show()
def swirl():
    global edited
    edited = transform.swirl(image, rotation=1, strength=25 ,radius=170)
    show(edited)
def crop():
    global edited
    if (len(image.shape)==3):
        edited = util.crop(image, ((50,80),(50,40),(0,0)), copy=False)    
    else:
        edited = util.crop(image, ((50,80),(50,40)), copy=False)  
    show(edited)

def Transform():
        
    pencere=Tk()
    pencere.title("Transform")
    pencere.geometry('200x200+350+200')
    yazi=ttk.Label(pencere)
    yazi.config(text="Select a process")
    yazi.pack()
   
    dugme1=ttk.Button(pencere)
    dugme1.config(text="Resize")
    dugme1.config(command=resize)
    dugme1.pack()
    
    dugme2=ttk.Button(pencere)
    dugme2.config(text="Rotate")
    dugme2.config(command=rotate)
    dugme2.pack()
    
    dugme3=ttk.Button(pencere)
    dugme3.config(text="Radon")
    dugme3.config(command=radon)
    dugme3.pack()
    
    dugme4=ttk.Button(pencere)
    dugme4.config(text="Swirl")
    dugme4.config(command=swirl)
    dugme4.pack()
    
    dugme5=ttk.Button(pencere)
    dugme5.config(text="Crop")
    dugme5.config(command=crop)
    dugme5.pack()
    
    pencere.mainloop()
    
def Rescale():
    def RescaleIntensity(image):
            inrange = input("in range(enter a string or int between 1-255): \n")
            if(inrange.isnumeric()): #if input is not string 
                intarget = input("in range target(int between 1-255): \n")
                list_a = [int(inrange),int(intarget)]
                inrange = tuple(list_a)
            outrange = input("out range(enter a string or int between 1-255): \n")
            
            if(outrange.isnumeric()):
                outtarget = input("out range target(int between 1-255): \n")
                list_b = [int(outrange),int(outtarget)]
                outrange = tuple(list_b)
            global edited
            edited = exposure.rescale_intensity(image, inrange, outrange)
            io.imshow(edited)
            io.show()
        
    RescaleIntensity(image)

def Morphological():
    pencere=Tk()
    pencere.title("Morphological")
    pencere.geometry('200x335+350+200')
    yazi=ttk.Label(pencere)
    yazi.config(text="Select one process")
    yazi.pack()
   
    dugme1=ttk.Button(pencere)
    dugme1.config(text="Opening")
    dugme1.config(command=opening)
    dugme1.pack()
    
    dugme2=ttk.Button(pencere)
    dugme2.config(text="Closing")
    dugme2.config(command=closing)
    dugme2.pack()
    
    dugme3=ttk.Button(pencere)
    dugme3.config(text="Erosion")
    dugme3.config(command=erosion)
    dugme3.pack()
    
    dugme4=ttk.Button(pencere)
    dugme4.config(text="Dilation")
    dugme4.config(command=dilation)
    dugme4.pack()
    
    dugme5=ttk.Button(pencere)
    dugme5.config(text="Local Maxima")
    dugme5.config(command=local_maxima)
    dugme5.pack()
    
    dugme6=ttk.Button(pencere)
    dugme6.config(text="Local Minima")
    dugme6.config(command=local_minima)
    dugme6.pack()
    
    dugme7=ttk.Button(pencere)
    dugme7.config(text="Watershed")
    dugme7.config(command=watershed)
    dugme7.pack()
    
    dugme8=ttk.Button(pencere)
    dugme8.config(text="Thinning")
    dugme8.config(command=thin)
    dugme8.pack()
    
    dugme9=ttk.Button(pencere)
    dugme9.config(text="White Tophat")
    dugme9.config(command=white_tophat)
    dugme9.pack()
    
    dugme10=ttk.Button(pencere)
    dugme10.config(text="Black Tophat")
    dugme10.config(command=black_tophat)
    dugme10.pack()
    pencere.mainloop()
    
def opening():
    global edited
    edited = morphology.opening(image, selem = None, out = None)
    show(edited)
        
def closing():
    global edited
    edited = morphology.closing(image, selem = None, out = None)
    show(edited)
        
def erosion():
    global edited
    edited = morphology.erosion(image, selem=None, out=None, shift_x=True, shift_y=False)
    show(edited)
        
def dilation():
    global edited
    edited = morphology.dilation(image, selem=None, out=None, shift_x=True, shift_y=True)
    show(edited)
        
def local_maxima():
    im = color.rgb2gray(image)
    global edited
    edited = morphology.local_maxima(im, selem=None, connectivity=None, indices=False, allow_borders=True)
    show(edited)
        
def local_minima():
    im = color.rgb2gray(image)
    global edited
    edited = morphology.local_minima(im, selem=None, connectivity=None, indices=False, allow_borders=True)
    show(edited)
        
def watershed():
    global edited
    edited = morphology.watershed(image, markers=None, connectivity=1, offset=None, mask=None, compactness=0, watershed_line=True)
    show(edited)
        
def thin():
    im = color.rgb2gray(image)
    global edited
    edited = morphology.thin(im, max_iter=None)
    show(edited)
        
def white_tophat():
    global edited
    edited = morphology.white_tophat(image, selem=None, out=None)
    show(edited)
        
def black_tophat():
    global edited
    edited = morphology.black_tophat(image, selem=None, out=None)
    show(edited)

def VideoProcessing():  
        writer = cv2.VideoWriter("myvideo.avi", cv2.VideoWriter_fourcc(*"MJPG"), 30.0,(640,480))          
        cap = cv2.VideoCapture(0)
        while(True): 
            ret, frame = cap.read()                   
            frame = cv2.Canny(frame,45,50,5)  
            if ret==True:
                writer.write(frame) 
            cv2.imshow('EDGES. PRESS Q FOR QUIT',frame)
            frame=cv2.resize(frame,None,fx=0.5,fy=0.6,interpolation=cv2.INTER_AREA)
            cv2.imshow('SMALLER WINDOW. PRESS Q FOR QUIT',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
                break
        cap.release()
        cv2.destroyAllWindows()

selectImage()     
   
pencere=Tk()
pencere.title("Processes")
pencere.geometry("270x300")
yazi=ttk.Label(pencere)
yazi.config(text="Select One Of The Processes")
yazi.pack()
    
dugme2=ttk.Button(pencere)
dugme2.config(text="Filters")
dugme2.config(command=Filters)
dugme2.pack()
    
dugme3=ttk.Button(pencere)
dugme3.config(text="Histogram")
dugme3.config(command=Histogram)
dugme3.pack()
    
dugme4=ttk.Button(pencere)
dugme4.config(text="Transform")
dugme4.config(command=Transform)
dugme4.pack()
    
dugme5=ttk.Button(pencere)
dugme5.config(text="Rescale Intensity")
dugme5.config(command=Rescale)
dugme5.pack()
    
dugme6=ttk.Button(pencere)
dugme6.config(text="Morphological")
dugme6.config(command=Morphological)
dugme6.pack()
    
dugme7=ttk.Button(pencere)
dugme7.config(text="Video Processing")
dugme7.config(command=VideoProcessing)
dugme7.pack()
    
dugme8 = ttk.Button(pencere)
dugme8.config(text="Choose Another Image", command=selectImage)
dugme8.pack()

dugme9 = ttk.Button(pencere)
dugme9.config(text="Save Edited Image", command=save)
dugme9.pack()
pencere.mainloop()


    
