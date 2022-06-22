import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
img=cv.imread('./train_img/thumb_111.jpg',0)
canny=cv.Canny(img,180,200)
h,w=canny.shape
def get_contour(canny):
    contour=[]
    for m in range(h):
        for n in range(w):
            if canny[m,n]==255:
                contour.append(m+1j*n)
    return np.array(contour)

def show_img(img):
    cv.imshow("out",canny)
    cv.waitKey(0)
    cv.destroyAllWindows()

def con_fft(contour_arr,amp,low=0,high=0):
    con_fft=np.fft.fft(contour_arr)
    con_fft[low:high]=0
    if amp:
        return np.abs(con_fft)
    else:
        return con_fft

def show_fft(amp, index_mode,name):
    n=amp.shape[0]
    x=np.arange(n)
    freq=np.fft.fftfreq(n)
    if index_mode:
        plt.plot(x,amp)
        plt.title("fft of contour points with index mode")
        plt.savefig('./img/{}.png'.format(name))
    else:
        plt.plot(freq,amp)
        plt.title("fft of contour points with frequency mode")
        plt.savefig('./img/{}.png'.format(name))
    plt.show()
def ifft(filtered_con,h,w,low,high):
    con_ifft=np.fft.ifft(filtered_con)
    new_img=np.ones(shape=(h,w))
    for p in con_ifft:
        x=int(np.abs(np.real(p)))
        y=int(np.abs(np.imag(p)))
        if 0<=x<h and 0<=y<w:
            new_img[x,y]=0
    plt.imshow(new_img,cmap='gray')
    plt.show()
    # cv.imwrite('./img/{low}_{high} removed.jpg'.format(low=low,high=high),new_img)
    # cv.imshow("{low}->{high} removed".format(low=low,high=high),new_img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

##########
### total 8900 points of contour index
contour=get_contour(canny)
amp=con_fft(contour,False,low=4000,high=4100)
# show_fft(amp, True, '8kto9k')
con_ifft=ifft(amp,h,w,2000,6000)
#show_fft(amp, True, "contour_filtered_freq_mode")