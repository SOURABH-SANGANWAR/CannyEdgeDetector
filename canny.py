#imports
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import *
from skimage.feature import canny
from skimage.metrics import peak_signal_noise_ratio,structural_similarity
from skimage import data


#Convolution
def clacMul(img,x,y,filter):
  x_off = len(filter)//2
  y_off = len(filter[0])//2
  x_new = x-x_off
  y_new = y-y_off
  ans = 0
  for i in range(len(filter)):
    for j in range(len(filter[0])):
      ans += img[max(0,min(len(img)-1,x_new+i))][max(0,min(len(img[0])-1,y_new+j))]*filter[i][j]
  return ans


def g_kernel(size = 5 , sigma = 1):
        size = int(size) // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
        return g
def convolv(image,filter):
    z = np.zeros(image.shape)
    for i in range(len(image)):
        for j in range(len(image[0])):
            z[i][j] = clacMul(image,i,j,filter)
    return z

#Gaussian Blur
def blur(image):
  temp =convolv(image,g_kernel()) 
  return temp

#Sobel Edge Detector
def Sobel(image):
  x = np.array([[-1,0,1],[-2,0,2],[-1,0,1],], np.float32)
  y = np.array([[1,2,1],[0,0,0],[-1,-2,-1],],np.float32)
  edge_left = convolv(image,x)
  edge_right = convolv(image,y)
  angleDegree = np.arctan2(edge_right, edge_left) * 180. / np.pi
  angleDegree[angleDegree<0] += 180
  cand = np.hypot(edge_left,edge_right)
  cand = (cand/cand.max())*255.0
  return cand,angleDegree

#NON-MAXIMAL-SUPRESSION

def supressedVal(edgeImg,currentAngle,x,y):
  ans = 0
  if currentAngle>180:
    currentAngle -=180
  if (0 <= currentAngle < 180 / 8) or currentAngle>=7*180/8:
    if(edgeImg[x][y] >=edgeImg[x][y-1] and edgeImg[x][y] >=edgeImg[x][y+1]):
      return edgeImg[x][y]
    else:
      return 0
  if (180 / 8 <= currentAngle < 3*180 / 8):
    if(edgeImg[x][y] >=edgeImg[x+1][y-1] and edgeImg[x][y] >=edgeImg[x-1][y+1]):
      return edgeImg[x][y]
    else:
      return 0
  if (3*180 / 8 <= currentAngle < 5*180 / 8):
    if(edgeImg[x][y] >=edgeImg[x+1][y] and edgeImg[x][y] >=edgeImg[x-1][y]):
      return edgeImg[x][y]
    else:
      return 0
  else:
    if(edgeImg[x][y] >=edgeImg[x+1][y+1] and edgeImg[x][y] >=edgeImg[x-1][y-1]):
      return edgeImg[x][y]
    else:
      return 0

def non_maximal_supression(edgeImg,edgeAngle):
  output = np.zeros(edgeImg.shape)
  for i in range(1,len(edgeImg)-1):
    for j in range(1,len(edgeImg[i])-1):
        output[i][j] = (supressedVal(edgeImg,edgeAngle[i,j],i,j)) 
  print(output.shape)
  return output
  

#TRESHOLD
def apply_treshold(val,low,high,weak,strong = 255):
  if val>=high:
    return strong
  if val>low:
    return weak
  return 0

def threshold(image, low, high, weak):
    high = high*image.max()
    low = low*high;
    return np.array([[apply_treshold(image[i][j],low,high,weak) for j in range(len(image[i]))] for i in range(len(image))])

#HYST
def findHysteresis(image,iter1,iter2,weak):
  array = image.copy()
  for i in iter1:
        for j in iter2:
            if array[i, j] == weak:
                if array[i, j + 1] == 255 or array[i, j - 1] == 255 or array[i - 1, j] == 255 or array[i + 1, j] == 255 or array[i - 1, j - 1] == 255 or array[i + 1, j - 1] == 255 or array[i - 1, j + 1] == 255 or array[i + 1, j + 1] == 255:
                    array[i, j] = 255
                else:
                    array[i, j] = 0
  return array

def hysteresises(image, weak):
    upDown = findHysteresis(image,range(1, len(image)),range(1, len(image[0])),weak)
    downUp = findHysteresis(image,range(len(image)- 1, 0, -1),range(len(image[0]) - 1, 0, -1),weak)
    leftRight = findHysteresis(image,range(len(image) - 1, 0, -1),range(1, len(image[0])),weak)
    rightLeft = findHysteresis(image,range(1, len(image)),range(len(image[0]) - 1, 0, -1),weak)
    output = upDown + downUp + leftRight + rightLeft
    output[output > 255] = 255
    return np.array(output, dtype=bool)


#CANNY EDGE DETECTOR
def myCannyEdgeDetector(image,Low_Threshold = 0.07, High_Threshold = 0.11):
  blurredImage = blur(image)
  edges,directions = Sobel(blurredImage)
  edgeImg = non_maximal_supression(edges,directions)
  weak = 50
  newImage = threshold(edgeImg,Low_Threshold,High_Threshold,weak)
  plt.imshow(newImage)
  plt.show()
  output = hysteresises(newImage,weak)
  return output



image = rgb2gray(plt.imread('dataset/camera.jpg'))
myCanny = myCannyEdgeDetector(image)
builtInCanny = canny(image)
plt.imshow(myCanny,cmap = 'gray')
plt.title('My Canny Output')
plt.axis('off')
plt.show()
plt.imshow(builtInCanny,cmap = 'gray')
plt.title('Built-In Canny Output')
plt.axis('off')
plt.show()
x = peak_signal_noise_ratio(builtInCanny,myCanny)
print("PSNR Value : "+str(x))
x = structural_similarity(builtInCanny,myCanny)
print("SSIN Value : "+str(x))
