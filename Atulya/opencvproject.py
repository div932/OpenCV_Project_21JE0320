import cv2
from cv2 import BORDER_CONSTANT
import cv2.aruco as aruco
import numpy as np
import math

def arucoid(img):                                               #this function finds id of arucomarker in image
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    key = getattr(aruco,f"DICT_5X5_250")
    arucodict = aruco.Dictionary_get(key)
    p = aruco.DetectorParameters_create()
    (c,i,r)= cv2.aruco.detectMarkers(img,arucodict,parameters=p)
    return (c,i,r)

def arucocords(img):                                            #this function finds coordinates of corner of arucomarker
    (c,i,r)=arucoid(img)
    if len(c)>0:
        i=i.flatten()
        for (markercorner,markerid) in zip(c,i):
            corner = markercorner.reshape((4,2))
            (topleft,topright,bottomright,bottomleft)=corner
            topleft = (int(topleft[0]),int(topleft[1]))
            topright = (int(topright[0]),int(topright[1]))
            bottomleft = (int(bottomleft[0]),int(bottomleft[1]))
            bottomright = (int(bottomright[0]),int(bottomright[1]))
        return topleft,topright,bottomright,bottomleft

def arucoangle(img):                                            #this function finds angle of arucomarker wrt x axis
    topleft,topright,bottomright,bottomleft=arucocords(img)
    cx = int((topleft[0]+bottomright[0])/2)
    cy = int((topleft[1]+bottomright[1])/2)
    px = int((topright[0]+bottomright[0])/2)
    py = int((topright[1]+bottomright[1])/2)
    m=(py-cy)/(px-cx)
    theta = math.atan(m)
    center = (cx,cy)
    return center,(theta*180)/math.pi

def rotate_image(image, angle,center):                          #this function rotates arucomarker
    rot_mat = cv2.getRotationMatrix2D(center, angle,0.8)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR,borderMode=BORDER_CONSTANT,borderValue=(255,255,255))
    return result

def maxcord(cord):                                              #to find extreme coordinates of image
    xmax=cord[0][0]
    xmin=cord[0][0]
    ymax=cord[0][1]
    ymin=cord[0][1]
    for i in cord:
        if i[0]>xmax:
            xmax = i[0]
        if i[0]<xmin:
            xmin = i[0]
        if i[1]>ymax:
            ymax = i[1]
        if i[1]<ymin:
            ymin = i[1]
    return xmax,xmin,ymax,ymin

def crp(img):                                                   #to select a portion of image
    topleft,topright,bottomright,bottomleft=arucocords(img)
    l=[topleft,topright,bottomright,bottomleft]
    xmax,xmin,ymax,ymin = maxcord(l)
    t = img[ymin:ymax,xmin:xmax]
    return t

L=['Ha.jpg','HaHa.jpg','LMAO.jpg','XD.jpg']                     #list of all present arucomarkers
iddict = {}                                                     #dict to store all arucomarker with their specific ids


for i in L:
    x = cv2.imread(i)
    (c,ids,r)=arucoid(x)
    iddict[i]=ids

img = cv2.imread('CVtask.jpg')                                  #loading actual image
r = cv2.imread('white.png')
imnew=cv2.resize(r,(img.shape[1],img.shape[0]))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_,thresh = cv2.threshold(gray,230,255,cv2.THRESH_BINARY)
color={'green':[79,209,146],'orange':[9,127,240],'white':[210,222,228],'black':[0,0,0]}  #dict of colour according to their pixels
cid = {'green':1,'orange':2,'white':4,'black':3}                                         #dict of colour with their specific id
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
g=[]

for c in contours:                                              #to paste arucomarkers on square box in new image
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.01 * peri, True)
    if len(approx) == 4:
        x,y,w,h=cv2.boundingRect(approx)
        aspectratio = float(w)/h
        if aspectratio >=0.95 and aspectratio<=1.05:
            cord = [o[0].tolist() for o in approx]
            xmax,xmin,ymax,ymin = maxcord(cord)
            subimg = img[ymin:ymax,xmin:xmax]
            shape1 = subimg.shape
            sx=shape1[0]
            sy = shape1[1]
            newshape = (sy-30,sx-30)
            m1=(int((cord[0][0]+cord[1][0])/2),int((cord[0][1]+cord[1][1])/2))
            c=(int((cord[0][0]+cord[2][0])/2),int((cord[0][1]+cord[2][1])/2))
            if (m1[0]-c[0]) != 0:
                theta = math.atan((m1[1]-c[1])/(m1[0]-c[0]))
            else :
                theta = math.pi/(-2)
            for i in color.keys():
                d = np.array(color[i])
                d.reshape((3,))
                if (d==img[c[1],c[0],:]).any():
                    wer = np.array(cid[i])
                    wer.reshape((1,1))
                    for j in iddict.keys():
                        if (wer==iddict[j]).any():
                            sr = j
                    ar = cv2.imread(sr)
                    c1,theta1=arucoangle(ar)
                    f=rotate_image(ar,theta1-(theta*180/math.pi),c1)
                    df = crp(f)
                    s=cv2.resize(df,newshape)
                    imnew[(ymin+15):(ymax-15),(xmin+15):(xmax-15),:]=s
            g.append(approx)

for c in contours[1:]:                                          #to copy remaining data to new image
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.01 * peri, True)
    if len(approx) == 4:
        x,y,w,h=cv2.boundingRect(approx)
        aspectratio = float(w)/h
    else:
        aspectratio = 0
    if aspectratio<=0.95 or aspectratio>=1.05:
        cord = [o[0].tolist() for o in approx]
        xmax,xmin,ymax,ymin = maxcord(cord)
        if len(approx)>6:
            to1 = img[ymin:ymax,(xmin-10):(xmax+10)]
            shape1 = to1.shape
            sw=shape1[0]
            se = shape1[1]
            
            newshape = (se,sw)
            to1 = cv2.resize(to1,newshape)
            imnew[ymin:(ymax),(xmin-10):(xmax+10),:]=to1
        else:
            to1 = img[ymin:ymax,(xmin):(xmax)]
            shape1 = to1.shape
            sw=shape1[0]
            se = shape1[1]
            
            newshape = (se,sw)
            to1 = cv2.resize(to1,newshape)
            imnew[ymin:(ymax),xmin:(xmax),:]=to1
cv2.drawContours(imnew,g, -1, (255, 0, 0), 3)
cv2.imwrite('new.jpg',imnew)                                    #saving final image

        
    
    
    
       
        
        



       


