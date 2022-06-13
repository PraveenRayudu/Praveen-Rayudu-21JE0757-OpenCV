import cv2
import numpy as np
import cv2.aruco as aruco 

#finding aruco marker ID's

Ha=cv2.imread("resources/Ha.jpg")
HaHa=cv2.imread("resources/HaHa.jpg")
LAMAO=cv2.imread("resources/LMAO.jpg")
XD=cv2.imread("resources/XD.jpg")

#defining function for finding aruco marker ids
def findaruco(main):
    imggrey = cv2.cvtColor(main,cv2.COLOR_BGR2GRAY)
    key = getattr(aruco,f'DICT_5X5_250')
    arucodict = aruco.Dictionary_get(key)
    arucoparam = aruco.DetectorParameters_create()

    (corners,ids,rejected) = cv2.aruco.detectMarkers(main,arucodict,parameters=arucoparam)
    print(ids)

findaruco(Ha)
findaruco(HaHa)
findaruco(LAMAO)
findaruco(XD)

cv2.namedWindow("output", cv2.WINDOW_NORMAL) 
cv2.resizeWindow("output",600,400)
main = cv2.imread("resources/CVtask.jpg")  
maingrey = cv2.cvtColor(main,cv2.COLOR_BGR2GRAY)

#shape detecting
_, thrash = cv2.threshold(maingrey, 240, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)
    cv2.drawContours(main, [approx], 0, (0, 0, 0), 5)
    area = cv2.contourArea(contour)
    if area>500:
        peri = cv2.arcLength(contour,True)
        #print(peri)
        objCor = len(approx)
        x, y, w, h = cv2.boundingRect(approx)
    if objCor == 3:
        cv2.putText(main, "Triangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255),2)
    elif objCor == 4:
        aspectRatio = float(w)/h
        if aspectRatio >= 0.98 and aspectRatio <= 1.03:
            print(x,y,w,h)
            objecttype="square"
            cv2.putText(main, "square", (x+(w//2)-10,y+(h//2)-10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255),2)
        else:
          cv2.putText(main, "rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255),2)
    elif objCor == 5:
        cv2.putText(main, "Pentagon", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255),2)
    elif objCor == 10:
        cv2.putText(main, "Star", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255),2)
    else:
        cv2.putText(main, "Circle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255),2)


#warping aruco codes and squares  
img1 = cv2.imread("resources/LMAO.jpg")
img2 = cv2.imread("resources/XD.jpg")
img3 = cv2.imread("resources/Ha.jpg")
img4 = cv2.imread("resources/HaHa.jpg")

 
width,height = 300,300
pts11 = np.float32([[51,162],[450,51],[562,450],[160,560]])
pts2 = np.float32([[0,0],[width,0],[width,height],[0,height]])
matrix1 = cv2.getPerspectiveTransform(pts11,pts2)
imgOutput1 = cv2.warpPerspective(img1,matrix1,(width,height))

pts12 = np.float32([[122,19],[575,123],[474,574],[21,473]])
matrix2 = cv2.getPerspectiveTransform(pts12,pts2)
imgOutput2 = cv2.warpPerspective(img2,matrix2,(width,height))

pts13 = np.float32([[77,77],[514,77],[514,514],[77,514]])
matrix3 = cv2.getPerspectiveTransform(pts13,pts2)
imgOutput3 = cv2.warpPerspective(img3,matrix3,(width,height))

pts14 = np.float32([[29,142],[448,29],[560,453],[140,566]])
matrix4 = cv2.getPerspectiveTransform(pts14,pts2)
imgOutput4 = cv2.warpPerspective(img4,matrix4,(width,height))

square1=np.float32([[201,34],[601,130],[504,529],[105,432]])
matrixsq1 = cv2.getPerspectiveTransform(square1,pts2)
sqOutput1 = cv2.warpPerspective(main,matrixsq1,(width,height))

square2=np.float32([[1169,77],[1522,78],[1522,427],[1174,428]])
matrixsq2 = cv2.getPerspectiveTransform(square2,pts2)
sqOutput2 = cv2.warpPerspective(main,matrixsq2,(width,height))

square3=np.float32([[1408,469],[1682,584],[1567,862],[1290,747]])
matrixsq3 = cv2.getPerspectiveTransform(square3,pts2)
sqOutput3 = cv2.warpPerspective(main,matrixsq3,(width,height))

square4=np.float32([[333,674],[796,417],[1048,880],[586,1139]])
matrixsq4 = cv2.getPerspectiveTransform(square4,pts2)
sqOutput4 = cv2.warpPerspective(main,matrixsq4,(width,height))




 

x_offset1=1169
y_offset1=75
x_end1 = x_offset1 + imgOutput1.shape[1]
y_end1 = y_offset1 + imgOutput1.shape[0]
x_offset2=1289
y_offset2=469
x_end2 = x_offset2 + imgOutput2.shape[1]
y_end2 = y_offset2 + imgOutput2.shape[0]
x_offset3=331
y_offset3=417
x_end3 = x_offset3 + imgOutput3.shape[1]
y_end3 = y_offset3 + imgOutput3.shape[0]
x_offset4=103
y_offset4=32
x_end4 = x_offset4 + imgOutput4.shape[1]
y_end4 = y_offset4 + imgOutput4.shape[0]



main[y_offset1:y_end1,x_offset1:x_end1] = imgOutput1
main[y_offset2:y_end2,x_offset2:x_end2] = imgOutput2
main[y_offset3:y_end3,x_offset3:x_end3] = imgOutput3
main[y_offset4:y_end4,x_offset4:x_end4] = imgOutput4

cv2.imshow("output",main)
cv2.waitKey(0)



















