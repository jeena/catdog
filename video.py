import cv

capture = cv.CaptureFromCAM(3)

while (cv.WaitKey(15)==-1):
    img = cv.QueryFrame(capture)
    if img:
    	#image = DetectFace(img, faceCascade)
    	cv.ShowImage("face detection test", img)
    print type(img)
 
cv.ReleaseCapture(capture)