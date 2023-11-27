import cv2
import numpy as np
# reading the vedio 
kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
source = cv2.VideoCapture('melon.mp4') 

# We need to set resolutions. 
# so, convert them from float to integer. 
frame_width = int(source.get(3)) 
frame_height = int(source.get(4)) 
   
size = (frame_width, frame_height) 

result = cv2.VideoWriter('gray.avi',  
            cv2.VideoWriter_fourcc(*'MJPG'), 
            10, size, 0) 
  
# running the loop 
while True: 
  
    # extracting the frames 
    ret, img = source.read() 
      
    # converting to gray-scale 
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.array(img)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    imagex = cv2.filter2D(gray, -1, kernelx)
    imagey = cv2.filter2D(gray, -1, kernely)
    gray = cv2.add(imagex, imagey)
     

    # write to gray-scale 
    result.write(gray)

    # displaying the video 
    cv2.imshow("Live", gray) 
  
    # exiting the loop 
    key = cv2.waitKey(10) 
    if key == ord("q"): 
        break
      
# closing the window 
cv2.destroyAllWindows() 
source.release()