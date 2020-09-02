import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# recording
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
writer = cv2.VideoWriter("Object_Detection.mp4", fourcc, fps, (width, height))

# Threshold
Threshold = 0.5

classNames= []
classFile = 'classnames'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success,img = cap.read()
    classIds, confs, bbox = net.detect(img,confThreshold=Threshold)

    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            print(box,classNames[classId-1].upper(),str(round(confidence*100,2)))

            x, y, w, h = box
            top = max(0, np.floor(x + 0.5).astype(int))
            left = max(0, np.floor(y + 0.5).astype(int))
            right = min(img.shape[1], np.floor(x + w + 0.5).astype(int))
            bottom = min(img.shape[0], np.floor(y + h + 0.5).astype(int))

            # cv2.circle(img, (int((top + right) / 2), int((left + bottom) / 2)), 40, color=(0, 0, 255), thickness=3, lineType=cv2.LINE_AA)

            cv2.rectangle(img, (top-2, left-2), (right+2, bottom+2),color=(238, 255, 0),thickness=1)
            cv2.rectangle(img, (top-4, left-4), (right+4, bottom+4),color=(255, 145, 0),thickness=1)
            cv2.rectangle(img, (top-6, left-6), (right+6, bottom+6),color=(255, 0, 111),thickness=1)
            cv2.rectangle(img, (top-8, left-8), (right+8, bottom+8),color=(255, 0, 238),thickness=1)
            cv2.rectangle(img, (top-10, left-10), (right+10, bottom+10),color=(145, 0, 255),thickness=1)

            cv2.rectangle(img,box,color=(111,255,0),thickness=1)

            cv2.putText(img,classNames[classId-1].upper(),(box[0]+5,box[1]+20),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,0,255),1)
            cv2.putText(img,str(round(confidence*100,2)),(box[0]+5,box[1]+40),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,255,0),1)

    writer.write(img)
    cv2.imshow("Object_Detection",img)
    key = cv2.waitKey(1)
    if key == 27:
            cv2.destroyAllWindows()
            break
