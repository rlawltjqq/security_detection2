import numpy as np
import cv2
import winsound #window 용
import playsound  # 1.2.2

classes = [] # 파이썬으로 배우는 인공지능 참고 
f = open('obj.names', 'r')
classes = [line.strip() for line in f.readlines()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

yolo_model = cv2.dnn.readNet('yolov4-tiny_20000_1.weights', 'yolov4-tiny_1.cfg')  # 욜로 읽어오기
layer_names = yolo_model.getLayerNames()
output_layers = [layer_names[i - 1]
                 for i in yolo_model.getUnconnectedOutLayers()]


def process_video():  # 비디오에서 침입자 검출해 알리기
    video = cv2.VideoCapture(0)
    while video.isOpened():
        success, img = video.read()
        if success:
            height, width, channels = img.shape
            blob = cv2.dnn.blobFromImage(
                img, 1.0/256, (416, 416), (0, 0, 0), swapRB=True, crop=False)

            yolo_model.setInput(blob)
            output = yolo_model.forward(output_layers)

            class_ids, confidences, boxes = [], [], []
            for output in output:
                for vec85 in output:
                    scores = vec85[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence >= 0.75:  # 신뢰도가 60% 이상인 경우만 취함
                        
                        centerx, centery = int(
                            vec85[0]*width), int(vec85[1]*height)
                        w, h = int(vec85[2]*width), int(vec85[3]*height) # [0,1] 표현을 영상 크기로 변환
                        x, y = int(centerx-w/2), int(centery-h/2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    text = str(classes[class_ids[i]]) + \
                        '%.2f' % confidences[i]  # 소숫점 2자리
                    cv2.rectangle(img, (x, y), (x+w, y+h),
                                  colors[class_ids[i]], 2)
                    cv2.putText(
                        img, text, (x, y+30), cv2.FONT_HERSHEY_PLAIN, 2, colors[class_ids[i]], 2)
        
            cv2.imshow('security detection', img)
            
            if 0 in class_ids:
                print('보행자 인식')  
                #winsound.Beep(frequency=2000, duration=500)
                playsound.playsound('person.mp3')
                
                # cv2.imwrite("%s_%05d_%3d.jpg" % (filepath, i, 0), img) # 이미지 저장
                # i += 1 # 이미지 저장
                
            elif 1 in class_ids:
                print('차량 접근')
                #winsound.Beep(frequency=2000, duration=500)
                #playsound.playsound('car.mp3')

            elif 2 in class_ids:
                print('자전거 접근')
                winsound.Beep(frequency=2000, duration=500)

            elif 3 in class_ids:
                print('오토바이 접근')
                winsound.Beep(frequency=2000, duration=500)

            elif 4 in class_ids:  # 자전거
                print('전동킥보드 접근')
                winsound.Beep(frequency=2000, duration=500)

            if 0 in class_ids and 1 in class_ids:
                print('차량 접근 알림')
                #winsound.Beep(frequency=5000, duration=2000)
                playsound.playsound('car2.mp3')
            if 0 in class_ids and 2 in class_ids:
                print('자전거 접근 알림')
                winsound.Beep(frequency=5000, duration=700)

            if 0 in class_ids and 3 in class_ids:
                print('오토바이 접근 알림')
                winsound.Beep(frequency=5000, duration=1500)

            if 0 in class_ids and 4 in class_ids:
                print('킥보드 접근 알림')
                winsound.Beep(frequency=5000, duration=900)

            # else :
                # print('인식 대기중') # 작동 확인용

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break  # q

    video.release()
    cv2.destroyAllWindows()


process_video()
