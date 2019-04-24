
import cv2
import time
from keras.preprocessing import image
from keras.models import Model, load_model

import numpy as np
from visualization_gui import plot_emotion_prediction


model = load_model('model_2_Dropout.h5')

#plot_emotion_prediction(custom[0])
emotion_dict = {}
ref_dict = {0: 'angry', 1: 'disgust', 2:'fear', 3: 'happy', 4:'sad', 5:'surprise', 6: 'neutral'}
cap = cv2.VideoCapture(0)
cap.set(3, 640) #WIDTH
cap.set(4, 480) #HEIGHT
time_ini = time.time()
interval = 2
face_cascade = cv2.CascadeClassifier('venv/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('venv/lib/python3.6/site-packages/cv2/data/haarcascade_eye.xml')

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    #print(len(faces)) -> detected or not
    # Display the resulting frame
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        k = time.time()
        if k-time_ini >= interval:
            #print("[INFO] Object found. Saving locally.")
            cv2.imwrite('data/faces.jpg', roi_color)
            time_ini = time.time()
            img = image.load_img("data/faces.jpg", grayscale=True, target_size=(48, 48))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis = 0)
            x /= 255
            custom = model.predict(x)
            #(value, emotion) = (max([(v,i) for i,v in enumerate(custom[0])]))
            #print(custom[0])
            #0: 'angry', 1: 'disgust', 2:'fear', 3: 'happy', 4:'sad', 5:'surprise', 6: 'neutral'
            emotion_dict['angry'] = custom[0][0]
            emotion_dict['disgust'] = custom[0][1]
            emotion_dict['fear'] = custom[0][2]
            emotion_dict['happy'] = custom[0][3]
            emotion_dict['sad'] = custom[0][4]
            emotion_dict['surprise'] = custom[0][5]
            emotion_dict['neutral'] = custom[0][6]
            print(emotion_dict)
            (value, emotion) = (max([(v,i) for i,v in enumerate(custom[0])]))
            print(ref_dict[emotion])

         # eyes = eye_cascade.detectMultiScale(roi_gray)
         # for (ex,ey,ew,eh) in eyes:
         #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()