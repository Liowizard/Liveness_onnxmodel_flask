import numpy as np
import onnxruntime as rt
from flask import Flask, Response ,render_template_string ,request
import cv2



app=Flask(__name__)
video = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier()
face_cascade.load(cv2.samples.findFile("haarcascade_frontalface_alt.xml"))





def predict(img, model_path="face_liveness.onnx"):
    if img.shape != (112, 112, 3):
        return -1

    dummy_face = np.expand_dims(np.array(img, dtype=np.float32), axis=0) / 255.

    providers = ['CPUExecutionProvider']
    m = rt.InferenceSession(model_path, providers=providers)
    onnx_pred = m.run(['activation_5'], {"input": dummy_face})
    # print(onnx_pred)
    # print(dummy_face.shape)
    liveness_score = list(onnx_pred[0][0])[1]
    # print(dummy_face)
    return liveness_score





def gen(video):
    while True:
        success, image = video.read()
        frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)

        faces = face_cascade.detectMultiScale(frame_gray,scaleFactor=1.1, minNeighbors=4)

        for (x, y, w, h) in faces:
            faceRegion = image[y:y + h, x:x + w]
            x
            faceRegion = cv2.resize(faceRegion, (112, 112))
            ff1s = predict(faceRegion)
            score=str(int(ff1s*100))+"%"
        cv2.putText(image, "Liveness: " + str(ff1s) , (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0, 255), 3)
        if ff1s < 0.5:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(image, 'Fake', (x, y + h + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
        elif ff1s <= 0.9:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 165, 255), 2)
            cv2.putText(image, "Face not clear", (x, y + h + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 165, 255))
        else:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, 'Real', (x, y + h + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))





            # center = (x + w//2, y + h//2)
            
            # image = cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

            faceROI = frame_gray[y:y+h, x:x+w]
        ret, jpeg = cv2.imencode('.jpg', image)

        frame = jpeg.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        # yield(frame)







@app.route('/', methods=['GET', 'POST'])

def index():
    if request.method == 'POST':
        # Function to be executed when start button is clicked
        
        return Video()
    else:
         start_button =  '''
            <form method="POST">
            <div style="text-align:center">
            <h1 style="margin-top:3rem;margin-bottom:1rem">Welcome to digival👨🏻‍🎓</h1>
            <img src="https://www.digi-val.com/assets/img/cover-img.svg" alt="Girl in a jacket" width="500" height="350">
            <p style="margin-top:5rem;margin-bottom:1rem">Click here to start <br/> 👇👇👇👇</p>
                <button type="submit">Liveness_detector</button>
                </div>
            </form>
        '''
         return render_template_string(start_button)

def Video():

    global video

    return Response(gen(video),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)
