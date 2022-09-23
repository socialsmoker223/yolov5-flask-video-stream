from flask import Flask, render_template, Response 
import cv2
import torch


app = Flask(__name__)
source = '0'
camera = cv2.VideoCapture(source)
model = torch.hub.load("ultralytics/yolov5", 'yolov5s', force_reload=True, skip_validation=True)

def predict(im):
    result = model(im, size=640)
    result.render()
    return result.ims[0]

def gen_frame():
    while True:
        success, frame = camera.read()
        if not success:
            break 
        else:
            frame = predict(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            # print(frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/")
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run('0.0.0.0', threaded=True)
