from flask import Flask, render_template, Response
import cv2
import FaceDetector as detector

app=Flask(__name__)
camera = cv2.VideoCapture(0)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(detector.PlaceObject("alok1.png", camera), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=='__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=5000, debug=False)