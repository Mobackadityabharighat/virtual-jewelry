from flask import Flask, render_template, Response, jsonify, request
import cv2
import requests
import FaceDetector as detector

app=Flask(__name__)
camera = cv2.VideoCapture(0)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(detector.PlaceObject("alok1.png", camera), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_api', methods=['POST'])
def video_feed_api():
    frame = request.form['frame']
    data = detector.PlaceObject1("alok1.png", frame)
    return Jsonify(data)


if __name__=='__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=5000, debug=False)