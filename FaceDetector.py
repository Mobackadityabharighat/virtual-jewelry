import cv2
import cv2,dlib
import sys

PREDICTOR_PATH = "models/shape_predictor_68_face_landmarks.dat"
jewel_img = cv2.imread("Images/alok2.png")
RESIZE_HEIGHT = 480
SKIP_FRAMES = 2
winName = "VitrualNeckless"

def init():
    camera = cv2.VideoCapture(0)
    return camera


def PlaceObject(imgName):

    camera = cv2.VideoCapture(0)
    jewel_img = cv2.imread("Images/"+imgName)
    if (camera.isOpened is False):
        print("Unable to open Camera")
        sys.exit()

    fps = 30.0
    ret, im = camera.read()

    if(ret == True):
        height = im.shape[0]
        RESIZE_SCALE = float(height)/RESIZE_HEIGHT
        size = im.shape[0:2]
    else:
        print("Unable to read Frame")

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    # initiate the tickCounter
    t = cv2.getTickCount()
    count = 0


    try:
        while True:
            if count == 0:
                t = cv2.getTickCount()
            success, frame = camera.read()  # read the camera frame
            if not success:
                break
            else:
                imDlib = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                imSmall = cv2.resize(frame, None, fx=1.0 / RESIZE_SCALE, fy=1.0 / RESIZE_SCALE,
                                     interpolation=cv2.INTER_LINEAR)
                imSmallDlib = cv2.cvtColor(imSmall, cv2.COLOR_BGR2RGB)
                if (count % SKIP_FRAMES == 0):
                    # Detect faces
                    faces = detector(imSmallDlib, 0)

                '''Run ForLoop for Number  Of Faces Detected '''
                for face in faces:
                    newRect = dlib.rectangle(int(face.left() * RESIZE_SCALE),
                                             int(face.top() * RESIZE_SCALE),
                                             int(face.right() * RESIZE_SCALE),
                                             int(face.bottom() * RESIZE_SCALE))

                    shape = predictor(imDlib, newRect)
                    x = shape.part(3).x - 15
                    y = shape.part(8).y

                    img_width = abs(shape.part(3).x - shape.part(14).x) + 20
                    img_height = int(1.02 * img_width)

                    jewel_area = cv2.getRectSubPix(frame, (img_width, img_height), (x + img_width / 2, y + img_height / 2))

                    jewel_imgResized = cv2.resize(jewel_img, (img_width, img_height), interpolation=cv2.INTER_AREA)
                    jewel_gray = cv2.cvtColor(jewel_imgResized, cv2.COLOR_BGR2GRAY)

                    thresh, jewel_mask = cv2.threshold(jewel_gray, 230, 255, cv2.THRESH_BINARY)
                    jewel_imgResized[jewel_mask == 255] = 0

                    masked_jewel_area = cv2.bitwise_and(jewel_area, jewel_area, mask=jewel_mask)
                    final_jewel = cv2.add(masked_jewel_area, jewel_imgResized)

                    if (im[y:y + final_jewel.shape[0], x:x + final_jewel.shape[1]].shape == final_jewel.shape):
                         frame[y:y + final_jewel.shape[0], x:x + final_jewel.shape[1]] = final_jewel

                key = cv2.waitKey(1) & 0xFF
                print(key)
                if key ==  cv2.WINDOW_NORMAL:  # ESC
                    print("Escaped")
                    # If ESC is pressed, exit.
                    sys.exit()

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

                count = count + 1

                if (count == 100):
                    t = (cv2.getTickCount() - t) / cv2.getTickFrequency()
                    fps = 100.0 / t
                    count = 0



        cv2.destroyAllWindows()
        frame.release()



    except Exception as e:
        print(e)