import cv2, dlib
import sys


class NecklaceAugumentation:
    def __init__(self):
        self.PREDICTOR_PATH = "../../Python Practice/ServerProgramWithAugumentation/models/shape_predictor_68_face_landmarks.dat"
        self.jewel_img = cv2.imread("../../Python Practice/ServerProgramWithAugumentation/Images/alok2.png")
        self.RESIZE_HEIGHT = 480
        #self.SKIP_FRAMES = 2
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.PREDICTOR_PATH)

    def Augument(self, frame, jewel_img):
        height = frame.shape[0]
        RESIZE_SCALE = float(height) / self.RESIZE_HEIGHT
        imDlib = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imSmall = cv2.resize(frame, None, fx=1.0 / RESIZE_SCALE, fy=1.0 / RESIZE_SCALE, interpolation=cv2.INTER_LINEAR)
        imSmallDlib = cv2.cvtColor(imSmall, cv2.COLOR_BGR2RGB)

        # Detect faces
        faces = self.detector(imSmallDlib, 0)

        '''Run ForLoop for Number  Of Faces Detected '''
        for face in faces:
            newRect = dlib.rectangle(int(face.left() * RESIZE_SCALE),
                                     int(face.top() * RESIZE_SCALE),
                                     int(face.right() * RESIZE_SCALE),
                                     int(face.bottom() * RESIZE_SCALE))

            shape = self.predictor(imDlib, newRect)
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

            if (frame[y:y + final_jewel.shape[0], x:x + final_jewel.shape[1]].shape == final_jewel.shape):
                frame[y:y + final_jewel.shape[0], x:x + final_jewel.shape[1]] = final_jewel

        return frame


def main():
    global newFrame
    camera = cv2.VideoCapture(0)
    jewel_img = cv2.imread("../../Python Practice/ServerProgramWithAugumentation/Images/alok2.png")
    if (camera.isOpened is False):
        print("Unable to open Camera")
        sys.exit()
    na = NecklaceAugumentation()
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
                if (count % 2 == 0):
                    newFrame = na.Augument(frame, jewel_img)

                key = cv2.waitKey(1) & 0xFF
                if key == cv2.WINDOW_NORMAL:  # ESC
                    print("Escaped")
                    # If ESC is pressed, exit.
                    sys.exit()

                count = count + 1
                if (count == 100):
                    t = (cv2.getTickCount() - t) / cv2.getTickFrequency()
                    fps = 100.0 / t
                    count = 0

                cv2.imshow("winName", newFrame)

        frame.release()



    except Exception as e:
        print(e)
        # yield (b'--frame\r\n'
        #        b'Content-Type: image/jpeg\r\n\r\n' + str(e) + b'\r\n')


if __name__ == '__main__':
    main()
