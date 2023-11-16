import cv2,dlib
import sys
from renderFace import renderFace
import cv2
import face_recognition
# from PIL import Image, ImageDraw
import numpy as np

PREDICTOR_PATH = "../../../../Python Practice/Kivy/FirstProject/VirtualJewelery/data/models/shape_predictor_68_face_landmarks.dat"
jewel_img = cv2.imread("../../../../Python Practice/Kivy/FirstProject/VirtualJewelery/data/images/jewelery.png")
RESIZE_HEIGHT = 480
SKIP_FRAMES = 2

try:
    winName = "VitrualMakeUP"
    cap = cv2.VideoCapture(0)
    if(cap.isOpened is False):
        print("Unable to open Camera")
        sys.exit()

    fps = 30.0
    ret, im = cap.read()

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

    while(True):
        if count == 0:
            t = cv2.getTickCount()
        ret, im = cap.read()
        imDlib = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        imSmall = cv2.resize(im, None, fx=1.0 / RESIZE_SCALE, fy=1.0 / RESIZE_SCALE, interpolation=cv2.INTER_LINEAR)
        imSmallDlib = cv2.cvtColor(imSmall, cv2.COLOR_BGR2RGB)
        if (count % SKIP_FRAMES == 0):
            # Detect faces
            faces = detector(imSmallDlib, 0)

            # Iterate over faces
        for face in faces:
            # Since we ran face detection on a resized image,
            # we will scale up coordinates of face rectangle
            newRect = dlib.rectangle(int(face.left() * RESIZE_SCALE),
                                     int(face.top() * RESIZE_SCALE),
                                     int(face.right() * RESIZE_SCALE),
                                     int(face.bottom() * RESIZE_SCALE))

            # Find face landmarks by providing reactangle for each face
            shape = predictor(imDlib, newRect)
            x = shape.part(3).x-15
            y = shape.part(8).y

            img_width = abs(shape.part(3).x - shape.part(14).x)+20
            img_height = int(1.02 * img_width)

            jewel_area = cv2.getRectSubPix(im, (img_width, img_height), (x + img_width / 2, y + img_height / 2))

            #jewel_area = im[y:y + img_height, x:x + img_width]

            jewel_imgResized = cv2.resize(jewel_img, (img_width, img_height), interpolation=cv2.INTER_AREA)
            jewel_gray = cv2.cvtColor(jewel_imgResized, cv2.COLOR_BGR2GRAY)

            thresh, jewel_mask = cv2.threshold(jewel_gray, 230, 255, cv2.THRESH_BINARY)
            jewel_imgResized[jewel_mask == 255] = 0
            # jewel_area = im[y:y + img_height, x:x + img_width]

            masked_jewel_area = cv2.bitwise_and(jewel_area, jewel_area, mask=jewel_mask)
            final_jewel = cv2.add(masked_jewel_area, jewel_imgResized)
            # print("final ", final_jewel.shape)
            # #
            # print( "tt" , y + final_jewel.shape[1],x+ final_jewel.shape[0])

            im[y:y + final_jewel.shape[0], x:x+ final_jewel.shape[1]] = final_jewel
            #im[y:y + img_height, x:x + img_width] = final_jewel
            # convert image to RGB format to read it in pillow library
            #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            # pil_img = Image.fromarray(rgb_img)
            # im = ImageDraw.Draw(pil_img, 'RGBA')

            # M = np.float32([[1, 0, x], [0, 1, y]])
            # final_jewel_transformed = cv2.warpAffine(final_jewel, M, (im.shape[1], im.shape[0]), borderValue=(0, 0, 0))
            # im = cv2.add(im, final_jewel_transformed)




            # Put fps at which we are processinf camera feed on frame
        cv2.putText(im, "{0:.2f}-fps".format(fps), (50, size[0] - 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255),1)
        # Display it all on the screen
        cv2.imshow(winName, im)
        # Wait for keypress
        key = cv2.waitKey(1) & 0xFF

        # Stop the program.
        if key == 27:  # ESC
            # If ESC is pressed, exit.
            sys.exit()

        # increment frame counter
        count = count + 1
        # calculate fps at an interval of 100 frames
        if (count == 100):
            t = (cv2.getTickCount() - t) / cv2.getTickFrequency()
            fps = 100.0 / t
            count = 0
    cv2.destroyAllWindows()
    cap.release()



except Exception as e:
  print(e)