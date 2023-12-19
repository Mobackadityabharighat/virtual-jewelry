import imutils
import numpy as np
import cv2
import mediapipe as mp
import time
import math

earringName = "earring3.png"

def CalculateDistance(face_landmarks,frame):
    average_inter_eye_distance = 6.5
    # Get the vertical separation of the eyes (e.g., the distance between landmarks 247 and 33)
    eye_distance = math.sqrt((face_landmarks.landmark[33].x - face_landmarks.landmark[247].x) ** 2 +
                             (face_landmarks.landmark[33].y - face_landmarks.landmark[247].y) ** 2 +
                             (face_landmarks.landmark[33].z - face_landmarks.landmark[247].z) ** 2)
    # Estimate the depth using the average inter-eye distance
    # estimated_depth = (average_inter_eye_distance * frame.shape[1]) / eye_distance
    # return estimated_depth
    reversed_depth = 1 / ((average_inter_eye_distance * frame.shape[1]) / eye_distance)
    return reversed_depth


def overlay_1earrings(frame, piercing_points, jweallaryname,  face_landmarks,ear_side,y_rotation,drawPoints = False):
    earring = cv2.imread('Images/' + jweallaryname, cv2.IMREAD_UNCHANGED)

    if earring is None:
        print("Error: Earring image not found.")
        return

    # Extract piercing points
    x1, y1 = piercing_points
    #calDist =70 + (CalculateDistance(face_landmarks,frame))*10000000
    calDist = (CalculateDistance(face_landmarks,frame))*60000000

    w = np.abs(calDist*.009)
    #print(w)

    if drawPoints:
        cv2.circle(frame, (x1, y1), 5, (255, 255, 0), cv2.FILLED)

    # Resize the earring based on the distance between piercing points
    # width = int(w*1.2)
    # height = int(  width)

    scale_factor = 0.10  * w # Adjust this value based on your preference
    print(scale_factor)
    earring_size = int(scale_factor * frame.shape[1])

    # Resize the earring
    earring_resized = cv2.resize(earring, (earring_size, int(earring_size*1.3)), interpolation=cv2.INTER_AREA)

    # if width > 0:
    #     earring_resized = cv2.resize(earring, (width, height), interpolation=cv2.INTER_AREA)

        # Calculate the position to place the earring
    '''int(calDist/2)is to adjust the postion on moving to or away from camera'''
    if ear_side == "LEFT":
        x_1 =x1 - int(earring_size/2) - int(y_rotation*1.2) + int(calDist*.12)
        y_1 = y1-int(calDist*.12)
        overlay(frame, earring_resized, (x_1, y_1))

    else:
        x_1 = x1 - int(earring_size / 2) - int(y_rotation * 1.2) + int(calDist * .12)
        y_1 = y1 - int(calDist * .12)
        overlay(frame, earring_resized, (x_1, y_1))
        # x_1 = int(x1 -(y_rotation*1.8)) - int(calDist/2)# 40*1.8
        # y_1 = (y1)-int(calDist*.12) #20
        # overlay(frame, earring_resized, (x_1, y_1))



        

def overlay(background, overlay, position1, position2=None):

    global lefty1, lefty2, leftx1, leftx2
    pos_x1, pos_y1 = position1
    righty1, righty2 = pos_y1, pos_y1 + overlay.shape[0]
    rightx1, rightx2 = pos_x1, pos_x1 + overlay.shape[1]

    if position2 != None:
        pos_x2, pos_y2 = position2
        lefty1, lefty2 = pos_y1, pos_y1 + overlay.shape[0]
        leftx1, leftx2 = pos_x2, pos_x2 + overlay.shape[1]

    alpha_s = overlay[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    if overlay.shape[0:2] == background[righty1:righty2, rightx1:rightx2].shape[0:2]:
        for c in range(0, 3):
            background[righty1:righty2, rightx1:rightx2, c] = (
                        alpha_s * overlay[:, :, c] + alpha_l * background[righty1:righty2, rightx1:rightx2, c])
            if position2 != None:
                background[lefty1:lefty2, leftx1:leftx2, c] = (
                            alpha_s * overlay[:, :, c] + alpha_l * background[lefty1: lefty2, leftx1 : leftx2 , c])



def CalculateRotation(frame,face_landmarksList):
    img_h, img_w, img_c = frame.shape
    face_2d = []
    face_3d = []
    # global nose_2d, nose_3d, cam_matrix, distortion_matrix, success, rotation_vec, translation_vec
    for idx, lm in enumerate(face_landmarksList.landmark):
        if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
            if idx == 1:
                nose_2d = (lm.x * img_w, lm.y * img_h)
                nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
            x, y = int(lm.x * img_w), int(lm.y * img_h)

            face_2d.append([x, y])
            face_3d.append(([x, y, lm.z]))
    # Get 2d Coord
    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)
    focal_length = 1 * img_w
    cam_matrix = np.array([[focal_length, 0, img_h / 2],
                           [0, focal_length, img_w / 2],
                           [0, 0, 1]])
    distortion_matrix = np.zeros((4, 1), dtype=np.float64)
    success, rotation_vec, translation_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, distortion_matrix)
    # getting rotational of face
    rmat, jac = cv2.Rodrigues(rotation_vec)
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
    x = angles[0] * 360
    y = angles[1] * 360
    z = angles[2] * 360
    return (x,y,z)


def main():

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5,min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(color=(128,0,128),thickness=2,circle_radius=1)
    cap = cv2.VideoCapture(0)
    success, image = cap.read()
    image = imutils.resize(image, width=1024, height=720)
    w, h = image.shape[:2]

    can_draw = False;
    while cap.isOpened():
        success, image = cap.read()
        image = imutils.resize(image, width=1024, height=720)
        # w, h = image.shape[:2]
        start = time.time()
        image = cv2.cvtColor(cv2.flip(image,1),cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)


        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                x,y,z = CalculateRotation(image,face_landmarks)

                #here based on axis rot angle is calculated
                if y < -6.5:
                    text = "Looking Left"
                elif y > 6.5:
                    text="Looking Right"
                elif x < -10:
                    text="Looking Down"
                elif x > 11:
                    text="Looking Up"
                else:
                    text="Forward"


                if can_draw:
                    cv2.putText(image,text,(20,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)
                    cv2.putText(image,"x: " + str(np.round(x,2)),(500,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                    cv2.putText(image,"y: "+ str(np.round(y,2)),(500,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                    cv2.putText(image,"z: "+ str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    mp_drawing.draw_landmarks(image=image,
                                              landmark_list=face_landmarks,
                                              connections=mp_face_mesh.FACEMESH_CONTOURS,
                                              landmark_drawing_spec=drawing_spec,
                                              connection_drawing_spec=drawing_spec)
                if(text !="Looking Up"):
                    if(text !="Looking Right"  ):
                        percingPoint1 = (int((face_landmarks.landmark[361].x) * h) + 10, int(face_landmarks.landmark[361].y * w))

                        overlay_1earrings(image, percingPoint1, earringName, face_landmarks,"RIGHT",y,True)
                        #cv2.circle(image, percingPoint1, 5, (255, 0, 0), cv2.FILLED)
                    if (text != "Looking Left"):
                        pearcingPoint2 = (int(face_landmarks.landmark[132].x * h) - 20, int(face_landmarks.landmark[132].y * w))
                        #
                        overlay_1earrings(image, pearcingPoint2, earringName, face_landmarks,"LEFT",y, False)
                        #cv2.circle(image, pearcingPoint2, 5, (255, 0, 0), cv2.FILLED)
                else:
                    ShowFaceNotDetected( image)

            end = time.time()
            totalTime = end-start

            fps = 1/totalTime
            # print("FPS: ",fps)

            cv2.putText(image,f'FPS: {int(fps)}',(w-80,200),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),2)

        cv2.imshow('Head Pose Detection',image)
        if cv2.waitKey(5) & 0xFF ==27:
            break
    cap.release()


def ShowFaceNotDetected( image):
    w,h = image.shape[:2]
    cv2.putText(image, "Find a Face ", (int(w / 2), int(h / 2) - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    cv2.putText(image, "Keep 10inch distance ", (int(w / 2), int(h / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                (255, 255, 255), 2)
    cv2.putText(image, " ensure Visibility", (int(w / 2), int(h / 2) + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                (255, 255, 255), 2)


if __name__ == "__main__":
    main()