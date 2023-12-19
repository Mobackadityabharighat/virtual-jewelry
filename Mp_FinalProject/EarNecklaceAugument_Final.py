import imutils
import numpy as np
import cv2
import mediapipe as mp
import time
import math
import PoseModule as pm


"""Necklace Augumentation"""

class Augumentation:
    def resize_image_Necklace(self, image, width=None, height=None):
        h, w = image.shape[:2]

        if width is not None and height is not None:
            dim = (width, height)
        elif width is not None:
            dim = (width, int(h * (width / w)))
        elif height is not None:
            dim = (int(w * (height / h)), height)
        else:
            return image

        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


    def rotate_image_Necklace(self, image, angle):
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, 180 - angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
        return rotated_image


    offset = (100, 130)


    def augment_necklace(self, image, landmarks, necklaceName):
        necklace = cv2.imread('images/'+ necklaceName, cv2.IMREAD_UNCHANGED)

        if necklace is None:
            print(f"Error: Necklace image '{necklaceName}' not found or could not be read.")
            return

            # Extract shoulder landmarks
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]

        # Check if both shoulders are detected
        if left_shoulder and right_shoulder:
            # Calculate shoulder width
            shoulder_width = int(abs(right_shoulder[1] - left_shoulder[1]) / 2)
            calculated_height = int(1.02 * shoulder_width)

            inclination_angle = np.arctan2(right_shoulder[2] - left_shoulder[2],
                                           right_shoulder[1] - left_shoulder[1]) * 180 / np.pi


            offset = (int(shoulder_width / 2)+10, int(calculated_height / 1.7))
            print(shoulder_width, offset)
            # Resize the necklace based on shoulder width
            if (shoulder_width > 0):
                necklace_resized = self.resize_image_Necklace(necklace, width = shoulder_width)

                # Rotate the necklace image
                rotated_necklace = self.rotate_image_Necklace(necklace_resized, inclination_angle)

                x_position_center = int((left_shoulder[1] + right_shoulder[1]) / 2)
                y_position_center = int((left_shoulder[2] + right_shoulder[2]) / 2)
                # Overlay the necklace on the image
                self.overlay_Necklace(image, rotated_necklace, (x_position_center - offset[0], y_position_center - offset[1]))


    def overlay_Necklace(self,frame, overlay, position):
        x, y = position
        y1, y2 = y, y + overlay.shape[0]
        x1, x2 = x, x + overlay.shape[1]

        alpha_s = overlay[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        # print(overlay.shape[0:2], frame[y1:y2, x1:x2].shape[0:2], y,x)
        if overlay.shape[0:2] == frame[y1:y2, x1:x2].shape[0:2]:
            for c in range(0, 3):
                frame[y1:y2, x1:x2, c] = (alpha_s * overlay[:, :, c] +
                                          alpha_l * frame[y1:y2, x1:x2, c])


    """Earring Augmentation"""
    def CalculateDistance(self,face_landmarks,frame):
        average_inter_eye_distance = 6.5
        # Get the vertical separation of the eyes (e.g., the distance between landmarks 247 and 33)
        eye_distance = math.sqrt((face_landmarks.landmark[33].x - face_landmarks.landmark[247].x) ** 2 +
                                 (face_landmarks.landmark[33].y - face_landmarks.landmark[247].y) ** 2 +
                                 (face_landmarks.landmark[33].z - face_landmarks.landmark[247].z) ** 2)

        reversed_depth = 1 / ((average_inter_eye_distance * frame.shape[1]) / eye_distance)
        return reversed_depth


    def augment_earring(self,frame, piercing_points, earringName, face_landmarks, ear_side, y_rotation, drawPoints = False):
        earring = cv2.imread('Images/' + earringName, cv2.IMREAD_UNCHANGED)
        if earring is None:
            print("Error: Earring image not found.")
            return

        # Extract piercing points
        x1, y1 = piercing_points
        # calDist =70 + (CalculateDistance(face_landmarks,frame))*10000000
        calDist = (self.CalculateDistance(face_landmarks, frame)) * 60000000

        w = np.abs(calDist * .009)
        # print(w)

        if drawPoints:
            cv2.circle(frame, (x1, y1), 5, (255, 255, 0), cv2.FILLED)

        # Resize the earring based on the distance between piercing points
        # width = int(w*1.2)
        # height = int(  width)

        scale_factor = 0.10 * w  # Adjust this value based on your preference
        print(scale_factor)
        earring_size = int(scale_factor * frame.shape[1])

        # Resize the earring
        earring_resized = cv2.resize(earring, (earring_size, int(earring_size * 1.3)), interpolation=cv2.INTER_AREA)

        # if width > 0:
        #     earring_resized = cv2.resize(earring, (width, height), interpolation=cv2.INTER_AREA)

        # Calculate the position to place the earring
        '''int(calDist/2)is to adjust the postion on moving to or away from camera'''
        if ear_side == "LEFT":
            x_1 = x1 - int(earring_size / 2) - int(y_rotation * 1.2) + int(calDist * .12)
            y_1 = y1 - int(calDist * .12)
            self.overlay_earring(frame, earring_resized, (x_1, y_1))

        else:
            x_1 = x1 - int(earring_size / 2) - int(y_rotation * 1.2) + int(calDist * .12)
            y_1 = y1 - int(calDist * .12)
            self.overlay_earring(frame, earring_resized, (x_1, y_1))
            # x_1 = int(x1 -(y_rotation*1.8)) - int(calDist/2)# 40*1.8
            # y_1 = (y1)-int(calDist*.12) #20
            # overlay(frame, earring_resized, (x_1, y_1))

    def overlay_earring(self,background, overlay, position1, position2=None):

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



    def CalculateFaceRotation(self,frame,face_landmarksList):
        img_h, img_w, img_c = frame.shape
        face_2d = []
        face_3d = []
        # global nose_2d,
        # nose_3d, cam_matrix, distortion_matrix, success, rotation_vec, translation_vec
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


    def main(self,earringName = None,necklaceName = None ):

        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5,min_tracking_confidence=0.5)
        mp_drawing = mp.solutions.drawing_utils
        drawing_spec = mp_drawing.DrawingSpec(color=(128,0,128),thickness=2,circle_radius=1)
        cap = cv2.VideoCapture(0)
        success, frame = cap.read()
        frame = imutils.resize(frame, width=1024, height=720)
        w, h = frame.shape[:2]
        detector = pm.PoseDetector(False, 1, True, True)
        # earringName = "earring3.png"
        can_draw = False
        while cap.isOpened():
            success, frame = cap.read()
            frame = imutils.resize(frame, width=1024, height=720)
            start = time.time()
            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            '''shoulders'''
            frame = detector.findPos(frame, False)
            lmList = detector.getPosition(frame, False)

            '''earring'''

            frame.flags.writeable = False
            results = face_mesh.process(frame)
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

            if necklaceName is not None:
                if (len(lmList) > 0):
                    # cv2.circle(frame, (lmList[12][1], lmList[12][2]), 5, (255, 0, 0), cv2.FILLED)
                    # cv2.circle(frame, (lmList[11][1], lmList[11][2]), 5, (255, 0, 0), cv2.FILLED)
                    self.augment_necklace(frame, lmList, necklaceName)

            if earringName is not None:
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        x,y,z = self.CalculateFaceRotation(frame,face_landmarks)

                        #here based on axis rot angle is calculated
                        if y < -6.5:
                            text = "Looking Left"
                        elif y > 6.5:
                            text="Looking Right"
                        elif x < -10:
                            text="Looking Down"
                        elif x > 12:
                            text="Looking Up"
                        else:
                            text="Forward"


                        if can_draw:
                            cv2.putText(frame,text,(20,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)
                            cv2.putText(frame,"x: " + str(np.round(x,2)),(500,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                            cv2.putText(frame,"y: "+ str(np.round(y,2)),(500,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                            cv2.putText(frame,"z: "+ str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                            mp_drawing.draw_landmarks(image=frame,
                                                      landmark_list=face_landmarks,
                                                      connections=mp_face_mesh.FACEMESH_CONTOURS,
                                                      landmark_drawing_spec=drawing_spec,
                                                      connection_drawing_spec=drawing_spec)
                        if(text !="Looking Up"):
                            if(text !="Looking Right"  ):
                                percingPoint1 = (int((face_landmarks.landmark[361].x) * h) + 10, int(face_landmarks.landmark[361].y * w))
                                #cv2.circle(image, percingPoint1, 5,(255, 0, 0), cv2.FILLED)
                                self.augment_earring(frame, percingPoint1, earringName, face_landmarks, "RIGHT", y, False)
                            if (text != "Looking Left"):
                                pearcingPoint2 = (int(face_landmarks.landmark[132].x * h) - 20, int(face_landmarks.landmark[132].y * w))
                                #cv2.circle(image, pearcingPoint2, 5,(255, 0, 0), cv2.FILLED)
                                self.augment_earring(frame, pearcingPoint2, earringName, face_landmarks, "LEFT", y, False)
                        else:
                            self.ShowFaceNotDetected(frame)

                end = time.time()
                totalTime = end-start

                fps = 1/totalTime
                # print("FPS: ",fps)

                cv2.putText(frame,f'FPS: {int(fps)}',(10,200),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),2)

            cv2.imshow('Head Pose Detection',frame)
            if cv2.waitKey(5) & 0xFF ==27:
                break
        cap.release()


    def ShowFaceNotDetected(self, image):
        w,h = image.shape[:2]
        cv2.putText(image, "Find a Face ", (int(w / 2), int(h / 2) - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        cv2.putText(image, "Keep 10inch distance ", (int(w / 2), int(h / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (255, 255, 255), 2)
        cv2.putText(image, " ensure Visibility", (int(w / 2), int(h / 2) + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (255, 255, 255), 2)


aug = Augumentation()
if __name__ == "__main__":
    aug.main("earring3.png", "NeckLace3.png")
    # aug.main("earring3.png",None)
    #aug.main(None, "NeckLace2.png")