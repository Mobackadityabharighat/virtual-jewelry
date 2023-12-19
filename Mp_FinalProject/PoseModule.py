import cv2
import mediapipe as mp;
import time


class PoseDetector():

    def __init__(self,mode = False, complex = 1, smooth = True, enable_seg = False, smooth_seg= True, detConf =0.5, trackConf= 0.5):
        self.mode = mode
        self.complex = complex
        self.smooth = smooth
        self.enable_seg = enable_seg
        self.smooth_seg = smooth_seg
        self.detConf = detConf
        self.trackConf= trackConf

        self.mpdraw = mp.solutions.drawing_utils
        self.mPose = mp.solutions.pose
        self.pose = self.mPose.Pose(self.mode,self.complex,self.smooth,self.enable_seg,self.smooth_seg,self.detConf,self.trackConf)

    def findPos(self, frame, draw = True ):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.result = self.pose.process(imgRGB)

        if(self.result.pose_landmarks):
            if (draw):
                 self.mpdraw.draw_landmarks(frame,self.result.pose_landmarks,self.mPose.POSE_CONNECTIONS)

        return frame


    def getPosition(self, frame, draw = True):
        lmList = []
        if (self.result.pose_landmarks):
            for id, lm in enumerate(self.result.pose_landmarks.landmark):
                h,w,c= frame.shape
                #print(id,lm.x,lm.y)
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id,cx,cy])
                if(draw):
                    cv2.circle(frame,(cx,cy),10,(255,255,0),cv2.FILLED )
        return lmList




def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = PoseDetector(False,1,True,True)
    while (True):
        success, frame = cap.read()
        frame = detector.findPos( frame , False)
        lmList = detector.getPosition(frame, False)
        if(len(lmList)>0):
            # print("list-> ",lmList[3][0][0])
            cv2.circle(frame, (lmList[15][1], lmList[15][2]), 5, (255, 255, 0), cv2.FILLED)
            cv2.circle(frame, (lmList[16][1], lmList[16][2]), 5, (255, 255, 0), cv2.FILLED)
            cv2.circle(frame, (lmList[19][1], lmList[19][2]), 5, (255, 255, 0), cv2.FILLED)
            cv2.circle(frame, (lmList[20][1], lmList[20][2]), 5, (0, 255, 0), cv2.FILLED)
            #cv2.circle(frame, (lmList[9][1], lmList[9][2]), 5, (0, 255, 0), cv2.FILLED)
        # print(lmList[1])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(frame, "Fps" + str(int(fps)), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)

        cv2.imshow("Img", frame)

        cv2.waitKey(1)


if __name__ == "__main__":
    main()
