import time
import cv2
import PoseModule as pm
import numpy as np

def resize_image_Necklace(image, width=None, height=None):
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

def rotate_image_Necklace(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 180-angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
    return rotated_image


offset = (100, 130)


def augment_necklace(image, landmarks,necklaceName):
    necklace = cv2.imread('images/' + necklaceName, cv2.IMREAD_UNCHANGED)

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



        offset = (int(shoulder_width / 2), int(calculated_height / 1.7))
        print(shoulder_width, offset)
        # Resize the necklace based on shoulder width
        if (shoulder_width > 0):
            necklace_resized = resize_image_Necklace(necklace, width=shoulder_width)

            # Rotate the necklace image
            rotated_necklace = rotate_image_Necklace(necklace_resized, inclination_angle)

            x_position_center = int((left_shoulder[1] + right_shoulder[1]) / 2)
            y_position_center = int((left_shoulder[2] + right_shoulder[2]) / 2)
            # Overlay the necklace on the image
            overlay_Necklace(image, rotated_necklace, (x_position_center - offset[0], y_position_center - offset[1]))


def overlay_Necklace(frame, overlay, position):
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


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = pm.PoseDetector(False, 1, True, True)

    while cap.isOpened():
        success, frame = cap.read()
        frame = detector.findPos(frame, False)
        lmList = detector.getPosition(frame, False)
        if (len(lmList) > 0):
            # cv2.circle(frame, (lmList[12][1], lmList[12][2]), 5, (255, 0, 0), cv2.FILLED)
            # cv2.circle(frame, (lmList[11][1], lmList[11][2]), 5, (255, 0, 0), cv2.FILLED)
            augment_necklace(frame, lmList,"NeckLace1.png")

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(frame, "Fps" + str(int(fps)), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)

        # Display the frame
        cv2.imshow('Augmented Necklace', frame)

        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
