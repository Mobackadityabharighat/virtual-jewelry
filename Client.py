import cv2
import socket
import struct
import pickle
import imutils

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('127.0.0.1', 8485))  # Replace with your server's IP and port

cam = cv2.VideoCapture(0)
img_counter = 0

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

while True:
    ret, frame = cam.read()

    frame = imutils.resize(frame, width=320)
    frame = cv2.flip(frame, 180)

    result, image = cv2.imencode('.jpg', frame, encode_param)
    data = pickle.dumps(image, 0)
    size = len(data)

    client_socket.sendall(struct.pack(">L", size) + data)

    # Receive augmented feed from the server
    data = b""
    payload_size = struct.calcsize(">L")
    while len(data) < payload_size:
        data += client_socket.recv(4096)

    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack(">L", packed_msg_size)[0]

    while len(data) < msg_size:
        data += client_socket.recv(4096)

    frame_data = data[:msg_size]
    data = data[msg_size:]

    # Decode and display the augmented frame
    augmented_frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
    augmented_frame = cv2.imdecode(augmented_frame, cv2.IMREAD_COLOR)
    #cv2.putText(augmented_frame ,"Client", (100,100),cv2.FONT_HERSHEY_SIMPLEX ,1,(255,0,0),2)
    cv2.imshow('Augmented Feed', augmented_frame)

    img_counter += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
client_socket.close()