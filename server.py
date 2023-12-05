# Import the required modules

import socket
import cv2
import pickle
import struct
import AugumentationModule as aug

def SendFeed(frame):
    feed =augmentation.Augument(frame, jewel_img)

    # Encode the augmented frame
    _, encoded_frame = cv2.imencode(".jpg", feed)
    frame_data = pickle.dumps(encoded_frame, protocol=0)

    # Send the size of the frame data
    size = struct.pack(">L", len(frame_data))
    conn.sendall(size)

    # Send the frame data
    conn.sendall(frame_data)



HOST = ''
PORT = 8485
augmentation = aug.NecklaceAugumentation()

s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
print('Socket created')
s.bind((HOST,PORT))
print('Socket bind complete')
s.listen(10)
print('Socket now listening')

conn,addr=s.accept()

data = b""
payload_size = struct.calcsize(">L")

print("payload_size: {}".format(payload_size))

jewel_img = cv2.imread("Images/alok2.png")

while True:
    while len(data) < payload_size:
        data += conn.recv(4096)
        if not data:
            cv2.destroyAllWindows()
            conn, addr = s.accept()
            continue
    # receive image row data form client socket
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack(">L", packed_msg_size)[0]
    while len(data) < msg_size:
        data += conn.recv(4096)
    frame_data = data[:msg_size]
    data = data[msg_size:]
    # unpack image using pickle 
    frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    SendFeed(frame)

    cv2.waitKey(1)

