import cv2
import tensorflow as tf
import numpy as np

from model import CNN
from tools.fastlk import FastLK

# cap = cv2.VideoCapture('/home/wilco/Documents/internship/video.h264')
# cap = cv2.VideoCapture('/home/wilco/Videos/20170804_095138.mp4')
cap = cv2.VideoCapture(0)


# cnn = CNN(split=False, normalize=True)
# sess = tf.Session(graph=cnn.graph)
# cnn.saver.restore(sess, '.logs/floyd/small/longrun-saved/step80000.ckpt')

cnn = CNN(split=True, normalize=True, fully_connected=500)
sess = tf.Session(graph=cnn.graph)
cnn.saver.restore(sess, '.logs/output/step2000.ckpt')

fastlk = FastLK(40, True)

input_pair = np.zeros((1, 64, 64, 2))

image0 = None
image1 = None

# dust = np.random.random((640, 640))
dust = cv2.imread('/home/wilco/Pictures/cloud.png', cv2.IMREAD_GRAYSCALE)
dx = 0
dy = 0

cv2.namedWindow('video')
cv2.createTrackbar('arrowscale', 'video', 20, 100, lambda x: None)

while True:
    ret, frame = cap.read()
    frame = frame[:800,:800]
    frame = cv2.flip(frame, 1)

    if not ret:
        break

    small = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
    image1 = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    if image0 is None:
        image0 = image1
        continue

    input_pair[0, :, :, 0] = image0
    input_pair[0, :, :, 1] = image1

    acc, flow = sess.run([cnn.accuracy, cnn.output], feed_dict={cnn.batch_input: input_pair})
    flow = flow[0]
    flow2 = fastlk.predict(image0, image1)

    print(acc)

    dx = (dx - flow[0]) % 64
    dy = (dy - flow[1]) % 64
    dust_rolled = np.roll(dust, (int(dx*10), int(dy*10)), (0, 1))

    image0 = image1

    big = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    big = cv2.resize(big, (640, 640), interpolation=cv2.INTER_NEAREST)
    big[dust_rolled > 200] = [0, 255, 0]

    scale = cv2.getTrackbarPos('arrowscale', 'video')
    cv2.arrowedLine(big, (300, 320), (300 - int(flow[1] * scale), 320 - int(flow[0] * scale)), (255, 0, 0), 2)
    cv2.arrowedLine(big, (340, 320), (340 - int(flow2[1] * scale), 320 - int(flow2[0] * scale)), (0, 0, 255), 2)

    cv2.imshow('video', big)

    k = cv2.waitKey(10) & 0xff

    if k == ord('s'):
        cv2.waitKey(0)

    if k == ord('q'):
        break

cap.release()
