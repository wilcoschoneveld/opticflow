import cv2
import tensorflow as tf
import numpy as np

from model import CNN
from tools.fastlk import FastLK

# cap = cv2.VideoCapture('/home/wilco/Documents/video.h264')
cap = cv2.VideoCapture(0)


cnn = CNN(split=False, normalize=True)
fastlk = FastLK(40, True)

sess = tf.Session(graph=cnn.graph)
cnn.saver.restore(sess, '.logs/floyd/small/longrun-saved/step1000.ckpt')

input_pair = np.zeros((1, 64, 64, 2))

image0 = None
image1 = None

while True:
    ret, frame = cap.read()

    if not ret:
        break

    small = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
    image1 = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    if image0 is None:
        image0 = image1
        continue

    input_pair[0, :, :, 0] = image0
    input_pair[0, :, :, 1] = image1

    flow = sess.run(cnn.output, feed_dict={cnn.batch_input: input_pair})[0]
    flow2 = fastlk.predict(image0, image1)

    image0 = image1

    big = cv2.resize(image1, (640, 640), interpolation=cv2.INTER_NEAREST)
    big = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
    cv2.arrowedLine(big, (300, 320), (300 - int(flow[1] * 10), 320 - int(flow[0] * 10)), (255, 0, 0), 2)
    cv2.arrowedLine(big, (340, 320), (340 - int(flow2[1] * 10), 320 - int(flow2[0] * 10)), (0, 0, 255), 2)

    cv2.imshow('video', big)

    k = cv2.waitKey(10) & 0xff

    if k == ord('s'):
        cv2.waitKey(0)

    if k == ord('q'):
        break

cap.release()
