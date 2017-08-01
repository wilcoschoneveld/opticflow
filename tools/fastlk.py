import cv2
import numpy as np

from tools.math import geometric_median


class FastLK(object):

    def __init__(self, threshold=40, adaptive=True):
        self.adaptive = adaptive
        self.fast = cv2.FastFeatureDetector_create(threshold=threshold)

    def batch_predict(self, batch_input):
        size = batch_input.shape[0]
        prediction = np.empty((size, 2))

        for i in range(size):
            image0 = batch_input[i, :, :, 0].astype(np.uint8)
            image1 = batch_input[i, :, :, 1].astype(np.uint8)

            prediction[i] = self.predict(image0, image1)

        return prediction

    def predict(self, image0, image1):

        if self.adaptive:
            for threshold in np.arange(200, 5, -5):
                self.fast.setThreshold(threshold)

                kp = self.fast.detect(image0, None)

                if len(kp) > 40:
                    break

        kp0 = cv2.KeyPoint_convert(kp)

        if not len(kp):
            return [0, 0]

        image0_kp = cv2.drawKeypoints(image0, kp, None, (255, 0, 0))

        kp1, st, err = cv2.calcOpticalFlowPyrLK(image0, image1, kp0, None, winSize=(10, 10), maxLevel=0)

        flows = np.fliplr(np.array(kp0) - np.array(kp1))

        mag = np.square(flows[:, 0]) + np.square(flows[:, 1])

        flows = flows[mag.argsort()]

        median = len(kp0) // 2

        lk_flow = np.mean(flows[median - 1:median + 1], axis=0)
        # lk_flow = flows[median]
        # lk_flow = np.mean(flows, axis=0)
        # lk_flow = geometric_median(flows)

        return lk_flow

        # print('found %i points' % len(kp0))
        # print('real flow:', flow)
        # print('lk flow:', lk_flow)
        # print('squared error:', np.mean(np.square(flow - lk_flow)))
