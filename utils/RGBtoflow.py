import cv2 as cv
import numpy as np
import os
from tqdm import tqdm


root_frame = ''

save_flow = ''


if not os.path.exists(save_flow):
    os.makedirs(save_flow)


for cur_dir in tqdm(os.listdir(root_frame)):
    cur_rgb_dir = os.path.join(root_frame, cur_dir)

    if not os.path.isdir(cur_rgb_dir):
        continue

    pre = None


    for cur_frame in sorted(os.listdir(cur_rgb_dir)):
        cur_frame_path = os.path.join(cur_rgb_dir, cur_frame)
        next_frame = cv.imread(cur_frame_path)

        if pre is not None:
            hsv = np.zeros_like(next_frame)
            hsv[..., 1] = 255


            con_pre = cv.cvtColor(pre, cv.COLOR_BGR2GRAY)
            con_next = cv.cvtColor(next_frame, cv.COLOR_BGR2GRAY)

            if con_pre.shape == con_next.shape:

                flow = cv.calcOpticalFlowFarneback(con_pre, con_next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])


                hsv[..., 0] = ang * 180 / np.pi / 2
                hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)


                bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)


                cur_save_path = os.path.join(save_flow, cur_dir)
                if not os.path.exists(cur_save_path):
                    os.mkdir(cur_save_path)


                frame_name = os.path.splitext(cur_frame)[0] + '.jpg'


                cv.imwrite(os.path.join(cur_save_path, frame_name), bgr)


        pre = next_frame
