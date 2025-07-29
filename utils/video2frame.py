import os
import json
import math


import cv2
# from tqdm import tqdm 

video_path = ''
save_dir = ''

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

def cap(single_video_path, save_dir, sample_rate=5):
    cap = cv2.VideoCapture(single_video_path)

    count = 0
    frameRate = cap.get(5)
    while(cap.isOpened()):
        frameId = cap.get(1)


        ret, frame = cap.read()
        if ret != True:
            break
        if (frameId % math.floor(frameRate) == 0):
            filename = save_dir + 'paradigm_frame_{}.jpg'.format(count)
            cv2.imwrite(filename, frame)
            print(frameId)
            count += 1
    cap.release()

#cap(video_path, save_dir)


# Rate = 1
save_dir_Rate_4 = ''
united_width = 171
united_height = 128
if not os.path.exists(save_dir_Rate_4):
    os.mkdir(save_dir_Rate_4)


def process(video_path,save_dir):
    capture = cv2.VideoCapture(video_path)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
    print(frame_count)   
    EXTRACT_FREQUENCY = 1
    if frame_count // EXTRACT_FREQUENCY <= 16:
        EXTRACT_FREQUENCY -= 1
        if frame_count // EXTRACT_FREQUENCY <= 16:
            EXTRACT_FREQUENCY -= 1
            if frame_count // EXTRACT_FREQUENCY <= 16:
                EXTRACT_FREQUENCY -= 1
    print(EXTRACT_FREQUENCY)
    count = 0
    retaining = True
    while (count < frame_count and retaining):
        retaining, frame = capture.read()
        if frame is None:
            continue
        if count % EXTRACT_FREQUENCY == 0:
            if (frame_height != united_height) or (frame_width != united_width):
                frame = cv2.resize(frame, (united_width, united_height))
            cv2.imwrite(filename=os.path.join(save_dir + 'paradigm_frame_{}.jpg'.format(count)), img=frame)
        count += 1

process(video_path, save_dir)




