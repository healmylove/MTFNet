import os
import cv2


def extract_frames(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        if os.path.isfile(file_path) and filename.endswith(('.mp4', '.avi', '.mov')):
            cap = cv2.VideoCapture(file_path)
            success, image = cap.read()
            count = 0

            while success:
                cv2.imwrite(f"{output_folder}/frame{count}.jpg", image)
                success, image = cap.read()
                print(f"frame {count} saved")
                count += 1

            print("提取完成！")

            # if ret:
            #     output_file = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_frame.jpg")
            #     cv2.imwrite(output_file, frame)
            # cap.release()


if __name__ == "__main__":
    input_folder = '/inputpath'
    output_folder = '/outputpath'
    extract_frames(input_folder, output_folder)