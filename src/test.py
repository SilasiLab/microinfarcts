import os
import cv2
raw_dir = "/mnt/4T/brain_raw_imgs/test/69032-2-images/raw"
img_list = [cv2.cvtColor(cv2.imread(os.path.join(raw_dir, img_dir)), cv2.COLOR_BGR2GRAY) for img_dir in os.listdir(raw_dir)]
frame_width, frame_height = img_list[0].shape[0], img_list[0].shape[1]

for img in img_list:
    img = cv2.resize(img, (int(frame_height * 0.5), int(frame_width * 0.5)))
    segmented_frame = segmentizer.fit_and_predict(img)
    cv2.imshow("ori", img)
    cv2.imshow("seg", segmented_frame)
    cv2.waitKey()
    


rect = (161,79,150,150)