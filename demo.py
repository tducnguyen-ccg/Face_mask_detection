import cv2
import scipy.io as sio
import os
from centerface import CenterFace
import matplotlib.pylab as plt
import numpy as np

global org_image

def calculate_avg_color(image, landmarks):
    global org_image
    avg_val = 0
    pixel_counter = 0
    crop_img = []
    # Extract regiopn
    if len(landmarks) == 2:
        # eye area
        comp = int(np.abs(landmarks[0][0] - landmarks[1][0]) / 5)
        x1 = min(landmarks[0][0], landmarks[1][0])
        y1 = min(landmarks[0][1], landmarks[1][1])
        x2 = max(landmarks[0][0], landmarks[1][0])
        y2 = max(landmarks[0][1], landmarks[1][1])
        n_x = int((x2 + x1) / 2)
        n_y = int((y2 + y1) / 2) - (comp * 4)
        # cv2.circle(org_image, (int(x1), int(y1)), 2, (0, 0, 255), -1)
        crop_img = image[n_y - comp:n_y + comp, n_x - comp: n_x + comp]
        avg_val = np.average(crop_img)
        # crop_img = image[y1 - comp:y2, x1:x2]
    else:
        # mouth area
        comp = int(np.abs(landmarks[1][0] - landmarks[2][0]) / 3)
        comp_s = int(comp/3)

        p1_x = min(landmarks[1][0], landmarks[0][0]) - comp
        p1_y = int((landmarks[1][1] + landmarks[0][1]) / 2)

        p2_x = max(landmarks[2][0], landmarks[0][0]) + comp
        p2_y = int((landmarks[2][1] + landmarks[0][1]) / 2)

        crop_img_1 = image[p1_y-comp_s:p1_y+comp_s, p1_x-comp_s:p1_x+comp_s]
        avg_val_1 = np.average(crop_img_1)

        crop_img_2 = image[p2_y - comp_s:p2_y + comp_s, p2_x - comp_s:p2_x + comp_s]
        avg_val_2 = np.average(crop_img_2)

        avg_val = (avg_val_1 + avg_val_2) / 2

    return avg_val


def camera():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    h, w = frame.shape[:2]
    centerface = CenterFace()
    global org_image
    while True:
        ret, frame = cap.read()
        dets, lms = centerface(frame, h, w, threshold=0.35)
        org_image = frame.copy()
        hsvframe = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:,:,0]
        for det in dets:
            boxes, score = det[:4], det[4]
            cv2.rectangle(frame, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (2, 255, 0), 1)
        for lm in lms:
            face_points = []
            for i in range(0, 5):
                cv2.circle(frame, (int(lm[i * 2]), int(lm[i * 2 + 1])), 2, (0, 0, 255), -1)
                face_points.append([int(lm[i * 2]), int(lm[i * 2 + 1])])

            # Extract color of region
            avg_top = calculate_avg_color(hsvframe, face_points[0:2])
            avg_bot = calculate_avg_color(hsvframe, face_points[2:5])
            if avg_top and avg_bot:
                diff = np.abs(avg_top - avg_bot) / max(avg_top, avg_bot) * 100
                print(diff)
                if diff <= 30:
                    cv2.putText(frame, "Vui long mang khau trang ", (int(boxes[0]), int(boxes[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

        cv2.imshow('out', frame)
        # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()


def test_image():
    frame = cv2.imread('000388.jpg')
    h, w = frame.shape[:2]
    landmarks = True
    centerface = CenterFace(landmarks=landmarks)
    if landmarks:
        dets, lms = centerface(frame, h, w, threshold=0.35)
    else:
        dets = centerface(frame, threshold=0.35)

    for det in dets:
        boxes, score = det[:4], det[4]
        cv2.rectangle(frame, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (2, 255, 0), 1)
    if landmarks:
        for lm in lms:
            for i in range(0, 5):
                cv2.circle(frame, (int(lm[i * 2]), int(lm[i * 2 + 1])), 2, (0, 0, 255), -1)
    cv2.imshow('out', frame)
    cv2.waitKey(0)


def test_image_tensorrt():
    frame = cv2.imread('000388.jpg')
    h, w = 480, 640  # must be 480* 640
    landmarks = True
    centerface = CenterFace(landmarks=landmarks, backend="tensorrt")
    if landmarks:
        dets, lms = centerface(frame, h, w, threshold=0.35)
    else:
        dets = centerface(frame, threshold=0.35)

    for det in dets:
        boxes, score = det[:4], det[4]
        cv2.rectangle(frame, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (2, 255, 0), 1)
    if landmarks:
        for lm in lms:
            for i in range(0, 5):
                cv2.circle(frame, (int(lm[i * 2]), int(lm[i * 2 + 1])), 2, (0, 0, 255), -1)
    cv2.imshow('out', frame)
    cv2.waitKey(0)


def test_widerface():
    Path = 'widerface/WIDER_val/images/'
    wider_face_mat = sio.loadmat('widerface/wider_face_split/wider_face_val.mat')
    event_list = wider_face_mat['event_list']
    file_list = wider_face_mat['file_list']
    save_path = 'save_out/'

    for index, event in enumerate(event_list):
        file_list_item = file_list[index][0]
        im_dir = event[0][0]
        # print(save_path + im_dir)
        if not os.path.exists(save_path + im_dir):
            os.makedirs(save_path + im_dir)
        landmarks = True
        centerface = CenterFace(landmarks=landmarks)
        for num, file in enumerate(file_list_item):
            im_name = file[0][0]
            zip_name = '%s/%s.jpg' % (im_dir, im_name)
            print(os.path.join(Path, zip_name))
            img = cv2.imread(os.path.join(Path, zip_name))
            h, w = img.shape[:2]
            if landmarks:
                dets, lms = centerface(img, h, w, threshold=0.05)
            else:
                dets = centerface(img, threshold=0.05)
            f = open(save_path + im_dir + '/' + im_name + '.txt', 'w')
            f.write('{:s}\n'.format('%s/%s.jpg' % (im_dir, im_name)))
            f.write('{:d}\n'.format(len(dets)))
            for b in dets:
                x1, y1, x2, y2, s = b
                f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.format(x1, y1, (x2 - x1 + 1), (y2 - y1 + 1), s))
            f.close()
            print('event:%d num:%d' % (index + 1, num + 1))


if __name__ == '__main__':
    camera()
    # test_image()
    # test_widerface()
