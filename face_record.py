import cv2
from centerface import CenterFace


save_path = 'data/normal/'
count_im = 0

def camera(count_im):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    h, w = frame.shape[:2]
    centerface = CenterFace()
    global org_image
    while True:
        ret, frame = cap.read()
        org_frame = frame.copy()
        croped_face = frame.copy()
        dets, lms = centerface(frame, h, w, threshold=0.35)
        org_image = frame.copy()
        for det in dets:
            boxes, score = det[:4], det[4]
            cv2.rectangle(frame, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (2, 255, 0), 1)
            croped_face = org_frame[int(boxes[1]):int(boxes[3]), int(boxes[0]):int(boxes[2])]

        cv2.imshow('out', frame)
        cv2.imshow('face', croped_face)
        cv2.imwrite(save_path + 'img_{}.jpg'.format(count_im), croped_face)
        count_im += 1
        # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

if __name__ == '__main__':
    camera(count_im)