import numpy as np
import dlib
import cv2


def get_landmarks(img, predictor):
    """get landmark with dlib
    :return: np.array shape=(68, 2)
    """
    detector = dlib.get_frontal_face_detector()

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dets = detector(rgb, 1)

    if len(dets) == 0:
        return None

    # choose the biggest bbox
    dets = {d:d.area() for d in dets}
    bbox = max(dets, key=dets.get)

    shape = predictor(img, bbox)

    t = list(shape.parts())
    a = []
    for tt in t:
        a.append([tt.x, tt.y])
    lm = np.array(a)
    return lm

def draw_face_landmarks(img, pts, color=(255,255,255), thickness=5, with_lines=True, with_circle=False):
    pts = pts.astype(np.int32)

    if with_lines:
        indices = [(range(17), False), (range(48, 60), True), (range(60, 65), False), ([64,65,66,67,60], False),
                   (range(27, 31), False), (range(36, 42), True), (range(42, 48), True),
                   (range(17, 22), False), (range(22, 27), False), (range(31, 36), False)]

        for idx, isClosed in indices:
            img = cv2.polylines(img, [pts[idx]], isClosed, color, thickness)

    if with_circle:
        for pt in pts:
            img = cv2.circle(img, (int(pt[0]), int(pt[1])), 2, (255,0,0), -1, cv2.LINE_AA)

    return img