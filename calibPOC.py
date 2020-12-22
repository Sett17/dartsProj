import cv2 as cv
import math
import numpy as np

frame = cv.imread("test.png")
resFrame = frame
frame = cv.resize(frame, (1000, 667))
resFrame = cv.resize(resFrame, (1000, 667))

p1 = (0, 0)
p2 = (0, 0)
p3 = (0, 0)
p4 = (0, 0)
mid = (0, 0)

mouse = (100, 100)

step = 0

zoomSize = 400
resSize = 800


def intersectLines(pt1, pt2, ptA, ptB):
    """ this returns the intersection of Line(pt1,pt2) and Line(ptA,ptB)

        returns a tuple: (xi, yi, valid, r, s), where
        (xi, yi) is the intersection
        r is the scalar multiple such that (xi,yi) = pt1 + r*(pt2-pt1)
        s is the scalar multiple such that (xi,yi) = pt1 + s*(ptB-ptA)
            valid == 0 if there are 0 or inf. intersections (invalid)
            valid == 1 if it has a unique intersection ON the segment    """

    DET_TOLERANCE = 0.00000001

    # the first line is pt1 + r*(pt2-pt1)
    # in component form:
    x1, y1 = pt1
    x2, y2 = pt2
    dx1 = x2 - x1
    dy1 = y2 - y1

    # the second line is ptA + s*(ptB-ptA)
    x, y = ptA
    xB, yB = ptB
    dx = xB - x
    dy = yB - y

    DET = (-dx1 * dy + dy1 * dx)

    if math.fabs(DET) < DET_TOLERANCE:
        return 0, 0

    # now, the determinant should be OK
    DETinv = 1.0 / DET

    # find the scalar amount along the "self" segment
    r = DETinv * (-dy * (x - x1) + dx * (y - y1))

    # find the scalar amount along the input line
    s = DETinv * (-dy1 * (x - x1) + dx1 * (y - y1))

    # return the average of the two descriptions
    x = (x1 + r * dx1 + x + s * dx) / 2.0
    y = (y1 + r * dy1 + y + s * dy) / 2.0
    return int(x), int(y)


def conv2cvPoints(li):
    points = np.array(li)
    points = np.float32(points[:, np.newaxis, :])
    return points


def click(ev, x, y, flags, param):
    global step, p1, p2, p3, p4, mouse
    mouse = (x, y)
    if ev == cv.EVENT_LBUTTONUP:
        if step == 0:
            p1 = (x, y)
            cv.circle(frame, p1, 3, (255, 0, 0), -1)
            step += 1
        elif step == 1:
            p2 = (x, y)
            cv.circle(frame, p2, 3, (255, 0, 0), -1)
            cv.line(frame, p1, p2, (255, 0, 0), 2)
            step += 1
        elif step == 2:
            p3 = (x, y)
            cv.circle(frame, p3, 3, (255, 0, 0), -1)
            step += 1
        elif step == 3:
            p4 = (x, y)
            cv.circle(frame, p4, 3, (255, 0, 0), -1)
            cv.line(frame, p3, p4, (255, 0, 0), 2)
            mid = intersectLines(p1, p2, p3, p4)
            cv.circle(frame, mid, 5, (0, 255, 0), -1)
            step += 1
        elif step == 4:
            transform()
            step += 1


def transform():
    global resFrame
    h, _ = cv.findHomography(
        conv2cvPoints([p1, p2, p3, p4]), conv2cvPoints([(resSize/2, 100), (resSize/2, resSize-100), (100, resSize/2), (resSize-100, resSize/2)]))

    print(h)

    resFrame = cv.warpPerspective(resFrame, h, (resSize, resSize))

    R = cv.getRotationMatrix2D((resSize/2, resSize/2), -9, 1)
    resFrame = cv.warpAffine(resFrame, R, (resSize, resSize))


cv.namedWindow('frame')
cv.setMouseCallback('frame', click)
while 1:
    if step > 4:
        cv.imshow('frame', resFrame)
    else:
        cv.imshow('frame', frame)
    if step < 4:
        if mouse[1]-50 > 0 and mouse[0]-50 > 0 and mouse[1]+50 < frame.shape[0] and mouse[0]+50 < frame.shape[1]:
            zoom = frame[mouse[1]-50:mouse[1]+50, mouse[0]-50:mouse[0]+50]
            zoom = cv.resize(zoom, (zoomSize, zoomSize))
            cv.line(zoom, (int(zoomSize/2), int(zoomSize/2)-50),
                    (int(zoomSize/2), int(zoomSize/2)+50), (255, 0, 255), 4)
            cv.line(zoom, (int(zoomSize/2)-50, int(zoomSize/2)),
                    (int(zoomSize/2)+50, int(zoomSize/2)), (255, 0, 255), 4)
            cv.imshow('zoom', zoom)
    else:
        cv.destroyWindow('zoom')
    k = cv.waitKey(20) & 0xFF
    if k == ord('q'):
        break

cv.destroyAllWindows()
