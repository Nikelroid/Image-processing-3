import math
import time

import cv2
import numpy as np


def normalize(img):
    # linear normalizing img between 0 to 255
    return ((img - img.min()) * (255 / (img.max() - img.min()))).astype('uint8')


def clip(img):
    # linear normalizing img between 0 to 255
    return img.clip(0, 255)


def resize(img, zarib):
    w = int(img.shape[1] * zarib)
    h = int(img.shape[0] * zarib)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def acc_maker(canny):
    w = int(canny.shape[1])
    h = int(canny.shape[0])
    ac_h = int(math.sqrt(w ** 2 + h ** 2))
    accumulator = np.zeros(((ac_h * 2), ind))
    for i in range(h):
        if np.max(canny[i, :]) == 0:
            continue
        for j in range(w):
            if canny[i, j] != 0:
                for tetha in range(ind):
                    r = j * math.cos(np.pi * tetha / ind) + i * math.sin(np.pi * tetha / ind)
                    accumulator[int(np.round(r)) + ac_h, tetha] += 1
        print('accumulator making:', i, '/', h)
    return accumulator


def transform(r, t, h):
    r0 = r - h / 2
    theta0 = t * np.pi / ind
    y0 = r0 * math.cos(theta0)
    x0 = r0 * math.sin(theta0)
    a = math.tan(-theta0)
    b = y0 - (a * x0)
    return a, b


def findchess(hough, img, chessline, order):
    w = int(hough.shape[1])
    h = int(hough.shape[0])
    difs = []
    start = False
    save1 = (0, 0)
    save2 = (0, 0)
    if order == 0:
        limit = (0, h, 1)
    else:
        limit = (h - 1, -1, -1)
    for r in range(limit[0], limit[1], limit[2]):
        for theta in range(w):
            if hough[r, theta] != 0:
                a, b = transform(r, theta, h)
                if save2[1] != 0:
                    if start:
                        av = np.abs(np.average(np.array(difs)))
                        a1 = np.abs(a - save1[0])
                        a2 = np.abs(save2[0] - save1[0])
                        if av + math.sqrt(av) * 2 >= np.abs(save1[1] - b) >= av - math.sqrt(av) * 2 and \
                                (a1 + math.sqrt(a1) * 2 >= a2 >= a1 - math.sqrt(a1) * 2 or
                                 a2 + math.sqrt(a2) * 2 > a1 > a2 - math.sqrt(a2) * 2):
                            chessline.append((a, b))
                            draw(a, b, img)
                            difs.append(np.abs(save1[1] - b))
                        else:
                            return img
                    else:
                        b1 = np.abs(b - save1[1])
                        b2 = np.abs(save2[1] - save1[1])
                        a1 = np.abs(a - save1[0])
                        a2 = np.abs(save2[0] - save1[0])
                        if b1 + math.sqrt(b1) * 2 >= b2 >= b1 - math.sqrt(b1) * 2 and \
                                (a1 + math.sqrt(a1) * 2 >= a2 >= a1 - math.sqrt(a1) * 2 or
                                 a2 + math.sqrt(a2) * 2 > a1 > a2 - math.sqrt(a2) * 2):
                            start = True
                            difs = []
                            difs.append(np.abs(save1[1] - b))
                            difs.append(np.abs(save2[1] - save1[1]))
                            chessline.append((a, b))
                            draw(a, b, img)
                            chessline.append((save1[0], save1[1]))
                            draw(save1[0], save1[1], img)
                            chessline.append((save2[0], save2[1]))
                            draw(save2[0], save2[1], img)
                save2 = save1
                save1 = (a, b)
        print('find chess:', r, '/', h)
    return img


def drawline(hough, img):
    w = int(hough.shape[1])
    h = int(hough.shape[0])
    for r in range(h):
        for theta in range(w):
            if hough[r, theta] != 0:
                a, b = transform(r, theta, h)
                draw(a, b, img)
        print('draw line:', r, '/', h)
    return img


def filter1(hough):
    w = int(hough.shape[1])
    h = int(hough.shape[0])
    for i in range(6):
        size = int(2 + 1.7 ** i)
        for y in range(h - size):
            if np.max(hough[y:y + size, :]) == 0:
                y += size
                continue
            for x in range(w - size):
                if np.max(hough[y:y + size, x:x + size]) == 0:
                    continue
                t, val, t, loc = cv2.minMaxLoc(hough[y:y + size, x:x + size])
                hough[y:y + size, x:x + size] = np.zeros((size, size))
                hough[y + loc[1], x + loc[0]] = val
            print('filter 1:', size, '-', y, '/', h)
    return hough


def filter2(hough):
    w = int(hough.shape[1])
    h = int(hough.shape[0])
    size = int(ind / 12)
    for y in range(h - size):
        if np.max(hough[y:y + size, :]) == 0:
            y += size
            continue
        for x in range(w - size):
            if np.max(hough[y:y + size, x:x + size]) == 0:
                continue
            points = np.transpose(np.nonzero(hough[y:y + size, x:x + size]))
            if len(points) == 2:
                if points[0][1] + int(math.sqrt(ind) / 8) > points[1][1] > points[0][1] - int(math.sqrt(ind) / 8):
                    continue
                t, val, t, loc = cv2.minMaxLoc(hough[y:y + size, x:x + size])
                hough[y:y + size, x:x + size] = np.zeros((size, size))
                hough[y + loc[1], x + loc[0]] = val
        print('filter 2:', size, '-', y, '/', h)
    return hough


def findcorners(chesslines, img):
    w = int(img.shape[1])
    for x in range(w):
        ys = []
        for line in chesslines:
            y = (line[0] * x) + line[1]
            if w >= y >= 0:
                for i in ys:
                    if i + w / 700 >= y >= i - w / 700:
                        cv2.circle(img, (int(y), x), 2, (0, 0, 255), 2)
                ys.append(y)
        print(x, '/', w)
    return img


def draw(a, b, img):
    wimg = img.shape[1]
    point1, point2 = (int(np.round(b)), 0), (int(np.round(a * wimg + b)), wimg)
    cv2.line(img, point1, point2, (255, 0, 0), 2)


def q1(img):
    canny = cv2.Canny(img, 510, 510)  # save edges of image by canny in canny matrix

    cv2.imwrite('res1.jpg', canny)  # save canny matrix as a picture
    # cv2.imwrite('res2.jpg', canny)

    acc = acc_maker(canny)  # call acc_maker function to generate and save accumulator in acc

    cv2.imwrite('res03-hough-space.jpg', acc)  # save accumulator matrix as a picture
    # cv2.imwrite('res04-hough-space.jpg', acc)

    acc[acc <= 100] = 0  # make under 100 values of acc zero
    acc = filter1(acc)  # operate first filter
    acc = filter2(acc)  # operate second filter

    res56 = np.copy(img)  # save a copy of img in res56
    res56 = drawline(acc, res56)  # fiding lines in findline function
    cv2.imwrite('res05-lines.jpg', res56)  # store res56 that lines drew on it
    # cv2.imwrite('res06-lines.jpg', res56)

    chesslines = []  # define chess line for store line of chess
    img = findchess(acc, img, chesslines, 0)  # find chess lines horizontally
    img = findchess(acc, img, chesslines, 1)  # find chess line vertically
    cv2.imwrite('res07-chess.jpg', img)  # store img that chess lines only drew on it
    # cv2.imwrite('res08-chess.jpg', img)

    img = findcorners(chesslines, img)  # find corners of chess ground and show them
    cv2.imwrite('res09-corners.jpg', img)  # store img that chess lines only and corners drew on it
    # cv2.imwrite('res10-corners.jpg', img)


t0 = time.time()

im = cv2.imread('im02.jpg', 1)  # load image
ind = 600  # scale for theta
q1(im)  # go to main function

t1 = time.time()
print('runtime: ' + str(int(t1 - t0)) + ' seconds')
