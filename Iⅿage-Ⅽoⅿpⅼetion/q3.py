import time

import cv2
import numpy as np
import random as rnd


def bcut_vertical(check1, check2):
    image_boundry = (check1 - check2) ** 2
    image_boundry = np.vstack(((np.zeros((1, 50))), image_boundry))

    cost = 0
    list = []
    bestcost = 255 * 2 * 150
    for start in range(15, 36):
        image_costs = np.zeros_like(image_boundry) + max
        image_costs[0, start] = 1
        for row in range(1, 150 + 1):

            for clmn in range(10, 39):
                if clmn < 10:
                    cost = np.min(image_costs[row - 1, clmn:clmn + 2]) + image_boundry[row, clmn]
                elif clmn > 39:
                    cost = np.min(image_costs[row - 1, clmn - 1:clmn + 1]) + image_boundry[row, clmn]
                else:
                    cost = np.min(image_costs[row - 1, clmn - 1:clmn + 2]) + image_boundry[row, clmn]
                image_costs[row, clmn] = cost
            if row == 150:
                totalcost = int(np.min(image_costs[150, :]))

        if bestcost > totalcost:
            bestcost = totalcost
            best_cost = image_costs

    for row in range(149, -1, -1):
        min = max
        minx = 0
        if row == 149:
            for clmn in range(10, 39):
                cost = best_cost[row, clmn]
                if cost < min:
                    min = cost
                    minx = clmn
            list = []
            save_cost = cost
            list.insert(0, minx)
        else:
            for clmn in range(-1, 2):
                if clmn + list[0] < 10 or clmn + list[0] > 39:
                    cost = max
                else:
                    cost = best_cost[row, clmn + list[0]]
                if cost < min:
                    min = cost
                    minx = clmn + list[0]
            list.insert(0, minx)
    return list


def bcut_horizontal(check1, check2):
    image_boundry = (check1 - check2) ** 2
    image_boundry = np.hstack(((np.zeros((50, 1))), image_boundry))
    cost = 0
    save_cost = 0
    list = []
    bestcost = 255 * 2 * 150
    for start in range(10, 39):
        image_costs = np.zeros_like(image_boundry) + max
        image_costs[start, 0] = 1
        for row in range(1, 151):

            for clmn in range(10, 39):
                if clmn < 10:
                    cost = np.min(image_costs[clmn:clmn + 2, row - 1, ]) + image_boundry[clmn, row]
                elif clmn > 39:
                    cost = np.min(image_costs[clmn - 1:clmn + 1, row - 1]) + image_boundry[clmn, row]
                else:
                    cost = np.min(image_costs[clmn - 1:clmn + 2, row - 1]) + image_boundry[clmn, row]
                image_costs[clmn, row] = cost
            if row == 150:
                totalcost = int(np.min(image_costs[:, 150]))

        if bestcost > totalcost:
            bestcost = totalcost
            best_cost = image_costs

    for row in range(149, -1, -1):
        min = max
        minx = 0
        if row == 149:
            for clmn in range(10, 39):
                cost = best_cost[clmn, row]

                if cost < min:
                    min = cost
                    minx = clmn
            list = []
            save_cost = cost
            list.insert(0, minx)
        else:
            for clmn in range(-1, 2):
                if clmn + list[0] < 10 or clmn + list[0] > 39:
                    cost = max
                else:
                    cost = best_cost[clmn + list[0], row]
                if cost < min:
                    min = cost
                    minx = clmn + list[0]
            list.insert(0, minx)
    return list


def bcut_v(h_patch, check1, check2):
    image_boundry = (check1 - check2) ** 2
    cost = 0
    list = []
    image_costs = np.zeros_like(image_boundry) + max
    t, t, loc, t = cv2.minMaxLoc(image_boundry[99, 10:39])
    image_costs[99, 10 + loc[1]] = 1
    for row in range(h_patch - 2, -1, -1):
        for clmn in range(10, 39):
            if clmn < 10:
                cost = np.min(image_costs[row + 1, clmn:clmn + 2]) + image_boundry[row, clmn]
            elif clmn > 39:
                cost = np.min(image_costs[row + 1, clmn - 1:clmn + 1]) + image_boundry[row, clmn]
            else:
                cost = np.min(image_costs[row + 1, clmn - 1:clmn + 2]) + image_boundry[row, clmn]
            image_costs[row, clmn] = cost

    for row in range(h_patch):
        min = max
        minx = 0
        if row == 0:
            for clmn in range(10, 39):
                cost = image_costs[0, clmn]
                if cost < min:
                    min = cost
                    minx = clmn
            list = []
            save_cost = cost
            list.append(minx)
        else:
            for clmn in range(-1, 2):
                if clmn + list[len(list) - 1] < 10 or clmn + list[len(list) - 1] > 39:
                    cost = max
                else:
                    cost = image_costs[row, clmn + list[len(list) - 1]]
                if cost < min:
                    min = cost
                    minx = clmn + list[len(list) - 1]
            list.append(minx)
    return list


def bcut_h(h_patch, check1, check2):
    image_boundry = (check1 - check2) ** 2
    cost = 0
    list = []
    image_costs = np.zeros_like(image_boundry) + max
    t, t, loc, t = cv2.minMaxLoc(image_boundry[10:39, 99])
    image_costs[10 + loc[1], 99] = 1

    for row in range(h_patch - 2, -1, -1):
        for clmn in range(10, 39):
            if clmn < 10:
                cost = np.min(image_costs[clmn:clmn + 2, row + 1]) + image_boundry[clmn, row]
            elif clmn > 39:
                cost = np.min(image_costs[clmn - 1:clmn + 1, row + 1]) + image_boundry[clmn, row]
            else:
                cost = np.min(image_costs[clmn - 1:clmn + 2, row + 1]) + image_boundry[clmn, row]
            image_costs[clmn, row] = cost

    for row in range(h_patch):
        min = max
        minx = 0
        if row == 0:
            for clmn in range(10, 39):
                cost = image_costs[clmn, 0]
                if cost < min:
                    min = cost
                    minx = clmn
            list = []
            save_cost = cost
            list.append(minx)
        else:
            for clmn in range(-1, 2):
                if clmn + list[len(list) - 1] < 10 or clmn + list[len(list) - 1] > 39:
                    cost = max
                else:
                    cost = image_costs[clmn + list[len(list) - 1], row]
                if cost < min:
                    min = cost
                    minx = clmn + list[len(list) - 1]
            list.append(minx)
    return list


def main(h_patch, w_area, limit, x, y, points):
    picb, picg, picr = cv2.split(image)

    matchb1 = np.mean(picb[y:y + 50, x + 50:x + 100])
    matchb2 = np.mean(picb[y + 50:y + 100, x:x + 50])
    matchb3 = np.mean(picb[y:y + 50, x:x + 50])

    matchg1 = np.mean(picg[y:y + 50, x + 50:x + 100])
    matchg2 = np.mean(picg[y + 50:y + 100, x:x + 50])
    matchg3 = np.mean(picg[y:y + 50, x:x + 50])

    matchr1 = np.mean(picr[y:y + 50, x + 50:x + 100])
    matchr2 = np.mean(picr[y + 50:y + 100, x:x + 50])
    matchr3 = np.mean(picr[y:y + 50, x:x + 50])

    matchb = (matchb1 + matchb2 + matchb3) / 3
    matchg = (matchg1 + matchg2 + matchg3) / 3
    matchr = (matchr1 + matchr2 + matchr3) / 3

    pic = np.copy(image[y:y + 100, x:x + 100])
    pic[50:, 50:] = (np.zeros((50, 50, 3), dtype='uint8') + (matchb, matchg, matchr))

    res = cv2.matchTemplate(image, pic, cv2.TM_CCOEFF_NORMED)
    threshold = 1
    while 1:
        destroyed = False
        threshold -= (1 / 200)
        y_coord, x_coord = np.where(res >= threshold)
        if len(x_coord) > limit:
            found = False
            for q in range(limit):
                index = rnd.randrange(0, len(y_coord))
                destroyed = False
                for point in points:
                    if point[0] - 100 <= y_coord[index] <= point[2] and point[1] - 100 <= x_coord[index] <= point[3]:
                        x_coord = np.delete(x_coord, index)
                        y_coord = np.delete(y_coord, index)
                        destroyed = True
                        break
                if not destroyed:
                    coordinates = int(y_coord[index]), int(x_coord[index])
                    found = True
                    break
            if found:
                break
            if destroyed:
                threshold += 9 / 2000
            if threshold == 0:
                threshold = 1

    image[y:y + 100, x:x + 100] = image[coordinates[0]:coordinates[0] + 100,
                                  coordinates[1]:coordinates[1] + 100]

    check1 = cv2.cvtColor(pic[:, :50], cv2.COLOR_BGR2GRAY)
    check2 = cv2.cvtColor(image[y:y + 100, x:x + 50], cv2.COLOR_BGR2GRAY)

    list = bcut_v(h_patch, check1, check2)

    for r in range(len(list)):
        l = list[r]
        ll = 50 - l
        for matte in range(l):
            image[y + r, x + matte] = np.round(
                pic[r, matte] * ((2 * l - matte) / (2 * l)) + image[y + r, x + matte] * (matte / (2 * l)))
        for matte in range(ll):
            image[y + r, x + matte + l] = np.round(
                pic[r, matte + l] * ((ll - matte) / (2 * ll)) + image[y + r, x + matte + l] * ((ll + matte) / (2 * ll)))

    check1 = cv2.cvtColor(pic[:50, :], cv2.COLOR_BGR2GRAY)
    check2 = cv2.cvtColor(image[y:y + 50, x:x + 100], cv2.COLOR_BGR2GRAY)
    list = bcut_h(h_patch, check1, check2)

    for r in range(len(list)):
        l = list[r]
        ll = 50 - l
        for matte in range(l):
            image[y + matte, x + r] = np.round(
                pic[matte, r] * ((2 * l - matte) / (2 * l)) + image[y + matte, x + r] * (matte / (2 * l)))
        for matte in range(ll):
            image[y + matte + l, x + r] = np.round(
                pic[matte + l, r] * ((ll - matte) / (2 * ll)) + image[y + matte + l, x + r] * ((ll + matte) / (2 * ll)))


# ********************************************************************************************************************

def last_row(limit, x, y, points):
    picb, picg, picr = cv2.split(image)

    matchb1 = np.mean(picb[y:y + 50, x + 50:x + 100])
    matchb2 = np.mean(picb[y + 100:y + 150, x + 50:x + 100])
    matchb3 = np.mean(picb[y:y + 150, x:x + 50])

    matchg1 = np.mean(picg[y:y + 50, x + 50:x + 100])
    matchg2 = np.mean(picg[y + 100:y + 150, x + 50:x + 100])
    matchg3 = np.mean(picg[y:y + 150, x:x + 50])

    matchr1 = np.mean(picr[y:y + 50, x + 50:x + 100])
    matchr2 = np.mean(picr[y + 100:y + 150, x + 50:x + 100])
    matchr3 = np.mean(picr[y:y + 150, x:x + 50])

    matchb = (matchb1 + matchb2 + (matchb3 * 3)) / 5
    matchg = (matchg1 + matchg2 + (matchg3 * 3)) / 5
    matchr = (matchr1 + matchr2 + (matchr3 * 3)) / 5

    pic = np.copy(image[y:y + 150, x:x + 100])
    pic[50:100, 50:] = (np.zeros((50, 50, 3), dtype='uint8') + (matchb, matchg, matchr))

    res = cv2.matchTemplate(image, pic, cv2.TM_CCOEFF_NORMED)
    threshold = 1
    while 1:
        destroyed = False
        threshold -= 1 / 200
        y_coord, x_coord = np.where(res >= threshold)
        if len(x_coord) > limit:
            found = False
            for q in range(limit):
                index = rnd.randrange(0, len(y_coord))
                destroyed = False
                for point in points:
                    if point[0] - 150 <= y_coord[index] <= point[2] and point[1] - 100 <= x_coord[index] <= point[3]:
                        x_coord = np.delete(x_coord, index)
                        y_coord = np.delete(y_coord, index)
                        destroyed = True
                        break
                if not destroyed:
                    coordinates = int(y_coord[index]), int(x_coord[index])
                    found = True
                    break
            if found:
                break
            if destroyed:
                threshold += 9 / 2000
            if threshold == 0:
                threshold = 1

    image[y:y + 150, x:x + 100] = image[coordinates[0]:coordinates[0] + 150,
                                  coordinates[1]:coordinates[1] + 100]
    check1 = cv2.cvtColor(pic[:, :50], cv2.COLOR_BGR2GRAY)
    check2 = cv2.cvtColor(image[y:y + 150, x:x + 50], cv2.COLOR_BGR2GRAY)
    list = bcut_vertical(check1, check2)

    for r in range(len(list)):
        l = list[r]
        ll = 50 - l
        for matte in range(l):
            image[y + r, x + matte] = np.round(
                pic[r, matte] * ((2 * l - matte) / (2 * l)) + image[y + r, x + matte] * (matte / (2 * l)))
        for matte in range(ll):
            image[y + r, x + matte + l] = np.round(
                pic[r, matte + l] * ((ll - matte) / (2 * ll)) + image[y + r, x + matte + l] * ((ll + matte) / (2 * ll)))

    # ********************************************************************************************************************

    check1 = cv2.cvtColor(pic[:50, :], cv2.COLOR_BGR2GRAY)
    check2 = cv2.cvtColor(image[y:y + 50, x:x + 100], cv2.COLOR_BGR2GRAY)
    list = bcut_h(100, check1, check2)

    for r in range(len(list)):
        l = list[r]
        ll = 50 - l
        for matte in range(l):
            image[y + matte, x + r] = np.round(
                pic[matte, r] * ((2 * l - matte) / (2 * l)) + image[y + matte, x + r] * (matte / (2 * l)))
        for matte in range(ll):
            image[y + matte + l, x + r] = np.round(
                pic[matte + l, r] * ((ll - matte) / (2 * ll)) + image[y + matte + l, x + r] * ((ll + matte) / (2 * ll)))

    # ********************************************************************************************************************

    check1 = cv2.cvtColor(pic[100:150, :], cv2.COLOR_BGR2GRAY)
    check2 = cv2.cvtColor(image[y:y + 50, x:x + 100], cv2.COLOR_BGR2GRAY)
    list = bcut_h(100, check1, check2)

    for r in range(len(list)):
        l = list[r]
        ll = 50 - l
        for matte in range(l):
            image[100 + y + matte, x + r] = np.round(
                image[100 + y + matte, x + r] * ((2 * l - matte) / (2 * l)) + pic[matte + 100, r] * (matte / (2 * l)))
        for matte in range(ll):
            image[100 + y + matte + l, x + r] = np.round(
                pic[matte + 100 + l, r] * ((ll + matte) / (2 * ll))) + image[100 + y + matte + l, x + r] * (
                                                        (ll - matte) / (2 * ll))


# ********************************************************************************************************************

def last_clmn(limit, x, y, points):
    picb, picg, picr = cv2.split(image)

    matchb1 = np.mean(picb[y + 50:y + 100, x:x + 50])
    matchb2 = np.mean(picb[y:y + 50, x:x + 150])
    matchb3 = np.mean(picb[y + 50:y + 100, x + 100:x + 150])

    matchg1 = np.mean(picg[y + 50:y + 100, x:x + 50])
    matchg2 = np.mean(picg[y:y + 50, x:x + 150])
    matchg3 = np.mean(picg[y + 50:y + 100, x + 100:x + 150])

    matchr1 = np.mean(picr[y + 50:y + 100, x:x + 50])
    matchr2 = np.mean(picr[y:y + 50, x:x + 150])
    matchr3 = np.mean(picr[y + 50:y + 100, x + 100:x + 150])

    matchb = (matchb1 + (matchb2 * 3) + matchb3) / 5
    matchg = (matchg1 + (matchg2 * 3) + matchg3) / 5
    matchr = (matchr1 + (matchr2 * 3) + matchr3) / 5

    pic = np.copy(image[y:y + 100, x:x + 150])
    pic[50:, 50:100] = (np.zeros((50, 50, 3), dtype='uint8') + (matchb, matchg, matchr))

    res = cv2.matchTemplate(image, pic, cv2.TM_CCOEFF_NORMED)
    threshold = 1.
    while 1:
        destroyed = False
        threshold -= (1 / 200)
        y_coord, x_coord = np.where(res >= threshold)
        if len(x_coord) > limit:
            found = False
            for q in range(limit):
                index = rnd.randrange(0, len(y_coord))
                destroyed = False
                for point in points:
                    if point[0] - 100 <= y_coord[index] <= point[2] and point[1] - 150 <= x_coord[index] <= point[3]:
                        x_coord = np.delete(x_coord, index)
                        y_coord = np.delete(y_coord, index)
                        destroyed = True
                        break
                if not destroyed:
                    coordinates = int(y_coord[index]), int(x_coord[index])
                    found = True
                    break
            if found:
                break
            if destroyed:
                threshold += 9 / 2000
            if threshold == 0:
                threshold = 1

    image[y:y + 100, x:x + 150] = image[coordinates[0]:coordinates[0] + 100,
                                  coordinates[1]:coordinates[1] + 150]

    check1 = cv2.cvtColor(pic[:50, :], cv2.COLOR_BGR2GRAY)
    check2 = cv2.cvtColor(image[y:y + 50, x:x + 150], cv2.COLOR_BGR2GRAY)
    list = bcut_horizontal(check1, check2)

    for r in range(len(list)):
        l = list[r]
        ll = 50 - l
        for matte in range(l):
            image[y + matte, x + r] = np.round(
                pic[matte, r] * ((2 * l - matte) / (2 * l)) + image[y + matte, x + r] * (matte / (2 * l)))
        for matte in range(ll):
            image[y + matte + l, x + r] = np.round(
                pic[matte + l, r] * ((ll - matte) / (2 * ll))) + image[y + matte + l, x + r] * ((ll + matte) / (2 * ll))

    # ********************************************************************************************************************

    check1 = cv2.cvtColor(pic[:, :50], cv2.COLOR_BGR2GRAY)
    check2 = cv2.cvtColor(image[y:y + 100, x:x + 50], cv2.COLOR_BGR2GRAY)

    list = bcut_v(100, check1, check2)

    for r in range(len(list)):
        l = list[r]
        ll = 50 - l
        for matte in range(l):
            image[y + r, x + matte] = np.round(
                pic[r, matte] * ((2 * l - matte) / (2 * l)) + image[y + r, x + matte] * (matte / (2 * l)))
        for matte in range(ll):
            image[y + r, x + matte + l] = np.round(
                pic[r, matte + l] * ((ll - matte) / (2 * ll)) + image[y + r, x + matte + l] * ((ll + matte) / (2 * ll)))

    # ********************************************************************************************************************

    check1 = cv2.cvtColor(pic[:, 100:150], cv2.COLOR_BGR2GRAY)
    check2 = cv2.cvtColor(image[y:y + 100, x + 100:x + 150], cv2.COLOR_BGR2GRAY)
    list = bcut_v(100, check1, check2)

    for r in range(len(list)):
        l = list[r]
        ll = 50 - l
        for matte in range(l):
            image[y + r, x + matte + 100] = np.round(
                image[y + r, x + matte + 100] * ((2 * l - matte) / (2 * l)) + pic[r, matte + 100] * (matte / (2 * l)))
        for matte in range(ll):
            image[y + r, x + matte + l + 100] = np.round(
                pic[r, matte + 100 + l] * ((ll + matte) / (2 * ll)) + image[y + r, x + matte + 100 + l] * (
                        (ll - matte) / (2 * ll)))


# *****************************************************************************


def end(limit, x, y, points):
    picb, picg, picr = cv2.split(image)

    matchb1 = np.mean(picb[y + 100:y + 150, x:x + 150])
    matchb2 = np.mean(picb[y:y + 50, x:x + 150])
    matchb3 = np.mean(picb[y + 50:y + 100, x:x + 50])
    matchb4 = np.mean(picb[y + 50:y + 100, x + 100:x + 150])

    matchg1 = np.mean(picg[y + 100:y + 150, x:x + 150])
    matchg2 = np.mean(picg[y:y + 50, x:x + 150])
    matchg3 = np.mean(picg[y + 50:y + 100, x:x + 50])
    matchg4 = np.mean(picg[y + 50:y + 100, x + 100:x + 150])

    matchr1 = np.mean(picr[y + 100:y + 150, x:x + 150])
    matchr2 = np.mean(picr[y:y + 50, x:x + 150])
    matchr3 = np.mean(picr[y + 50:y + 100, x:x + 50])
    matchr4 = np.mean(picr[y + 50:y + 100, x + 100:x + 150])

    matchb = ((matchb1 * 3) + (matchb2 * 3) + matchb3 + matchb4) / 8
    matchg = ((matchg1 * 3) + (matchg2 * 3) + matchg3 + matchg4) / 8
    matchr = ((matchr1 * 3) + (matchr2 * 3) + matchr3 + matchr4) / 8

    pic = np.copy(image[y:y + 150, x:x + 150])
    pic[50:100, 50:100] = (np.zeros((50, 50, 3), dtype='uint8') + (matchb, matchg, matchr))

    res = cv2.matchTemplate(image, pic, cv2.TM_CCOEFF_NORMED)
    threshold = 1
    while 1:
        destroyed = False
        threshold -= (1 / 200)
        y_coord, x_coord = np.where(res >= threshold)
        if len(x_coord) > limit:
            found = False
            for q in range(limit):
                index = rnd.randrange(0, len(y_coord))
                destroyed = False
                for point in points:
                    if point[0] - 150 <= y_coord[index] <= point[2] and point[1] - 150 <= x_coord[index] <= point[3]:
                        x_coord = np.delete(x_coord, index)
                        y_coord = np.delete(y_coord, index)
                        destroyed = True
                        break
                if not destroyed:
                    coordinates = int(y_coord[index]), int(x_coord[index])
                    found = True
                    break
            if found:
                break
            if destroyed:
                threshold += 9 / 2000
            if threshold == 0:
                threshold = 1

    image[y:y + 150, x:x + 150] = image[coordinates[0]:coordinates[0] + 150,
                                  coordinates[1]:coordinates[1] + 150]

    check1 = cv2.cvtColor(pic[:50, :], cv2.COLOR_BGR2GRAY)
    check2 = cv2.cvtColor(image[y:y + 50, x:x + 150], cv2.COLOR_BGR2GRAY)
    list = bcut_horizontal(check1, check2)

    for r in range(len(list)):
        l = list[r]
        ll = 50 - l
        for matte in range(l):
            image[y + matte, x + r] = np.round(
                pic[matte, r] * ((2 * l - matte) / (2 * l)) + image[y + matte, x + r] * (matte / (2 * l)))
        for matte in range(ll):
            image[y + matte + l, x + r] = np.round(
                pic[matte + l, r] * ((ll - matte) / (2 * ll))) + image[y + matte + l, x + r] * ((ll + matte) / (2 * ll))

    # ********************************************************************************************************************
    check1 = cv2.cvtColor(pic[:50, :], cv2.COLOR_BGR2GRAY)
    check2 = cv2.cvtColor(image[y + 100:y + 150, x:x + 150], cv2.COLOR_BGR2GRAY)
    list = bcut_horizontal(check1, check2)

    for r in range(len(list)):
        l = list[r]
        ll = 50 - l
        for matte in range(l):
            image[y + matte + 100, x + r] = np.round(
                image[y + matte + 100, x + r] * ((2 * l - matte) / (2 * l)) + pic[matte + 100, r] * (matte / (2 * l)))
        for matte in range(ll):
            image[y + matte + 100 + l, x + r] = np.round(
                image[y + matte + 100 + l, x + r] * ((ll - matte) / (2 * ll))) + pic[matte + 100 + l, r] * (
                                                        (ll + matte) / (2 * ll))

    # ********************************************************************************************************************

    check1 = cv2.cvtColor(pic[:, :50], cv2.COLOR_BGR2GRAY)
    check2 = cv2.cvtColor(image[y:y + 150, x:x + 50], cv2.COLOR_BGR2GRAY)
    list = bcut_vertical(check1, check2)

    for r in range(len(list)):
        l = list[r]
        ll = 50 - l
        for matte in range(l):
            image[y + r, x + matte] = np.round(
                pic[r, matte] * ((2 * l - matte) / (2 * l)) + image[y + r, x + matte] * (matte / (2 * l)))
        for matte in range(ll):
            image[y + r, x + matte + l] = np.round(
                pic[r, matte + l] * ((ll - matte) / (2 * ll)) + image[y + r, x + matte + l] * ((ll + matte) / (2 * ll)))

    # ********************************************************************************************************************

    check1 = cv2.cvtColor(pic[:, :50], cv2.COLOR_BGR2GRAY)
    check2 = cv2.cvtColor(image[y:y + 150, x + 100:x + 150], cv2.COLOR_BGR2GRAY)
    list = bcut_vertical(check1, check2)

    for r in range(len(list)):
        l = list[r]
        ll = 50 - l
        for matte in range(l):
            image[y + r, x + matte + 100] = np.round(
                image[y + r, x + matte + 100] * ((2 * l - matte) / (2 * l)) + pic[r, matte + 100] * (matte / (2 * l)))
        for matte in range(ll):
            image[y + r, x + matte + 100 + l] = np.round(
                image[y + r, x + matte + l + 100] * ((ll - matte) / (2 * ll)) + pic[r, matte + l + 100] * (
                        (ll + matte) / (2 * ll)))

    # ********************************************************************************************************************


def smoothing(image, size_w, size_h, point):
    for j in range(size_w + 1):
        for i in range(point[0] - 50, point[2] + 50):
            for o in range(-1, 2):
                x = o + point[1] + 50 * j
                image[i, x] = np.sum(image[i - 1:i + 2, x - 1:x + 2] / 9, (0, 1))

    for j in range(size_h + 1):
        for o in range(-1, 2):
            x = o + point[0] + 50 * j
            for i in range(point[1] - 50, point[3] + 50):
                image[x, i] = np.sum(image[x - 1:x + 2, i - 1:i + 2] / 9, (0, 1))


def find_points(new_pic, x1, y1):
    for ii in range(new_pic.shape[0]):
        if new_pic[x1 + ii, y1] != 0:
            for jj in range(new_pic.shape[1]):
                if new_pic[x1 + ii - 1, y1 + jj] != 0:
                    x2 = x1 + ii - 1
                    y2 = y1 + jj - 1
                    break
            break
    return x1, y1, x2, y2


def find_area(new_pic):
    points = []
    for i in range(new_pic.shape[0]):
        for j in range(new_pic.shape[1]):
            find = True
            if new_pic[i, j] == 0:
                for point in points:
                    if point[0] <= i <= point[2] and point[1] <= j <= point[3]:
                        find = False
                if find:
                    found = find_points(new_pic, i, j)
                    if found[2] - found[0] > 5 and found[3] - found[1] > 5:
                        points.append(found)

    return points


def q3(image):
    new_pic = np.zeros((image.shape[0], image.shape[1])) + image[:, :, 0] + image[:, :, 1] + image[:, :, 2]
    points = find_area(new_pic)

    for point in points:

        p += 1
        print('**************************************************************************')
        print(p, '/', len(points), 'case -> ', point)
        size_h = int((point[2] - point[0]) / 50)
        size_w = int((point[3] - point[1]) / 50)

        for i in range(size_h):
            r = point[0] - 50 + (50 * i)
            print(i + 1, '/', size_h, ' -main structure in progress ...')
            for j in range(size_w):
                x = point[1] - 50 + (50 * j)
                main(patch_main, match_area, limit, x, r, points)
        print('main structure completed')

        image[point[2] - 50:point[2], point[1]:point[3]] = np.zeros_like(
            image[point[2] - 50:point[2], point[1]:point[3]])

        for i in range(size_w):
            print(i + 1, '/', size_w, ' -last row in progress ...')
            last_row(limit, point[1] - 50 + (i * 50), point[2] - 100, points)
        print('last row completed')

        image[point[0]:point[2], point[3] - 50:point[3]] = np.zeros_like(
            image[point[0]:point[2], point[3] - 50:point[3]])

        for j in range(size_h):
            print(j + 1, '/', size_h, ' -last column in progress ...')
            last_clmn(limit, point[3] - 99, point[0] - 50 + (j * 50), points)
        print('last column completed')

        print('last block in progress ...')
        image[point[2] - 50:point[2] + 1, point[3] - 50:point[3] + 1] = np.zeros_like(
            image[point[2] - 50:point[2] + 1, point[3] - 50:point[3] + 1])
        end(limit, point[3] - 100, point[2] - 100, points)
        print('last block completed')

        print('smoothing in progress ...')
        smoothing(image, size_w, size_h, point)
        print('smoothing completed')
        print('')
    return image


t0 = time.time()

image = cv2.imread("im04.png", 1)  # get holed image

patch_main = 100  # define height of patches
match_area = 50  # define width of patches
limit = 3  # minimum matches of template
max = 255 ** 2 * 150  # maximum value for boundary cutes
res = q3(image)  # filling holes

cv2.imwrite('res16.jpg', res)  # save image

t1 = time.time()
print('runtime: ' + str(int(t1 - t0)) + ' seconds')
