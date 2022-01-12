import time
import cv2
import numpy as np
import random as rnd


def normalize(img):
    # linear normalizing img between 0 to 255
    return ((img - img.min()) * (255 / (img.max() - img.min()))).astype('uint8')


def resize(img, zarib):
    w = int(img.shape[1] * zarib)
    h = int(img.shape[0] * zarib)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def first_row(h_patch, w_area, limit, x, a, b):
    match_pic = new_pic[:h_patch, x:x + w_area]
    res = cv2.matchTemplate(texture[:, :texture.shape[1] - (100 - w_area)], match_pic, cv2.TM_CCOEFF_NORMED)
    for threshold in range(100):
        threshold = 1 - (threshold / 100)
        x_coord, y_coord = np.where(res >= threshold)
        if len(x_coord) > limit:
            break
    index = rnd.randrange(0, len(y_coord))
    coordinates = int(x_coord[index]), int(y_coord[index])

    new_pic[:h_patch, x + w_area:x + 100] = texture[coordinates[0]:coordinates[0] + h_patch,
                                            coordinates[1] + w_area:coordinates[1] + 100]
    check1 = cv2.cvtColor(new_pic[:h_patch, x:x + w_area], cv2.COLOR_BGR2GRAY)

    texture_cut = texture[coordinates[0]:coordinates[0] + h_patch, coordinates[1]:coordinates[1] + w_area]
    check2 = cv2.cvtColor(texture_cut, cv2.COLOR_BGR2GRAY)

    image_boundry = (check1 - check2) ** 2
    image_boundry = np.vstack(((np.zeros((1, w_area))), image_boundry))
    list = []
    max = 255 * 2 * h_patch
    bestcost = max
    for start in range(a, b):

        image_costs = np.zeros_like(image_boundry) + max
        image_costs[0, start] = 1
        for row in range(1, h_patch + 1):

            for clmn in range(w_area):
                if clmn < a + 1:
                    cost = np.min(image_costs[row - 1, clmn:clmn + 2]) + image_boundry[row, clmn]
                elif clmn > b - 1:
                    cost = np.min(image_costs[row - 1, clmn - 1:clmn + 1]) + image_boundry[row, clmn]
                else:
                    cost = np.min(image_costs[row - 1, clmn - 1:clmn + 2]) + image_boundry[row, clmn]
                image_costs[row, clmn] = cost
            if row == h_patch:
                totalcost = int(np.min(image_costs[h_patch, :]))

        if bestcost > totalcost:
            bestcost = totalcost
            best_cost = image_costs

    for row in range(h_patch - 1, -1, -1):
        min = max
        minx = 0
        if row == h_patch - 1:
            for clmn in range(a, b):
                cost = best_cost[row, clmn]
                if cost < min:
                    min = cost
                    minx = clmn
            list = []
            list.insert(0, minx)
        else:
            for clmn in range(-1, 2):
                if clmn + list[0] < a or clmn + list[0] > b:
                    cost = max
                else:
                    cost = best_cost[row, clmn + list[0]]
                if cost < min:
                    min = cost
                    minx = clmn + list[0]
            list.insert(0, minx)
    if gradient:
        for r in range(len(list)):
            l = list[r]
            ll = 50 - l
            for matte in range(l):
                new_pic[r, x + matte] = np.round(
                    new_pic[r, x + matte] * ((2 * l - matte) / (2 * l)) + texture_cut[r, matte] * (matte / (2 * l)))
            for matte in range(ll):
                new_pic[r, x + matte + l] = np.round(
                    new_pic[r, x + matte + l] * ((ll - matte) / (2 * ll)) + texture_cut[r, matte + l] * (
                            (ll + matte) / (2 * ll)))
    else:
        for r in range(len(list)):
            new_pic[r, x + list[r]: x + w_area] = texture_cut[r, list[r]: w_area]


def first_clm(h_patch, w_area, limit, x, a, b):
    match_pic = new_pic[x:x + w_area, :h_patch]
    res = cv2.matchTemplate(texture[:texture.shape[0] - (100 - w_area), :], match_pic, cv2.TM_CCOEFF_NORMED)
    for threshold in range(100):
        threshold = 1 - (threshold / 100)
        y_coord, x_coord = np.where(res >= threshold)
        if len(x_coord) > limit:
            break
    index = rnd.randrange(0, len(y_coord))
    coordinates = int(x_coord[index]), int(y_coord[index])

    new_pic[x + w_area:x + 100, :h_patch] = texture[coordinates[1] + w_area:coordinates[1] + 100,
                                            coordinates[0]:coordinates[0] + h_patch, ]
    check1 = cv2.cvtColor(new_pic[x:x + w_area, :h_patch], cv2.COLOR_BGR2GRAY)

    texture_cut = texture[coordinates[1]:coordinates[1] + w_area, coordinates[0]:coordinates[0] + h_patch]
    check2 = cv2.cvtColor(texture_cut, cv2.COLOR_BGR2GRAY)

    image_boundry = (check1 - check2) ** 2
    image_boundry = np.hstack(((np.zeros((w_area, 1))), image_boundry))
    cost = 0
    list = []
    max = 255 * 2 * h_patch
    bestcost = max
    for start in range(a, b):

        image_costs = np.zeros_like(image_boundry) + max
        image_costs[start, 0] = 1
        for row in range(1, h_patch + 1):

            for clmn in range(a, b):
                if clmn < a + 1:
                    cost = np.min(image_costs[clmn:clmn + 2, row - 1]) + image_boundry[clmn, row]
                elif clmn > b - 1:
                    cost = np.min(image_costs[clmn - 1:clmn + 1, row - 1]) + image_boundry[clmn, row]
                else:
                    cost = np.min(image_costs[clmn - 1:clmn + 2, row - 1]) + image_boundry[clmn, row]
                image_costs[clmn, row] = cost
            if row == h_patch:
                totalcost = int(np.min(image_costs[:, h_patch]))

        if bestcost > totalcost:
            bestcost = totalcost
            best_cost = image_costs

    for row in range(h_patch - 1, -1, -1):
        min = max
        minx = 0
        if row == h_patch - 1:
            for clmn in range(a, b):
                cost = best_cost[clmn, row]
                if cost < min:
                    min = cost
                    minx = clmn
            list = []
            save_cost = cost
            list.insert(0, minx)
        else:
            for clmn in range(-1, 2):

                if clmn + list[0] < a or clmn + list[0] > b:
                    cost = max
                else:
                    cost = best_cost[clmn + list[0], row]
                if cost < min:
                    min = cost
                    minx = clmn + list[0]
            list.insert(0, minx)
    if gradient:
        for r in range(len(list)):
            l = list[r]
            ll = 50 - l
            for matte in range(l):
                new_pic[x + matte, r] = np.round(
                    new_pic[x + matte, r] * ((2 * l - matte) / (2 * l)) + texture_cut[matte, r] * (matte / (2 * l)))
            for matte in range(ll):
                new_pic[x + matte + l, r] = np.round(
                    new_pic[x + matte + l, r] * ((ll - matte) / (2 * ll)) + texture_cut[matte + l, r] * (
                            (ll + matte) / (2 * ll)))
    else:
        for r in range(len(list)):
            new_pic[x + list[r]: x + w_area, r] = texture_cut[list[r]: w_area, r]


def main(h_patch, w_area, limit, x, y, a, b):
    picb, picg, picr = cv2.split(new_pic)

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

    pic = np.copy(new_pic[y:y + 100, x:x + 100])
    pic[50:, 50:] = (np.zeros((50, 50, 3), dtype='uint8') + (matchb, matchg, matchr))

    res = cv2.matchTemplate(texture, pic, cv2.TM_CCOEFF_NORMED)

    for threshold in range(200):
        threshold = 1 - (threshold / 200)
        x_coord, y_coord = np.where(res >= threshold)
        if len(x_coord) > limit:
            break
    index = rnd.randrange(0, len(y_coord))
    coordinates = int(x_coord[index]), int(y_coord[index])

    new_pic[y:y + 100, x:x + 100] = texture[coordinates[0]:coordinates[0] + 100,
                                    coordinates[1]:coordinates[1] + 100]

    check21 = cv2.cvtColor(pic[:50, :], cv2.COLOR_BGR2GRAY)
    check22 = cv2.cvtColor(new_pic[y:y + 50, x:x + 100], cv2.COLOR_BGR2GRAY)

    image_boundry = (check21 - check22) ** 2
    list = []
    max = 255 * 2 * h_patch
    image_costs = np.zeros_like(image_boundry) + max
    if gradient:
        t, t, loc, t = cv2.minMaxLoc(image_boundry[a:b, 99])
        image_costs[a + loc[1], 99] = 1
    else:
        image_costs[49, 99] = 1
    for row in range(h_patch - 2, -1, -1):
        for clmn in range(a, b):
            if clmn < a + 1:
                cost = np.min(image_costs[clmn:clmn + 2, row + 1]) + image_boundry[clmn, row]
            elif clmn > b - 1:
                cost = np.min(image_costs[clmn - 1:clmn + 1, row + 1]) + image_boundry[clmn, row]
            else:
                cost = np.min(image_costs[clmn - 1:clmn + 2, row + 1]) + image_boundry[clmn, row]
            image_costs[clmn, row] = cost

    for row in range(h_patch):
        min = max
        minx = 0
        if row == 0:
            for clmn in range(a, b):
                cost = image_costs[clmn, 0]
                if cost < min:
                    min = cost
                    minx = clmn
            list = []
            list.append(minx)
        else:
            for clmn in range(-1, 2):
                if clmn + list[len(list) - 1] < a or clmn + list[len(list) - 1] > b:
                    cost = max
                else:
                    cost = image_costs[clmn + list[len(list) - 1], row]
                if cost < min:
                    min = cost
                    minx = clmn + list[len(list) - 1]
            list.append(minx)

    if gradient:
        for r in range(len(list)):
            l = list[r]
            ll = 50 - l
            for matte in range(l):
                new_pic[y + matte, x + r] = np.round(
                    pic[matte, r] * ((2 * l - matte) / (2 * l)) + new_pic[y + matte, x + r] * (matte / (2 * l)))
            for matte in range(ll):
                new_pic[y + matte + l, x + r] = np.round(
                    pic[matte + l, r] * ((ll - matte) / (2 * ll)) + new_pic[y + matte + l, x + r] * (
                            (ll + matte) / (2 * ll)))

    else:
        for r in range(len(list)):
            new_pic[y: y + list[r], x + r] = pic[:list[r], r]

    # ********************************************************************************************************************

    check11 = cv2.cvtColor(pic[:, :50], cv2.COLOR_BGR2GRAY)
    check12 = cv2.cvtColor(new_pic[y:y + 100, x:x + 50], cv2.COLOR_BGR2GRAY)
    max = 255 * 2 * h_patch
    image_boundry = (check11 - check12) ** 2
    list = []
    image_costs = np.zeros_like(image_boundry) + max
    if gradient:
        t, t, loc, t = cv2.minMaxLoc(image_boundry[99, a:b])
        image_costs[99, a + loc[1]] = 1
    else:
        image_costs[99, 49] = 1
    for row in range(h_patch - 2, -1, -1):
        for clmn in range(a, b):
            if clmn < a + 1:
                cost = np.min(image_costs[row + 1, clmn:clmn + 2]) + image_boundry[row, clmn]
            elif clmn > b - 1:
                cost = np.min(image_costs[row + 1, clmn - 1:clmn + 1]) + image_boundry[row, clmn]
            else:
                cost = np.min(image_costs[row + 1, clmn - 1:clmn + 2]) + image_boundry[row, clmn]
            image_costs[row, clmn] = cost

    for row in range(h_patch):
        min = max
        minx = 0
        if row == 0:
            for clmn in range(a, b):
                cost = image_costs[0, clmn]
                if cost < min:
                    min = cost
                    minx = clmn
            list = []
            list.append(minx)
        else:
            for clmn in range(-1, 2):
                if clmn + list[len(list) - 1] < a or clmn + list[len(list) - 1] > b:
                    cost = max
                else:
                    cost = image_costs[row, clmn + list[len(list) - 1]]
                if cost < min:
                    min = cost
                    minx = clmn + list[len(list) - 1]
            list.append(minx)
    if gradient:
        for r in range(len(list)):
            l = list[r]
            ll = 50 - l
            for matte in range(l):
                new_pic[r + y, x + matte] = np.round(
                    pic[r, matte] * ((2 * l - matte) / (2 * l)) + new_pic[r + y, x + matte] * (matte / (2 * l)))
            for matte in range(ll):
                new_pic[r + y, x + matte + l] = np.round(
                    pic[r, matte + l] * ((ll - matte) / (2 * ll)) + new_pic[r + y, x + matte + l] * (
                            (ll + matte) / (2 * ll)))
    else:
        for r in range(len(list)):
            new_pic[y + r, x: x + list[r]] = pic[r, : list[r]]


def q2(new_pic):
    if gradient:
        a, b = 10, match_area - 11
    else:
        a, b = 0, match_area - 1
    new_pic[:patch_main, :patch_main] = texture[:patch_main, :patch_main]

    for i in range(size):
        x = (100 - match_area) + ((100 - match_area) * i)
        first_row(patch_main, match_area, limit, x, a, b)
        first_clm(patch_main, match_area, limit, x, a, b)
        print('1/2 -', int(51 * i / size), '%')

    for i in range(size):
        r = (100 - match_area) + ((100 - match_area) * i)
        for j in range(size):
            x = (100 - match_area) + ((100 - match_area) * j)
            main(patch_main, match_area, limit, x, r, a, b)
        print('2/2 -', int(51 + (51 * i / size)), '%')
    return new_pic


t0 = time.time()

texture = cv2.imread("tx5.png", 1)  # get texture
new_pic = np.zeros((2500, 2500, 3), 'uint8')  # make a black 2500 * 2500 image
patch_main = 100  # define height of patches
match_area = 50  # define width of patches
limit = 2  # minimum matches of template
size = int((2500 - patch_main) / (100 - match_area))  # size for loops for cover hole 2500*2500 picture
gradient = False  # select method of matching
result = q2(new_pic)  # make big picture
cv2.imwrite('res.jpg', result)  # save big image

t1 = time.time()
print('runtime: ' + str(int(t1 - t0)) + 'seconds')
