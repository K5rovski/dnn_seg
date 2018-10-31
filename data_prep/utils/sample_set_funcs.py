import cv2

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import random
import sys
import itertools
from sklearn import cluster
from scipy.spatial.distance import euclidean as euclidean
import pickle
import math

random.seed(8)
np.random.seed(9)

dbscan = cluster.DBSCAN

class ConfigSample:
    def __init__(self,dikt={}):
        for key in dikt:
            setattr(self, key, dikt[key])




def get_good_patches(test_images, save_loc, win_offs,conf_pass):
    global conf
    conf=conf_pass

    good_pix = {}
    print("Iterating all images...")
    for i, filen in enumerate(test_images['smallc']):
        img_smallc = cv2.imread(test_images['smallc'][i], 0)
        img_bigc = cv2.imread(test_images['bigc'][i], 0)
        img_int = cv2.imread(test_images['interim'][i], 0)
        img_gnd = cv2.imread(test_images['groundtruth'][i], 0)
        good_pix[i] = []

        #         print(set(img_gnd.flatten()))

        print('img is: ', i,'/',len(test_images['smallc']),', ', 
            np.sum(img_bigc[:, :] == 255), 
			' big ROIs border points.'
			# np.sum(img_bigc[:, :] == 255) / float(conf.img_size ** 2)
			)

        err_map_big = (img_bigc[:, :] == 255).astype(np.uint8)
        ret, err_conts_big = cv2.connectedComponents(err_map_big, connectivity=8)

        err_map_small = (img_smallc[:, :] == 255).astype(np.uint8)
        ret, err_conts_small = cv2.connectedComponents(err_map_small, connectivity=8)

        print(filen)

        img_c = img_bigc.copy()
        add_goodpix(err_conts_big, good_pix, i, img_c, 0.85, label='Big')

        img_c = img_smallc.copy()
        add_goodpix(err_conts_small, good_pix, i, img_c, 0.95, label='Small')

        add_goodpix_noise(good_pix, i, img_int, img_gnd, label='Noise', i=i)

    # Sth else         good_pix[i].extend(get_points_debri(test_images['debri'][i],i,img_int))

    #         if i==3: break

    all_lens = 0
    non_noise_lens=0
    noise_lens=0

    for ind in good_pix:
        file_key = get_img_key(test_images['groundtruth'][ind])
        file_key += '_mask.tif'

        save_img = np.zeros((conf.img_size, conf.img_size), dtype=np.uint8)

        print('Number of good_patches: ', len(good_pix[ind]))

        for x, y, label in good_pix[ind]:

            if label == 'Small':
                apply_random_window(save_img, x, y, conf.win_offs_small,
                                    conf.win_sparsity_small, 255)
                save_img[x, y] = 255
            elif label == 'Big':
                apply_random_window(save_img, x, y,
                                    conf.win_offs_big, conf.win_sparsity_big)
                save_img[x, y] = 255
            elif label == 'Noise_roi':
                pass
            elif label == 'Noise':
                apply_random_window(save_img, x, y, conf.win_offs_noise,
                                    conf.win_sparsity_noise, conf.noise_val)
                save_img[x, y] = conf.noise_val

        all_lens += np.sum(save_img != 0)
        
        non_noise_lens+=np.sum((save_img !=0)& (save_img!=conf.noise_val))
        noise_lens+=np.sum(save_img==conf.noise_val)

        cv2.imwrite(os.path.join(save_loc, file_key), save_img)

    print('All Patches are: {}\nRoi patches are: {}\nDebri patches are: {}'\
	.format( all_lens,non_noise_lens,noise_lens))
    return all_lens


def get_rad(wid, height):
    ret = math.sqrt(wid ** 2 + height ** 2)

    return round(ret / 2)


def num_neigh(x, y, a, val=255):
    count = 0
    for indo, (xo, yo) in enumerate(iterate_neigh()):
        if a[x + xo, y + yo] == val:
            count += 1
    return count


def append_iterables(*args):
    for iter_i in args:
        for it_i in iter_i:
            yield it_i


def iterate_neigh():
    one = itertools.product([0], [1, -1])
    one2 = itertools.product([1, -1], [0])
    sec = itertools.product([-1, 1], [-1, 1])
    return append_iterables(one, one2, sec, )


def run_neigh_cross():
    for x, y in list([(x, y) for x, y in itertools.permutations([-1, 1, 0], 2) if not np.all((x, y))]):
        yield x, y


list(iterate_neigh())


def run_neigh():
    for i, j in itertools.combinations_with_replacement((-1, 1, 0), 2):
        if i != 0:
            yield (i, j)
    for i, j in itertools.combinations_with_replacement((1, -1, 0), 2):
        if i != j:
            yield (i, j)


def get_img_key(filename):
    return filename[filename.rindex(os.path.sep) + 1:filename.rindex('img') + 5]


def check_neighbourhood_mask(img, x, y):
    blacks = 0
    grays = 0
    for i, j in run_neigh():
        if img[x + i, y + j] == 0:
            blacks += 1
        elif img[x + i, y + j] == 255:
            grays += 1

    if (blacks + grays) == 8 and blacks >= 1 and img[x, y] == 255:
        return True
    return False


def manhattan(a, b):
    first = abs(a[0] - b[0])
    second = abs(a[0] - b[0])
    #     if first==second and (first+second)>0:
    #         return first
    return first + second


def iterate_pixels(a, curves):
    start_pix = curves[0]
    a[curves[0][0], curves[0][1]] = 50

    lenPix = len(curves)
    countDikt = {tuple(start_pix): 0}

    iterating = [tuple(start_pix)]
    while True:
        new_nodes = []
        for x, y in iterating:
            count = 0
            for xo, yo in iterate_neigh():
                if x + xo >= a.shape[0] or y + yo >= a.shape[1]:
                    continue
                if a[x + xo, y + yo] == 255:
                    count += 1
                    new_nodes.append((x + xo, y + yo))

                    countDikt[(x + xo, y + yo)] = countDikt[(x, y)] + 1

                    a[x + xo, y + yo] = 50

        iterating = new_nodes

        if len(iterating) == 0:
            break

    countTup = [(key, countDikt[key]) for key in countDikt]
    countTup.sort(key=lambda x: x[1])

    return [ct[0] for ct in countTup]


def check_neigh_noise(img, x, y, howMuch, whichVal=255):
    count = 0
    for xi, yi in iterate_neigh():
        if (x + xi) >= img.shape[0] or (x + xi) < 0 or (y + yi) >= img.shape[1] or (y + yi) < 0:
            continue
        try:
            if img[x + xi, y + yi] == whichVal:
                count += 1
        except:
            print(x + xi, y + yi)
            raise Exception()
    return count >= howMuch


def check_neigh_noise(img, x, y, howMuch, whichVal=255):
    count = 0
    for xi, yi in iterate_neigh():
        if (x + xi) >= img.shape[0] or (x + xi) < 0 or (y + yi) >= img.shape[1] or (y + yi) < 0:
            continue
        try:
            if img[x + xi, y + yi] == whichVal:
                count += 1
        except:
            print(x + xi, y + yi)
            raise Exception()
    return count >= howMuch


def add_goodpix_noise(good_pix, gpix_ind, img_int, img_gnd, label='Noise', i=None):
    xses, yses = np.where(((img_int == 128) | (img_int == 0)) & (img_gnd == 3))

    #     print('This is: ',i,check_neigh_noise(img_gnd,1887,571,2))

    for x, y in zip(xses, yses):
        if np.random.random() < conf.noise_big_sparsity and check_neigh_noise(img_int, x, y, 1,
                                                                         whichVal=255) and not check_neigh_noise(
                img_gnd, x, y, 2, whichVal=1):
            # This x,y are noise
            good_pix[gpix_ind].append((x, y, label))


def add_goodpix(err_conts, good_pix, gpix_ind, img_c, curve_sparsity=0.3, label='Small'):
    #     print('There are this many in here',len(set(err_conts.flatten())))

    #     if gpix_ind==13:
    #         plt.figure(figsize=(20,20))
    #         plt.imshow(img_c[1095:1140,1010:1055],cmap='gray')
    #         print(img_c[1095:1140,1010:1055])
    for err_cont in set(err_conts.flatten()):
        if err_cont == 0:
            continue

        err_cont_pnts = np.where(err_conts == err_cont)
        if len(err_cont_pnts[0]) > 11000:
            pass
            # print('!!!!!!!!!!!!!!!!!! VERY LARCGE',len(err_cont_pnts[0]),11000)
            # continue
            # raise Exception('Very big contour' + str(len(err_cont_pnts[0])))

        err_cont_pnts = list(np.vstack(err_cont_pnts).T)

        iterated_pix = np.array(iterate_pixels(img_c, err_cont_pnts))

        howMuch = int(len(err_cont_pnts) / (len(err_cont_pnts) * curve_sparsity))
		
        #         if gpix_ind==13:
        #             print(howMuch,len(err_cont_pnts))

        #         if gpix_ind==13 and test_pixels(iterated_pix):
        #             print('err conts are: ',err_cont,iterated_pix[np.arange(0,len(err_cont_pnts),howMuch)])

        good_pix[gpix_ind].extend(
            [(i, j, label) for i, j in iterated_pix[np.arange(0, len(err_cont_pnts), howMuch)]])


def test_pixels(iterated_pix):
    for x, y in iterated_pix:
        if x > 1095 and x < 1140 and y > 1010 and y < 1055:
            return True
    return False


def apply_random_window(img, x, y, offset, sparsity, valm=255):
    pass
    left_x = max(x - offset, 0)
    left_y = max(y - offset, 0)

    right_x = min(x + offset + 1, img.shape[0] - 1)
    right_y = min(y + offset + 1, img.shape[1] - 1)

    window = np.zeros(((right_x - left_x), (right_y - left_y)), dtype=np.uint8)

    window[(np.random.rand(window.shape[0], window.shape[1]) < sparsity)] = valm

    midpoint_x = (window.shape[0] - 1) // 2
    midpoint_y = (window.shape[1] - 1) // 2

    window[midpoint_x, midpoint_y] = valm

    #     plt.imshow(window)
    #     print(window,window.shape)

    img[left_x:right_x, left_y:right_y] = window


#     sys.exit('im at window')

def calc_obj_dist(ac, bc, c=0):
    raw_dist = euclidean(ac[1:3], bc[1:3])

    raw_dist -= ac[3] + bc[3]
    if np.sum(ac - ac.astype(int)) > 0 or np.sum(bc - bc.astype(int)) > 0:
        if random.random() < 0.05: print('non done', ac, bc)
    if raw_dist < 0:
        #         print('bad raw dist',raw_dist,a,b)
        pass
        raw_dist += ac[3] + bc[3]
        c += 1
    # raise Exception('negative Distance'+str(raw_dist))


    return raw_dist, c


def get_sample_noise(window, winoffs, sample_spars, label='Noise_roi'):
    minx, maxx, miny, maxy = window
    centx = int((maxx - minx) / 2) + minx
    centy = int((maxy - miny) / 2) + miny

    roomx = int((centx - minx) / winoffs) + 1
    roomy = int((centy - miny) / winoffs) + 1

    points = []
    for indx, indy in itertools.product(range(roomx), range(roomy)):
        locx = int(centx - (winoffs * indx))
        locy = int(centy - (winoffs * indy))
        points.append((locx, locy, label))

    # if roomx>4:
    #         print('len points',len(points),roomx,roomy)
    #         sys.exit('')

    spar_xs, spar_ys = np.where(np.random.rand(maxx - minx, maxy - miny) < sample_spars)
    #     print('len spars',len(spar_xs),maxx-minx,maxy-miny)

    for spx, spy in zip(spar_xs, spar_ys):
        points.append((spx + minx, spy + miny, label))

    return points


def add_singlepix_dist(dist_mat, objects):
    onepix_len = np.sum(objects[:, 4] == 1)
    morepix_len = np.sum(objects[:, 4] != 1)

    objects_one = objects[objects[:, 4] == 1]
    objects_two = objects[objects[:, 4] != 1]

    dist_matadd = np.zeros((onepix_len + morepix_len, onepix_len)) - 100
    c = 0
    for obi, obj in itertools.product(range(morepix_len), range(onepix_len)):
        if ((obi * objects.shape[0]) + obj) % 810300 == 3:
            print(  obi)
        dist_matadd[obi, obj], c = calc_obj_dist(objects_two[obi], objects_one[obj], c)

    for obi, obj in itertools.product(range(onepix_len), range(onepix_len)):
        if ((obi * objects.shape[0]) + obj) % 810300 == 3:
            print( obi)
        dist_matadd[obi + morepix_len, obj], c = calc_obj_dist(objects_one[obi], objects_one[obj], c)

    dist_mat1 = np.hstack((dist_mat, dist_matadd[:morepix_len]))

    dist_mat2 = np.vstack((dist_mat1, dist_matadd.T))

    return dist_mat2


def get_noise_objects(img_mask_str):
    img = cv2.imread(img_mask_str, 0)
    img[img > 1] = 0
    comlen, comps, cstats, ccents = cv2.connectedComponentsWithStats(img)

    if np.any(cstats[0, :2]) or cstats[0, 2] != img.shape[0]:
        raise Exception('Firs component is not background')

    objects = np.zeros((comlen - 1, 6))
    for indo, (ccent, cstat) in enumerate(zip(ccents[1:], cstats[1:])):
        objects[indo, 1:3] = map(round, ccent)
        objects[indo, 3] = get_rad(cstat[cv2.CC_STAT_WIDTH], cstat[cv2.CC_STAT_HEIGHT])
        objects[indo, 0] = indo
        objects[indo, 4] = cstat[cv2.CC_STAT_AREA]

    return objects


def get_points_debri(imgstr, img_ind, img_interim):
    img = cv2.imread(imgstr, 0)
    img[img > 1] = 0
    comlen, comps, cstats, ccents = cv2.connectedComponentsWithStats(img)

    if np.any(cstats[0, :2]) or cstats[0, 2] != img.shape[0]:
        raise Exception('Firs component is not background')

    objects = np.zeros((comlen - 1, 6))
    for indo, (ccent, cstat) in enumerate(zip(ccents[1:], cstats[1:])):
        objects[indo, 1:3] = map(round, ccent)
        objects[indo, 3] = get_rad(cstat[cv2.CC_STAT_WIDTH], cstat[cv2.CC_STAT_HEIGHT])
        objects[indo, 0] = indo
        objects[indo, 4] = cstat[cv2.CC_STAT_AREA]

    reducedObj = False
    if np.percentile(objects[:, 4], conf.pix_remove_thres) == 1:
        reducedObj = True
        inds = np.arange(objects.shape[0], dtype=int)
        rand_ind = np.random.rand(inds.shape[0])
        non_dots = np.sum(objects[:, 4] != 1)
        inds = inds[(objects[:, 4] != 1) | (
        (objects[:, 4] == 1) & (rand_ind < ((conf.leave_pix_noise * non_dots) / objects.shape[0])))]
        objects = objects[inds]

    if objects.shape[0] > 11000:
        raise Exception('Too many objects')

    if os.path.exists('Saved_Dist/saved_object_{}.dist'.format(img_ind)):
        with open('Saved_Dist/saved_object_{}.dist'.format(img_ind), 'rb') as f:
            dist_mat = pickle.load(f)
    else:
        raise Exception('No saved distance matrix')

    if reducedObj:
        dist_mat = add_singlepix_dist(dist_mat, objects)
        objects = np.vstack((objects[objects[:, 4] != 1], objects[objects[:, 4] == 1]))

    db = dbscan(eps=conf.dbscan_eps, min_samples=conf.dbscan_mins, metric='precomputed', algorithm='auto')

    objects[:, 5] = db.fit_predict(dist_mat, sample_weight=objects[:, 4])
    if not np.all(np.sum(objects - objects.astype(int), axis=1) == 0):
        raise Exception('Dbscan damaged objects')

    print('Clusters found: ', len(set(objects[:, 5])))

    show_img = np.zeros(comps.shape, dtype=int)

    good_points = []

    imgc = np.zeros(img.shape, dtype=np.uint8)

    for objc in set(objects[:, 5]):
        mask_c = True

        if objc < 0:
            continue

        ys = objects[objects[:, -1] == objc, 1]
        xs = objects[objects[:, -1] == objc, 2]

        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)

        if maxx - minx > 1500:
            print('4 img', objc, len(xs), objects[(objects[:, -1] == objc) & (objects[:, 4] == 1), 1:3], objects[(objects[:,     \
                                                                                                            -1] == objc) & (    \
                                                                                                           objects[:,         \
                                                                                                           4] != 1),   \
                                                                                                   1:3][:2]    )

        show_img[minx:maxx, miny:maxy] = 3
        good_points.extend(get_sample_noise((minx, maxx, miny, maxy), conf.win_offs, conf.win_noiseroi_spar))
        imgc[minx:maxx, miny:maxy] = img_interim[minx:maxx, miny:maxy]

    for obj in objects:
        if obj[-1] == -1:
            show_img[obj[2] - 5:obj[2] + 5, obj[1] - 5:obj[1] + 5] = 1
        else:
            show_img[obj[2] - 5:obj[2] + 5, obj[1] - 5:obj[1] + 5] = 2

    empty_clus = objects[objects[:, 5] == -1, 4]
    print('empty clus nonclustered: {}, onepix in clusters: {}'.format(np.sum(objects[:, 5] == -1), np.sum()
        (objects[:, 5] != -1) & (objects[:, 4] == 1))), np.percentile(empty_clus, (20, 50, 80, 95))

    plt.figure(figsize=(20, 20))
    plt.imshow(show_img, cmap='viridis')
    plt.savefig('Debri_Clusters/debri_cluseter_{}.png'.format(imgstr[imgstr.rindex('sp'):imgstr.rindex('.')]), dpi=300)
    plt.close()
    cv2.imwrite('Debri_Clusters/debri_patches_{}.png'.format(imgstr[imgstr.rindex('sp'):imgstr.rindex('.')]), imgc)

    return good_points
