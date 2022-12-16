import sys
import pandas as pd
from pandas import concat
from math import cos, sin, sqrt
from scipy.spatial.transform import Rotation as R
import numpy as np
from numpy import logical_and
from transforms3d import quaternions
import random

def dot(a, b):
    return np.sum(a * b, axis=-1)

def mag(a):
    return np.sqrt(np.sum(a * a, axis=-1))

def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))

def rodrigues(r: np.ndarray) -> np.ndarray:
    if r.size == 3:
        return R.from_rotvec(r.squeeze()).as_matrix()
    else:
        return R.from_matrix(r).as_rotvec().reshape((3, 1))

def decode_angle(rx, ry, rz, mode='euclid'):
    rotMat = rodrigues(np.array([rx, ry, rz]))

    if mode == 'quaternion':
        #print(quaternions.axangle2quat([rx,ry,rz],np.linalg.norm([rx,ry,rz])))
        #print(quaternions.mat2quat(rotMat))
        return quaternions.mat2quat(rotMat)
    xz = np.array([rotMat[2][0], rotMat[2][2]])
    t = angle_between(xz, np.array([0, 1]))
    return t # [0,360]

locations = []
query_locations = []

CELL_SIZE = (114.6, 57.0) #Change based on desired cell size and train each cell individually
maxx = -10000
minx = 10000
maxz = -10000
minz = 10000
maxy = -10000
miny = 10000

qmaxx = -10000
qminx = 10000
qmaxz = -10000
qminz = 10000
qmaxy = -10000
qminy = 10000

train_img_directory = "../augmented_images/"
test_img_directory = "../query_imgs_grayscale/"

train_img_data = "./all_img_data.csv"
test_img_data = "./query_img_data.csv"

def importData():
    global maxx, minx, maxz, minz, maxy, miny
    global qmaxx, qminx, qmaxz, qminz, qmaxy, qminy

    col_list = ["photoID", "cameraNum", "xrot", "yrot", "zrot", "xpos", "ypos", "zpos"]
    df = pd.read_csv(train_img_data, usecols=col_list)
    col_list = ["photoID", "xrot", "yrot", "zrot", "xpos", "ypos", "zpos", "inlier"]
    qdf = pd.read_csv(test_img_data, usecols=col_list)
    xs = []
    zs = []

    for x, z in zip(df["xpos"], df["zpos"]):
        xs.append(x)
        zs.append(z)
    xs = np.array(xs)
    zs = np.array(zs)

    # compute average translation
    pt_org = concat((xs.reshape((1, -1)), zs.reshape((1, -1))), 0)
    pt_f = pt_org[:, :-1]
    pt_b = pt_org[:, 1:]
    pt_diff = sqrt(sum((pt_f - pt_b) ** 2, 0))
    avg_tr = np.mean(pt_diff[pt_diff < 5])
    print('# average translation: %.3f m', avg_tr)

    # shift coordinates
    cx, cz = np.mean(xs), np.mean(zs)
    center = np.array([cx, cz]).reshape((2, 1))
    deg = np.deg2rad(0)
    rot = np.array([cos(deg), sin(deg), -sin(deg), cos(deg)]).reshape((2, 2))
    pt = concat((xs.reshape((1, -1)), zs.reshape((1, -1))), 0) - center
    pt = rot @ pt + center
    min_pt = np.min(pt, 1)[:, np.newaxis]
    pt -= min_pt

    # compute cell index
    cells = [[0, 2]] #Change depending on which cell you want to train model on
    for cell in cells:
        target_cell_xy = np.array(cell).reshape((2, 1))
        cell_wh = np.array(CELL_SIZE).reshape((2, 1))
        cell_xy = (pt / cell_wh).astype(np.int)
        print(cell_xy.shape)
        cond = logical_and(
            cell_xy[0, :] == target_cell_xy[0], cell_xy[1, :] == target_cell_xy[1])
        nkfs = sum(cond)
        print('# number of kfs in (%02d, %02d) cell: %d', target_cell_xy[0], target_cell_xy[1], nkfs)

        for i, j, x, y, z, rx, ry, rz, photoid, camnum in zip(cell_xy[0], cell_xy[1], xs, df["ypos"], zs, df["xrot"], df["yrot"], df["zrot"],df["photoID"], df["cameraNum"]):
            if i == target_cell_xy[0] and j == target_cell_xy[1]:
                maxx = max(maxx, x)
                minx = min(minx, x)
                maxy = max(maxy, y)
                miny = min(miny, y)
                maxz = max(maxz, z)
                minz = min(minz, z)

                angs, img = decode_angle(rx, ry, rz, mode='quaternion'), ""
                if len(str(photoid)) == 4:
                    img = train_img_directory + "00" + str(photoid) + "_" + str(camnum)+".png"
                if len(str(photoid)) == 5:
                    img = train_img_directory + "0" + str(photoid) + "_" + str(camnum)+".png"
                locations.append([img, x, y, z, angs[0], angs[1], angs[2], angs[3]])
                
                if len(str(photoid)) == 4:
                    img = train_img_directory + "1_00" + str(photoid) + "_" + str(camnum)+".png"
                if len(str(photoid)) == 5:
                    img = train_img_directory + "1_0" + str(photoid) + "_" + str(camnum)+".png"
                locations.append([img, x, y, z, angs[0], angs[1], angs[2], angs[3]])
                
                if len(str(photoid)) == 4:
                    img = train_img_directory + "2_00" + str(photoid) + "_" + str(camnum)+".png"
                if len(str(photoid)) == 5:
                    img = train_img_directory + "2_0" + str(photoid) + "_" + str(camnum)+".png"
                locations.append([img, x, y, z, angs[0], angs[1], angs[2], angs[3]])

        i = 0
        for qx, qy, qz, qrx, qry, qrz, photoid in zip(qdf["xpos"], qdf["ypos"],qdf["zpos"],qdf["xrot"],qdf["yrot"],qdf["zrot"], qdf["photoID"]):
            if qx >= minx and qx <= maxx and qz >= minz and qz <= maxz:
                qmaxx = max(qmaxx, qx)
                qminx = min(qminx, qx)
                qmaxy = max(qmaxy, qy)
                qminy = min(qminy, qy)
                qmaxz = max(qmaxz, qz)
                qminz = min(qminz, qz)
                angs, img = decode_angle(qrx, qry, qrz, mode='quaternion'), ""
                if len(str(photoid)) == 4:
                    img = test_img_directory + "0" + str(photoid)+".png"
                if len(str(photoid)) == 5:
                    img = test_img_directory + str(photoid)+".png"
                query_locations.append([img, qx, qy, qz, angs[0], angs[1], angs[2], angs[3]])
                i += 1
        print('# number of query frames in (%02d, %02d) cell: %d', target_cell_xy[0], target_cell_xy[1], i)
    i = 0
    for coord in locations:
        locations[i][1] = coord[1] - minx
        locations[i][2] = coord[2] - miny
        locations[i][3] = coord[3] - minz
        i += 1
    i = 0
    for coord in query_locations:
        query_locations[i][1] = coord[1] - minx
        query_locations[i][2] = coord[2] - miny
        query_locations[i][3] = coord[3] - minz
        i += 1

    print("MAX_X, MIN_X, MAX-MIN : ", maxx, minx, maxx - minx, " (114.6)")
    print("MAX_Y, MIN_Y, MAX-MIN : ", maxy, miny, maxy - miny)
    print("MAX_Z, MIN_Z, MAX-MIN : ", maxz, minz, maxz - minz, " (57.0)")

vals = []
importData()

vals.append(maxx)
vals.append(minx)
vals.append(maxy)
vals.append(miny)
vals.append(maxz)
vals.append(minz)

random.shuffle(locations)
random.shuffle(query_locations)

strList = []
for row in locations:
    locations = [str(item) for item in row]
    strList.append(locations)
with open("./NewData/dataset_train.txt", "w",encoding="utf-8") as fo:
    for row in strList:
        list1 = [(item+" ") for item in row]
        fo.writelines(list1)
        fo.write('\n')

strList = []
for row in query_locations:
    query_locations = [str(item) for item in row]
    strList.append(query_locations)
with open("./NewData/dataset_test.txt", "w",encoding="utf-8") as fo:
    for row in strList:
        list1 = [(item+" ") for item in row]
        fo.writelines(list1)
        fo.write('\n')

np.asarray(vals)
np.savetxt('minmax_vals.txt', vals, delimiter=' ')
