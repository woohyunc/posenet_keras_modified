import math
import helper
import posenet
import numpy as np
from keras.optimizers import Adam
from transforms3d import quaternions
from scipy.spatial.transform import Rotation as R
import csv
import time

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

def quat2AA(quat):
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    yawMatrix = np.matrix([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ])
    pitchMatrix = np.matrix([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])
    rollMatrix = np.matrix([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
    ])

    R = yawMatrix * pitchMatrix * rollMatrix
    theta = math.acos(((R[0, 0] + R[1, 1] + R[2, 2]) - 1) / 2)
    multi = 1 / (2 * math.sin(theta))

    rx = multi * (R[2, 1] - R[1, 2]) * theta
    ry = multi * (R[0, 2] - R[2, 0]) * theta
    rz = multi * (R[1, 0] - R[0, 1]) * theta
    return rx, ry, rz


def decode_angle(rx, ry, rz, mode='euclid'):
    rotMat = rodrigues(np.array([rx, ry, rz]))
    if mode == 'quaternion':
        return quaternions.mat2quat(rotMat)
    xz = np.array([rotMat[2][0], rotMat[2][2]])
    t = angle_between(xz, np.array([0, 1]))
    return t  # [0,360]


class PriorityQueue(object):
    def __init__(self):
        self.queue = []
    def __str__(self):
        return ' '.join([str(i) for i in self.queue])
    def isEmpty(self):
        return len(self.queue) == 0
    def insert(self, data):
        self.queue.append(data)
    def delete(self):
        try:
            max_val = 0
            for i in range(len(self.queue)):
                if self.queue[i] < self.queue[max_val]:
                    max_val = i
            item = self.queue[max_val]
            del self.queue[max_val]
            return item
        except IndexError:
            print()
            exit()

if __name__ == "__main__":
    # Test model
    model = posenet.create_posenet()
    model.load_weights('200_weights.h5')
    adam = Adam(lr=0.001, clipvalue=1.5)
    model.compile(optimizer=adam, loss={'cls1_fc_pose_xyz': posenet.euc_loss1x, 'cls1_fc_pose_wpqr': posenet.euc_loss1q,
                                        'cls2_fc_pose_xyz': posenet.euc_loss2x, 'cls2_fc_pose_wpqr': posenet.euc_loss2q,
                                        'cls3_fc_pose_xyz': posenet.euc_loss3x,
                                        'cls3_fc_pose_wpqr': posenet.euc_loss3q})

    dataset_train, dataset_test = helper.getData()
    vals = []
    with open('minmax_vals.txt', 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            vals.append(row)
    maxx = vals[0]
    minx = vals[1]
    maxy = vals[2]
    miny = vals[3]
    maxz = vals[4]
    minz = vals[5]

    X_test = np.squeeze(np.array(dataset_test.images))
    y_test = np.squeeze(np.array(dataset_test.poses))

    testPredict = model.predict(X_test)

    valsx = testPredict[4]
    valsq = testPredict[5]

    results = np.zeros((len(dataset_test.images), 2))

    n_list = []
    num_n = 0
    N_list = [1,5,10,20]
    recall_bool = False
    start_time = time.time()

    for N in N_list:
        num_n = 0
        for j in range (len(dataset_test.images)):
            n_list = PriorityQueue()
            #n_list = []
            recall_bool = False
            for i in range(len(dataset_test.images)):
                # Calculate L2 distances for each image in relation to query image
                # Save the top @N closest images to a list
                # Return "True" if at least one of the @N images are within 5 meters of the Ground truth
                # Of the 314 images, how many query images returned True? ==> n
                # return n * 100 / N
                pose_q = np.asarray(dataset_test.poses[j][3:7])
                pose_x = np.asarray(dataset_test.poses[j][0:3])
                predicted_x = valsx[i]
                predicted_q = valsq[i]

                pose_q = np.squeeze(pose_q)
                pose_x = np.squeeze(pose_x)
                predicted_q = np.squeeze(predicted_q)
                predicted_x = np.squeeze(predicted_x)
                
                # Compute Individual Distance
                error_x = np.linalg.norm(pose_x - predicted_x)
                q1 = pose_q / np.linalg.norm(pose_q)
                q2 = predicted_q / np.linalg.norm(predicted_q)
                d = abs(np.sum(np.multiply(q1, q2)))
                theta = 2 * np.arccos(d) * 180 / math.pi
                #n_list.append([error_x+theta, error_x])
                n_list.insert([error_x+theta, error_x, theta])
            #n_list.sort()
            for k in range(N):
                n = n_list.delete()
                if n[1] <= 10 and n[2] <= 20: #10m radius
                    #print(n_list[k])
                    recall_bool = True
            if recall_bool is True:
                num_n +=1
        print("Recall for this no angle model @",N," is : ", num_n*100/len(dataset_test.images),"%")
        #print("Time taken : ", time.time()-start_time, "seconds")
