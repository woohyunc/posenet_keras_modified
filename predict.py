import math
import helper
import posenet
import numpy as np
from keras.optimizers import Adam
from transforms3d import quaternions
from scipy.spatial.transform import Rotation as R
import csv
import time
start_time = time.time()

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
    #vec, theta = quaternions.quat2axangle([quat[0],quat[1],quat[2],quat[3]])
    #vec = vec*theta
    #print("MODULE : ",vec)
    #print("FUNCTION : ", [rx,ry,rz])

    #Function verified with vec = [rx,ry,rz]
    return rx,ry,rz


def decode_angle(rx, ry, rz, mode='euclid'):
    '''
    rx,ry,rz ==> axis-angle representation of an angle
    mode ==> euclid : returns angle between [rx,ry,rz] and Z axis projected in relation to the XZ-plane
             quaternion : returns quaternion representation of [rx,ry,rz]
    '''
    rotMat = rodrigues(np.array([rx, ry, rz]))
    if mode == 'quaternion':
        return quaternions.mat2quat(rotMat)

    xz = np.array([rotMat[2][0], rotMat[2][2]])
    t = angle_between(xz, np.array([0, 1]))
    return t # [0,360]

if __name__ == "__main__":
    # Test model
    model = posenet.create_posenet()
    model.load_weights('200_weights.h5')
    adam = Adam(lr=0.001, clipvalue=1.5)
    model.compile(optimizer=adam, loss={'cls1_fc_pose_xyz': posenet.euc_loss1x, 'cls1_fc_pose_wpqr': posenet.euc_loss1q,
                                        'cls2_fc_pose_xyz': posenet.euc_loss2x, 'cls2_fc_pose_wpqr': posenet.euc_loss2q,
                                        'cls3_fc_pose_xyz': posenet.euc_loss3x, 'cls3_fc_pose_wpqr': posenet.euc_loss3q})

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
    
    start_time = time.time()
    testPredict = model.predict(X_test)
    print("Inference time : ",(time.time()-start_time)/len(dataset_test.images),"seconds")
    valsx = testPredict[4]
    valsq = testPredict[5]
    
    results = np.zeros((len(dataset_test.images),2))
    xzresults = np.zeros((len(dataset_test.images),2))
    max_angle = 0
    max_angle_index = 0
    
    for i in range(len(dataset_test.images)):

        pose_q= np.asarray(dataset_test.poses[i][3:7])
        pose_x= np.asarray(dataset_test.poses[i][0:3])
        predicted_x = valsx[i]
        predicted_q = valsq[i]

        pose_q = np.squeeze(pose_q)
        pose_x = np.squeeze(pose_x)
        predicted_q = np.squeeze(predicted_q)
        predicted_x = np.squeeze(predicted_x)
        
        x = float(pose_x[0])
        y = float(pose_x[1])
        z = float(pose_x[2])
        x += float(minx[0])
        y += float(miny[0])
        z += float(minz[0])
        
        px = float(predicted_x[0])
        py = float(predicted_x[1])
        pz = float(predicted_x[2])
        px += float(minx[0])
        py += float(miny[0])
        pz += float(minz[0])
    
        #Compute Individual Sample Error
        q1 = pose_q / np.linalg.norm(pose_q)
        q2 = predicted_q / np.linalg.norm(predicted_q)
        d = abs(np.sum(np.multiply(q1,q2)))
        theta = 2 * np.arccos(d) * 180/math.pi

        #Calculate angle using quat2AA(rpy ==> AA), decode_angle(AA ==> rodrigues ==> angle bw z-axis)
        rx,ry,rz = quat2AA(q1)
        ang_true = decode_angle(rx,ry,rz, mode="euclid")
        prx,pry,prz = quat2AA(q2)
        ang_pred = decode_angle(prx,pry,prz, mode="euclid")

        #recalculate angle using quaterions module
        qxz1 = np.array([quaternions.quat2mat(q1)[2][0], quaternions.quat2mat(q1)[2][2]])
        qxz2 = np.array([quaternions.quat2mat(q2)[2][0], quaternions.quat2mat(q2)[2][2]])
        t = np.abs(angle_between(qxz1, np.array([0, 1])) - angle_between(qxz2, np.array([0, 1])))

        error_x = np.linalg.norm(pose_x-predicted_x)
        results[i, :] = [error_x, theta]
        xzresults[i, :] = [math.sqrt((x - px) * (x - px) + (z - pz) * (z - pz)), min(np.abs(ang_true - ang_pred),np.abs(360 - ang_true + ang_pred))]
        if xzresults[i,1] > 180:
            xzresults[i,1] = 360 - xzresults[i,1]

        if xzresults[i,1] > max_angle:
            max_angle = xzresults[i,1]
            max_angle_index = i

        print('\nIteration:  ', i, '  Error XYZ (m):  ', error_x, '  Error Q (degrees):  ', theta)
        print("Angular function check : ", t, " VS ", np.abs(ang_true - ang_pred))
        print("(x, y, z), angle from Z-axis (XZ plane): ", x, y, z, ang_true)
        print("(x, y, z), angle from Z-axis (XZ plane): ",px, py, pz, ang_pred)
        print("QUATERNION REAL : ", q1)
        print("QUATERNION PRED : ", q2)

    # XZ results show the displacement and rotation in relation to the XZ-plane, not considering the y-axis at all. Rotation is in relation to Z-axis.
    median_result = np.median(results,axis=0)
    median_xzresults = np.median(xzresults, axis=0)
    print('\nMedian error ', median_result[0], 'm  and ', median_result[1], 'degrees.')
    print('XZ-Median error ', median_xzresults[0], 'm  and ', median_xzresults[1], 'degrees.')

    mean_result = np.mean(results,axis=0)
    mean_xzresults = np.mean(xzresults, axis=0)
    print('\nMean error ', mean_result[0], 'm  and ', mean_result[1], 'degrees.')
    print('XZ-Mean error ', mean_xzresults[0], 'm  and ', mean_xzresults[1], 'degrees.')

    min_result = np.min(results,axis=0)
    min_xzresults = np.min(xzresults, axis=0)
    print('\nMinimum error ', min_result[0], 'm  and ', min_result[1], 'degrees.')
    print('XZ-Minimum error ', min_xzresults[0], 'm  and ', min_xzresults[1], 'degrees.')

    max_result = np.max(results,axis=0)
    max_xzresults = np.max(xzresults, axis=0)
    print('\nMaximum error ', max_result[0], 'm  and ', max_result[1], 'degrees.')
    print('XZ-Maximum error ', max_xzresults[0], 'm  and ', max_xzresults[1], 'degrees.')
    print("Maximum angle at line #: ",max_angle_index+1)

    std_result = np.std(results,axis=0)
    std_xzresults = np.std(xzresults, axis=0)
    print('\nSTD error ', std_result[0], 'm  and ', std_result[1], 'degrees.')
    print('XZ-STD error ', std_xzresults[0], 'm  and ', std_xzresults[1], 'degrees.\n')

# Convert Predictions to get Location (x,z) and Angle in relation to x-z plane(theta)
#  ==>
# pose_x[0] += minx, pose_x[1] += miny, pose_x[2] += minz
# (x,z) = (pose_x[0], pose_x[2]). Divide by 114.6 if you want a value between 0 and 1.
# theta = decode_angle(quat2AA(pose_q), mode == "euclid")

