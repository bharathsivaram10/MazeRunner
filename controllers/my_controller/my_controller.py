from controller import Motor, Camera, CameraRecognitionObject, Supervisor
import numpy as np
from numpy import inf
import matplotlib.pyplot as plt
import math
import scipy.cluster.hierarchy as hcluster
import time

# helper functions
def omegaToWheelSpeeds(omega, v):
    wd = omega * axleLength * 0.5
    return v - wd, v + wd

def rotMat(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def findLines(dists):

    fov = 3.14

    theta_is = np.zeros(512)
    for i in range(len(dists)):
        theta_is[i] = i * (fov / 512)


    theta_js = np.zeros(90)
    for j in range(90):
        theta_js[j] = j * (fov / 90)

    dists[dists == inf] = 0
    max_dist = np.amax(dists)

    dist_js = np.zeros(100)
    for j in range(100):
        if j <= 50:
            dist_js[j] = (50-j) * (-1 * max_dist / 50)
        else:
            dist_js[j] = (j-50) * (max_dist / 50)


    accum = np.zeros((len(dist_js), len(theta_js)))

    for i in range(len(dists)):
        theta_i = theta_is[i]
        for j in range(len(theta_js)):
            dj = dists[i] * math.cos(theta_js[j] - theta_i)
            dj_id = find_nearest(dist_js, dj)
            accum[dj_id, j] += 1

    rows, cols = np.where(accum>=70)

    # plt.imshow(accum)
    # plt.colorbar()
    # plt.show()

    final = np.zeros((2,len(rows)))
    for i in range(len(rows)):
        final[0][i] = dist_js[rows[i]]
        final[1][i] = theta_js[cols[i]]
    final = final.T
    return final

# Task: Finish EKFPropagate and EKFRelPosUpdate
def EKFPropagate(x_hat_t, # robot position and orientation
                 Sigma_x_t, # estimation uncertainty
                 u, # control signals
                 Sigma_n, # uncertainty in control signals
                 dt # timestep
    ):
    # TODO: Calculate the robot state estimation and variance for the next timestep

    x = x_hat_t[0] + u[0] * np.cos(x_hat_t[2]) * dt
    y = x_hat_t[1] + u[0] * np.sin(x_hat_t[2]) * dt
    theta = x_hat_t[2] + u[1] * dt
    x_hat_t = np.array([x, y, theta])

    phi = np.asarray([[1, 0, -v * np.sin(x_hat_t[2]) * dt], [0,1,v * np.cos(x_hat_t[2]) * dt], [0,0,1]])
    G = np.asarray([[np.cos(x_hat_t[2]) * dt, 0], [np.sin(x_hat_t[2]) * dt, 0], [0,dt]])

    Sigma_x_t = phi @ Sigma_x_t @ phi.T + G @ Sigma_n @ G.T

    return x_hat_t, Sigma_x_t

def EKFRelPosUpdate(x_hat_t, # robot position and orientation
                    Sigma_x_t, # estimation uncertainty
                    z, # measurements
                    Sigma_m, # measurements' uncertainty
                    G_p_L, # landmarks' global positions
                    dt # timestep
                   ):
    # TODO: Update the robot state estimation and variance based on the received measurement

    x_hat_t = np.asarray(x_hat_t)
    Sigma_x_t = np.asarray(Sigma_x_t)
    z = np.asarray(z)
    Sigma_m = np.asarray(Sigma_m)
    G_p_L = np.asarray(G_p_L)
    dt = np.asarray(dt)
    pos = np.asarray(x_hat_t[0:2])
    theta = np.asarray(x_hat_t[2])

    z_hat = rotMat(theta).T @ (np.array(G_p_L)[0:2]-pos)
    r = z-z_hat
    top = -rotMat(theta).T
    bottom = (-rotMat(theta).T @ np.array([[0,-1],[1,0]]) @ (np.array(G_p_L)[0:2]-pos)).reshape((2,1))
    H = np.hstack((top,bottom))

    S = (H @ Sigma_x_t @ H.T) + Sigma_m
    K = (Sigma_x_t @ H.T @ np.linalg.inv(S))
    x_hat_t = x_hat_t + (K @ r)
    Sigma_x_t = Sigma_x_t - (Sigma_x_t @ H.T @ np.linalg.inv(S) @ H @ Sigma_x_t)

    return x_hat_t, Sigma_x_t

def posOfImgToBearing(x, # landmark's horizontal position on image
                      w, # image width
                      fov): # the camera's field of view
    # TODO: Calculate the bearing measurement based on landmark's position on image
    d = (0.5 * w)/np.tan(0.5 * fov)
    bear = np.arctan2(0.5*w-x,d)

    return bear

def rotMat(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


wheelRadius = 0.0205
axleLength = 0.053 # Data from Webots website seems wrong. The real axle length should be about 56-57mm
max_vel = 6.28


# create the Robot instance.
robot = Supervisor()
camera = robot.getDevice('camera')
camera.enable(1)

if camera.hasRecognition():
    camera.recognitionEnable(1)
    camera.enableRecognitionSegmentation()
else:
    print("Your camera does not have recognition")

timestep = int(robot.getBasicTimeStep())
leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))
rightMotor.setVelocity(0)
leftMotor.setVelocity(0)

# Create position sensor object
left_ps = robot.getDevice('left_wheel_sensor')
left_ps.enable(timestep)
right_ps = robot.getDevice('right_wheel_sensor')
right_ps.enable(timestep)

# Enable point cloud
lidar = robot.getDevice('lidar')
lidar.enable(1)
lidar.enablePointCloud()


# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())


robotNode = robot.getFromDef("e-puck")

diff = 100

wall_thresh = 0.1

#0 default (rob turns towards goal, then goes forward)
#1 wall turn (wall inbetween rob and goal so it gets parallel with wall)
#2 wall forward (go along wall)
#3 ball (rob sees two balls on camera and uses to SLAM)
mode = 0

mode_dir = -1
flag = False

start_pos = robotNode.getPosition()
start_orient = robotNode.getOrientation()

x_s = start_pos[0]
y_s = start_pos[1]
orient = np.arctan2(start_orient[3],start_orient[0])
if orient < 0:
    orient += 2*np.pi
last_theta_s = np.arctan2(start_orient[3],start_orient[0])
if last_theta_s < 0:
    last_theta_s += 2*np.pi

x_s_est = start_pos[0]
y_s_est = start_pos[1]


dt = 0.032

ps_vals = [0,0]
dist_vals = [0,0]
last_ps_vals = [0,0]
differs = [0,0]


#EKF variables
Sigma_n = np.zeros((2,2))
std_n_v = 0.01
std_n_omega = np.pi/60
Sigma_n[0,0] = std_n_v * std_n_v
Sigma_n[1,1] = std_n_omega * std_n_omega

updateFreq = 200

flag3 = False

goal_pos = [-2.9, 0.747, 0.0]


# Main loop:
while robot.step(timestep) != -1:

    # orientation code
    start_pos = robotNode.getPosition()
    start_orient = robotNode.getOrientation()
    vel = robotNode.getVelocity()

    theta_s = np.arctan2(start_orient[3],start_orient[0])

    if theta_s < 0:
        theta_s += 2*np.pi

    ps_vals[0]=left_ps.getValue()
    ps_vals[1]=right_ps.getValue()

    for ind in range(2):
        differ = ps_vals[ind]-last_ps_vals[ind]
        if differ < 0.001:
            differ = 0
            ps_vals[ind] = last_ps_vals[ind]
        dist_vals[ind] = differ * wheelRadius

    # for ind in range(2):
    #     differ = ps_vals[ind]-last_ps_vals[ind]
    #     if differ < 0.01:
    #         differ = 0
    #         ps_vals[ind] = last_ps_vals[ind]
    #     differs[ind] = differ


    # v = (differs[0]+differs[1])/2.0
    # w = (differs[1]-differs[0])/axleLength

    v = (dist_vals[0] + dist_vals[1])/ (2.0)
    w = (dist_vals[1] - dist_vals[0])/ (axleLength)

    # orient += w

    # if orient < 0:
    #     orient += 2*np.pi

    diff_theta = theta_s - last_theta_s
    
    # x_s_est += v*np.cos(orient+w/2)
    # y_s_est += v*np.sin(orient+w/2)


    # x_s_est += v*np.cos(theta_s+diff_theta/2)
    # y_s_est += v*np.sin(theta_s+diff_theta/2)

    x_s_est += vel[0]*dt
    y_s_est += vel[1]*dt

    orient += vel[5]*dt

    if orient < 0:
        orient += 2*np.pi

    for ind in range(2):
        last_ps_vals[ind] = ps_vals[ind]

    x_s = start_pos[0]
    y_s = start_pos[1]

    rot = rotMat(orient)

    #print('Diff in pos',np.array([x_s, y_s,theta_s])-np.array([x_s_est, y_s_est,orient]))

    y = goal_pos[1]-y_s
    x = goal_pos[0]-x_s

    theta = np.arctan2(y,x)  % (2*np.pi)

    diff = theta-theta_s

    # line code
    lidar_dat = np.array(lidar.getRangeImage())
    lidar_dat_copy = np.copy(lidar_dat)
    lines = findLines(lidar_dat_copy)

    thresh = 0.05
    clusters = hcluster.fclusterdata(lines, thresh, criterion="distance")

    a = np.hstack((lines,clusters[:,None]))
    a = a[a[:, -1].argsort()]
    a = np.split(a[:,:-1], np.unique(a[:, -1], return_index=True)[1][1:])


    centroids = []
    for i in range(len(a)):
        m = np.mean(a[i], axis=0)
        centroids.append(m)


    b = np.asarray(centroids)

    #check for wall
    for i in range(len(centroids)):
        mean_range = lidar_dat[226:286]

        mean_range[mean_range == inf] = 1
        mean_range = np.mean(mean_range)

        if np.abs(centroids[i][0]) <= wall_thresh and np.abs(centroids[i][0]) != 0.0 and mean_range <= 0.5:
            if np.abs(diff) < 0.05:
                mode = 1
                close_wall = np.abs(centroids[i])

                rightMotor.setVelocity(0)
                leftMotor.setVelocity(0)

                break

    #default
    if mode == 0:
        if np.abs(diff) > 0.05:
            print('Mode0: Rot')

            if diff > 0:
                rightMotor.setVelocity(1.2)
                leftMotor.setVelocity(-1.2)

            if diff < 0:
                rightMotor.setVelocity(-1.2)
                leftMotor.setVelocity(1.2)
        elif np.abs(diff) <= 0.05:
            print('Mode0: Forward')
            rightMotor.setVelocity(max_vel)
            leftMotor.setVelocity(max_vel)
    elif mode == 1:
        print("wall turn mode")

        diff_par = close_wall[1]-theta_s % np.pi

        if np.abs(diff_par) >= 0.03:

            zero = lidar_dat[0:30]
            end = lidar_dat[480:511]

            zero[zero == inf] = 100
            end[end == inf] = 100

            zero = np.mean(zero)
            end = np.mean(end)

            if zero < end and mode_dir == -1:
                mode_dir = 1

            if mode_dir == 1 :

                # Turn right
                rightMotor.setVelocity(-1)
                leftMotor.setVelocity(1)

            else:
                #Turn left
                mode_dir = 0
                rightMotor.setVelocity(1)
                leftMotor.setVelocity(-1)

        else:
            rightMotor.setVelocity(0)
            leftMotor.setVelocity(0)

            mode = 2
            mode_dir = -1

    elif mode == 2:
        print("wall forward mode")
        rightMotor.setVelocity(6)
        leftMotor.setVelocity(6)

        centroids_arr = np.asarray(centroids)
        centroids_arr[centroids_arr == 0] = 100

        min = np.argmin(np.abs(centroids_arr[:,0]))

        if (np.abs(centroids_arr[min][0]) >= close_wall[0]+0.05) and flag == False:
            flag = True
            iter = 0

        elif flag == True:
            iter += 1
            print(iter)
            if iter == 40:
                mode = 0
                flag = False

    elif mode == 3:
        print("SLAM MODE")

        if flag_slam == True:
            iter_slam += 1
            print(iter_slam)
            if iter_slam == 10:
                flag_slam = False

        else:

            print("LETS GO")

            robot_vel = np.copy(robotNode.getVelocity())
            v_ekf = np.linalg.norm(robot_vel[:2])
            omega = robot_vel[5]
            u = np.array([v_ekf, omega])

            # EKF Propergate
            x_hat_t, Sigma_x_t = EKFPropagate(x_hat_t, Sigma_x_t, u, Sigma_n, dt)

            # EKF Update
            recObjs = camera.getRecognitionObjects()
            recObjsNum = camera.getRecognitionNumberOfObjects()
            z_pos = np.zeros((recObjsNum, 2)) # relative position measurements
            furthest_dist = -1 #id
            furthest_pos = -1
            furthest_bear = -1


            if counter % updateFreq == 0:
                for i in range(0, recObjsNum):

                    landmark = robot.getFromId(recObjs[i].get_id())
                    G_p_L = landmark.getPosition()

                    bear = posOfImgToBearing(recObjs[i].get_position_on_image()[0], 320, 0.84)

                    lidar_bear = np.round(256 - (bear / np.pi) * 512)

                    lidar_dat_2 = np.array(lidar.getRangeImage())
                    lidar_dist = lidar_dat_2[int(lidar_bear)]


                    if lidar_dist == inf:
                        pass

                    else:
                        # print(rot.shape)

                        land = np.array([lidar_dist * np.cos(bear),lidar_dist * np.sin(bear)])

                        trans = rot @ land + x_hat_t[:2]
                        trans = np.append(trans,0.05)

                        G_p_L_2 = [trans[0], trans[1], 0.05]

                        # print("info")
                        print('x_hat_t',x_hat_t)

                        if lidar_dist > furthest_dist:
                            furthest_dist = lidar_dist
                            furthest_pos = recObjs[i].get_position()
                            furthest_bear = bear

                        rel_lm_trans = landmark.getPose(robotNode)

                        # print(rel_lm_trans[3],land[0],rel_lm_trans[7],land[1])

                        std_m = 0.05
                        Sigma_m = [[std_m*std_m, 0], [0,std_m*std_m]]
                        #z_pos[i] = [rel_lm_trans[3]+np.random.normal(0,std_m), rel_lm_trans[7]+np.random.normal(0,std_m)]

                        z_pos[i] = [land[0]+np.random.normal(0,std_m), land[1]+np.random.normal(0,std_m)]

                        x_hat_t, Sigma_x_t = EKFRelPosUpdate(x_hat_t, Sigma_x_t, z_pos[i], Sigma_m, G_p_L_2, dt)

                
            sphere_pos = furthest_pos
            d = furthest_dist

            gain_d = 5.0
            gain_a = 30.0

            if d <= 0.05:
                leftMotor.setVelocity(1.0)
                rightMotor.setVelocity(1.0)
            else:

                g = gain_d * d

                if g <= 6.28:
                    v = g
                else:
                    v = 6.28

                goal_angle = math.atan((sphere_pos[1]) / (sphere_pos[0]))

                angle_delta = ((goal_angle + math.pi) % (2 * math.pi)) - math.pi

                w = gain_a * angle_delta

                left_prelim = v - (w * 0.052 * 0.5)
                right_prelim = v + (w * 0.052 * 0.5)

                if left_prelim <= 6.28:
                    v_left = left_prelim
                else:
                    v_left = 6.28

                if right_prelim <= 6.28:
                    v_right = right_prelim
                else:
                    v_right = 6.28

                leftMotor.setVelocity(v_left)
                rightMotor.setVelocity(v_right)

            # print('x_hat_t',x_hat_t)


            if recObjsNum == 1:
                flag3 = True
                mode = 0

    objs = camera.getRecognitionObjects()



    if len(objs) >= 3 and mode != 3 and flag3 == False:
        rightMotor.setVelocity(6)
        leftMotor.setVelocity(6)

        print("GOING TO MODE 3 -------------------- #################")

        counter = 0
        # orientation code (MOVE OUTSIDE LOOP)
        start_pos = robotNode.getPosition()
        start_orient = robotNode.getOrientation()

        x_s = start_pos[0]
        y_s = start_pos[1]

        theta_s = np.arctan2(start_orient[3],start_orient[0])
        if theta_s < 0:
            theta_s += 2*np.pi

        x_hat_t = [x_s, y_s, theta_s]

        # x_hat_t = [x_s_est,y_s_est,orient]

        Sigma_x_t = np.zeros((3,3))
        Sigma_x_t[0,0], Sigma_x_t[1,1], Sigma_x_t[2,2] = 0.01, 0.01, np.pi/90

        flag_slam = True
        iter_slam = 0

        mode = 3



    pass

# Enter here exit cleanup code.
