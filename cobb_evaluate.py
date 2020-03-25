###########################################################################################
## This code is transfered from matlab version of the MICCAI challenge
## Oct 1 2019
###########################################################################################
import numpy as np
import cv2


def is_S(mid_p_v):
    # mid_p_v:  34 x 2
    ll = []
    num = mid_p_v.shape[0]
    for i in range(num-2):
        term1 = (mid_p_v[i, 1]-mid_p_v[num-1, 1])/(mid_p_v[0, 1]-mid_p_v[num-1, 1])
        term2 = (mid_p_v[i, 0]-mid_p_v[num-1, 0])/(mid_p_v[0, 0]-mid_p_v[num-1, 0])
        ll.append(term1-term2)
    ll = np.asarray(ll, np.float32)[:, np.newaxis]   # 32 x 1
    ll_pair = np.matmul(ll, np.transpose(ll))        # 32 x 32
    a = sum(sum(ll_pair))
    b = sum(sum(abs(ll_pair)))
    if abs(a-b)<1e-4:
        return False
    else:
        return True

def cobb_angle_calc(pts, image):
    pts = np.asarray(pts, np.float32)   # 68 x 2
    h,w,c = image.shape
    num_pts = pts.shape[0]   # number of points, 68
    vnum = num_pts//4-1

    mid_p_v = (pts[0::2,:]+pts[1::2,:])/2   # 34 x 2
    mid_p = []
    for i in range(0, num_pts, 4):
        pt1 = (pts[i,:]+pts[i+2,:])/2
        pt2 = (pts[i+1,:]+pts[i+3,:])/2
        mid_p.append(pt1)
        mid_p.append(pt2)
    mid_p = np.asarray(mid_p, np.float32)   # 34 x 2

    for pt in mid_p:
        cv2.circle(image,
                   (int(pt[0]), int(pt[1])),
                   12, (0,255,255), -1, 1)

    for pt1, pt2 in zip(mid_p[0::2,:], mid_p[1::2,:]):
        cv2.line(image,
                 (int(pt1[0]), int(pt1[1])),
                 (int(pt2[0]), int(pt2[1])),
                 color=(0,0,255),
                 thickness=5, lineType=1)

    vec_m = mid_p[1::2,:]-mid_p[0::2,:]           # 17 x 2
    dot_v = np.matmul(vec_m, np.transpose(vec_m)) # 17 x 17
    mod_v = np.sqrt(np.sum(vec_m**2, axis=1))[:, np.newaxis]    # 17 x 1
    mod_v = np.matmul(mod_v, np.transpose(mod_v)) # 17 x 17
    cosine_angles = np.clip(dot_v/mod_v, a_min=0., a_max=1.)
    angles = np.arccos(cosine_angles)   # 17 x 17
    pos1 = np.argmax(angles, axis=1)
    maxt = np.amax(angles, axis=1)
    pos2 = np.argmax(maxt)
    cobb_angle1 = np.amax(maxt)
    cobb_angle1 = cobb_angle1/np.pi*180
    flag_s = is_S(mid_p_v)
    if not flag_s: # not S
        # print('Not S')
        cobb_angle2 = angles[0, pos2]/np.pi*180
        cobb_angle3 = angles[vnum, pos1[pos2]]/np.pi*180
        cv2.line(image,
                 (int(mid_p[pos2 * 2, 0] ), int(mid_p[pos2 * 2, 1])),
                 (int(mid_p[pos2 * 2 + 1, 0]), int(mid_p[pos2 * 2 + 1, 1])),
                 color=(0, 255, 0), thickness=5, lineType=2)
        cv2.line(image,
                 (int(mid_p[pos1[pos2] * 2, 0]), int(mid_p[pos1[pos2] * 2, 1])),
                 (int(mid_p[pos1[pos2] * 2 + 1, 0]), int(mid_p[pos1[pos2] * 2 + 1, 1])),
                 color=(0, 255, 0), thickness=5, lineType=2)

    else:
        if (mid_p_v[pos2*2, 1]+mid_p_v[pos1[pos2]*2,1])<h:
            # print('Is S: condition1')
            angle2 = angles[pos2,:(pos2+1)]
            cobb_angle2 = np.max(angle2)
            pos1_1 = np.argmax(angle2)
            cobb_angle2 = cobb_angle2/np.pi*180

            angle3 = angles[pos1[pos2], pos1[pos2]:(vnum+1)]
            cobb_angle3 = np.max(angle3)
            pos1_2 = np.argmax(angle3)
            cobb_angle3 = cobb_angle3/np.pi*180
            pos1_2 = pos1_2 + pos1[pos2]-1

            cv2.line(image,
                     (int(mid_p[pos1_1 * 2, 0]), int(mid_p[pos1_1 * 2, 1])),
                     (int(mid_p[pos1_1 * 2+1, 0]), int(mid_p[pos1_1 * 2 + 1, 1])),
                     color=(0, 255, 0), thickness=5, lineType=2)

            cv2.line(image,
                     (int(mid_p[pos1_2 * 2, 0]), int(mid_p[pos1_2 * 2, 1])),
                     (int(mid_p[pos1_2 * 2+1, 0]), int(mid_p[pos1_2 * 2 + 1, 1])),
                     color=(0, 255, 0), thickness=5, lineType=2)

        else:
            # print('Is S: condition2')
            angle2 = angles[pos2,:(pos2+1)]
            cobb_angle2 = np.max(angle2)
            pos1_1 = np.argmax(angle2)
            cobb_angle2 = cobb_angle2/np.pi*180

            angle3 = angles[pos1_1, :(pos1_1+1)]
            cobb_angle3 = np.max(angle3)
            pos1_2 = np.argmax(angle3)
            cobb_angle3 = cobb_angle3/np.pi*180

            cv2.line(image,
                     (int(mid_p[pos1_1 * 2, 0]), int(mid_p[pos1_1 * 2, 1])),
                     (int(mid_p[pos1_1 * 2+1, 0]), int(mid_p[pos1_1 * 2 + 1, 1])),
                     color=(0, 255, 0), thickness=5, lineType=2)

            cv2.line(image,
                     (int(mid_p[pos1_2 * 2, 0]), int(mid_p[pos1_2 * 2, 1])),
                     (int(mid_p[pos1_2 * 2+1, 0]), int(mid_p[pos1_2 * 2 + 1, 1])),
                     color=(0, 255, 0), thickness=5, lineType=2)

    return [cobb_angle1, cobb_angle2, cobb_angle3]