import cv2
import numpy as np

colors = [[0.76590096, 0.0266074, 0.9806378],
           [0.54197179, 0.81682527, 0.95081629],
           [0.0799733, 0.79737015, 0.15173816],
           [0.93240442, 0.8993321, 0.09901344],
           [0.73130136, 0.05366301, 0.98405681],
           [0.01664966, 0.16387004, 0.94158259],
           [0.54197179, 0.81682527, 0.45081629],
           # [0.92074915, 0.09919099 ,0.97590748],
           [0.83445145, 0.97921679, 0.12250426],
           [0.7300924, 0.23253621, 0.29764521],
           [0.3856775, 0.94859286, 0.9910683],  # 10
           [0.45762137, 0.03766411, 0.98755338],
           [0.99496697, 0.09113071, 0.83322314],
           [0.96478873, 0.0233309, 0.13149931],
           [0.33240442, 0.9993321 , 0.59901344],
            # [0.77690519,0.81783954,0.56220024],
           # [0.93240442, 0.8993321, 0.09901344],
           [0.95815068, 0.88436046, 0.55782268],
           [0.03728425, 0.0618827, 0.88641827],
           [0.05281129, 0.89572238, 0.08913828],

           ]



def draw_landmarks_regress_test(pts0, ori_image_regress, ori_image_points):
    for i, pt in enumerate(pts0):
        # color = np.random.rand(3)
        color = colors[i]
        # print(i+1, color)
        color_255 = (255 * color[0], 255 * color[1], 255 * color[2])
        cv2.circle(ori_image_regress, (int(pt[0]), int(pt[1])), 6, color_255, -1, 1)
        # cv2.circle(ori_image, (int(pt[2]), int(pt[3])), 5, color_255, -1,1)
        # cv2.circle(ori_image, (int(pt[4]), int(pt[5])), 5, color_255, -1,1)
        # cv2.circle(ori_image, (int(pt[6]), int(pt[7])), 5, color_255, -1,1)
        # cv2.circle(ori_image, (int(pt[8]), int(pt[9])), 5, color_255, -1,1)
        cv2.arrowedLine(ori_image_regress, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), color_255, 2, 1,
                        tipLength=0.2)
        cv2.arrowedLine(ori_image_regress, (int(pt[0]), int(pt[1])), (int(pt[4]), int(pt[5])), color_255, 2, 1,
                        tipLength=0.2)
        cv2.arrowedLine(ori_image_regress, (int(pt[0]), int(pt[1])), (int(pt[6]), int(pt[7])), color_255, 2, 1,
                        tipLength=0.2)
        cv2.arrowedLine(ori_image_regress, (int(pt[0]), int(pt[1])), (int(pt[8]), int(pt[9])), color_255, 2, 1,
                        tipLength=0.2)
        cv2.putText(ori_image_regress, '{}'.format(i + 1),
                    (int(pt[4] + 10), int(pt[5] + 10)),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.2,
                    color_255,  # (255,255,255),
                    1,
                    1)
        # cv2.circle(ori_image, (int(pt[0]), int(pt[1])), 6, (255,255,255), -1,1)
        cv2.circle(ori_image_points, (int(pt[2]), int(pt[3])), 5, color_255, -1, 1)
        cv2.circle(ori_image_points, (int(pt[4]), int(pt[5])), 5, color_255, -1, 1)
        cv2.circle(ori_image_points, (int(pt[6]), int(pt[7])), 5, color_255, -1, 1)
        cv2.circle(ori_image_points, (int(pt[8]), int(pt[9])), 5, color_255, -1, 1)
    return ori_image_regress, ori_image_points



def draw_landmarks_pre_proc(out_image, pts):
    for i in range(17):
        pts_4 = pts[4 * i:4 * i + 4, :]
        color = colors[i]
        color_255 = (255 * color[0], 255 * color[1], 255 * color[2])
        cv2.circle(out_image, (int(pts_4[0, 0]), int(pts_4[0, 1])), 5, color_255, -1, 1)
        cv2.circle(out_image, (int(pts_4[1, 0]), int(pts_4[1, 1])), 5, color_255, -1, 1)
        cv2.circle(out_image, (int(pts_4[2, 0]), int(pts_4[2, 1])), 5, color_255, -1, 1)
        cv2.circle(out_image, (int(pts_4[3, 0]), int(pts_4[3, 1])), 5, color_255, -1, 1)
    return np.uint8(out_image)


def draw_regress_pre_proc(out_image, pts):
    for i in range(17):
        pts_4 = pts[4 * i:4 * i + 4, :]
        pt = np.mean(pts_4, axis=0)
        color = colors[i]
        color_255 = (255 * color[0], 255 * color[1], 255 * color[2])
        cv2.arrowedLine(out_image, (int(pt[0]), int(pt[1])), (int(pts_4[0, 0]), int(pts_4[0, 1])), color_255, 2, 1,
                        tipLength=0.2)
        cv2.arrowedLine(out_image, (int(pt[0]), int(pt[1])), (int(pts_4[1, 0]), int(pts_4[1, 1])), color_255, 2, 1,
                        tipLength=0.2)
        cv2.arrowedLine(out_image, (int(pt[0]), int(pt[1])), (int(pts_4[2, 0]), int(pts_4[2, 1])), color_255, 2, 1,
                        tipLength=0.2)
        cv2.arrowedLine(out_image, (int(pt[0]), int(pt[1])), (int(pts_4[3, 0]), int(pts_4[3, 1])), color_255, 2, 1,
                        tipLength=0.2)
        cv2.putText(out_image, '{}'.format(i + 1), (int(pts_4[1, 0] + 10), int(pts_4[1, 1] + 10)),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, color_255, 1, 1)
    return np.uint8(out_image)