import numpy as np
import cv2
import argparse
from DoG import Difference_of_Gaussian


def plot_keypoints(img_gray, keypoints, save_path):
    img = np.repeat(np.expand_dims(img_gray, axis = 2), 3, axis = 2)
    for y, x in keypoints:
        cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
    cv2.imwrite(save_path, img)

def main():
    parser = argparse.ArgumentParser(description='main function of Difference of Gaussian')
    parser.add_argument('--threshold', default=3.0, type=float, help='threshold value for feature selection')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    parser.add_argument('--gt_path', default='./testdata/1_gt.npy', help='path to ground truth .npy')
    args = parser.parse_args()

    print('Processing %s ...'%args.image_path)
    img = cv2.imread(args.image_path, 0).astype(np.float64)

    ### TODO ###
    DoG = Difference_of_Gaussian(args.threshold)
    keypoints_gt = np.load(args.gt_path)
    keypoints = DoG.get_keypoints(img)
    plot_keypoints(img, keypoints_gt, 'result/keypoint_gt.png')
    plot_keypoints(img, keypoints, 'result/keypoint.png')
    np.savetxt('result/keypoint.txt', keypoints)
    np.savetxt('result/keypoint_gt.txt', keypoints_gt)
if __name__ == '__main__':
    main()