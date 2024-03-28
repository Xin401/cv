import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter


def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/1_setting.txt', help='path to setting file')
    args = parser.parse_args()

    # Init
    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float32)
    cost = np.zeros(6)

    ### TODO ###

    # Read setting file
    fp = open(args.setting_path)
    setting = [i.replace('\n','').split(',') for i in fp.readlines()]
    RBG_setting = setting[1:6]
    sigma_s, sigma_r = int(setting[6][1]), float(setting[6][3])    
    # Create JBF class
    JBF = Joint_bilateral_filter(sigma_s, sigma_r)
    # JBF: RGB and Gray 
    bf_output = JBF.joint_bilateral_filter(img_rgb, img_rgb).astype(np.uint8)
    jbf_output = JBF.joint_bilateral_filter(img_rgb, img_gray).astype(np.uint8)
    # Calculate cost by L1 normaliztion
    cost[0]  = np.sum(np.abs(jbf_output.astype('int32')-bf_output.astype('int32')))
    # Save images
    if not os.path.exists('output'):
        os.mkdir("output")
    cv2.imwrite('output/gray_0.png',img_gray.astype(np.uint8))
    cv2.imwrite('output/bf.png', cv2.cvtColor(bf_output,cv2.COLOR_RGB2BGR).astype(np.uint8))
    cv2.imwrite('output/jbf_0.png', cv2.cvtColor(jbf_output,cv2.COLOR_RGB2BGR).astype(np.uint8))
    
    # Conver RBG to Gray by setting
    for i in range(len(RBG_setting)):
        rgb = img_rgb.copy()
        # Gray = wr * R + wg * G + wb * B
        img_gray = rgb[:,:,0] * float(RBG_setting[i][0]) + \
                   rgb[:,:,1] * float(RBG_setting[i][1]) + \
                   rgb[:,:,2] * float(RBG_setting[i][2])
        # JBF: RGB and Gray_Setting
        jbf_output = JBF.joint_bilateral_filter(img_rgb, img_gray).astype(np.uint8)
        # Calculate cost by L1 normaliztion
        cost[i+1]  = np.sum(abs(np.abs(jbf_output.astype('int32')-bf_output.astype('int32'))))
        # save images
        cv2.imwrite('output/gray_' + str(i+1) + '.png',img_gray.astype(np.uint8))
        cv2.imwrite('output/jbf_' + str(i+1) + '.png', cv2.cvtColor(jbf_output,cv2.COLOR_RGB2BGR).astype(np.uint8))

    # Show the Total Cost
    print('Total Cost:',cost,'\nMin Cost:' ,np.argmin(cost), 'Max Cost:',np.argmax(cost))

if __name__ == '__main__':
    main()