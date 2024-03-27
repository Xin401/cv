import numpy as np
import cv2


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
    
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)

        ### TODO ###
        padded_guidance = padded_guidance.astype(np.float64) / 255.0
        guidance = guidance.astype(np.float64) / 255.0

        spatial_kernel = np.zeros((self.wndw_size, self.wndw_size))
        for x in range(self.wndw_size):
            for y in range(self.wndw_size):
                spatial_kernel[x, y] = np.exp(-((x - self.wndw_size//2)**2 + (y - self.wndw_size//2)**2) / (2 * self.sigma_s**2))


        output = np.zeros_like(img)


        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                window_img = padded_img[i:i+self.wndw_size, j:j+self.wndw_size, :]
                window_guidance = padded_guidance[i:i+self.wndw_size, j:j+self.wndw_size]

                if guidance.ndim == 3:
                    range_kernel = np.exp(-((window_guidance - guidance[i,j])**2).sum(axis=2) /  (2*self.sigma_r**2))
                else:
                    range_kernel = np.exp(-((window_guidance - guidance[i,j])**2) / (2*self.sigma_r**2))

                jbf = spatial_kernel * range_kernel

                output[i, j, :] = np.sum(window_img * jbf[:, :, np.newaxis], axis=(0, 1)) / np.sum(jbf)


        return np.clip(output, 0, 255).astype(np.uint8)
    

    