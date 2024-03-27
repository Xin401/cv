import numpy as np
import cv2

class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        gaussian_images = []

        for i in range(self.num_octaves):
            if i == 1:
                image = cv2.resize(gaussian_images[-1], (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
            for j in range(self.num_guassian_images_per_octave):
                if(j == 0):
                    gaussian_images.append(image)
                else:
                    gaussian_images.append(cv2.GaussianBlur(image, (0, 0), self.sigma**j))

        for i in range(len(gaussian_images)):
            cv2.imwrite('result/gaussian_images_'+str(i)+'.png', gaussian_images[i])
        
        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images = []
        for i in range(self.num_octaves):
            for j in range(self.num_DoG_images_per_octave):
                dog_images.append(cv2.subtract(gaussian_images[i*self.num_guassian_images_per_octave+j+1], gaussian_images[i*self.num_guassian_images_per_octave+j]))
        for i in range(len(dog_images)):
            cv2.imwrite('result/dog_images_'+str(i)+'.png', dog_images[i])

        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint
        keypoints = []
        for i in range(self.num_octaves):
            for j in range(1,self.num_DoG_images_per_octave-1):
                dog_image = dog_images[i*self.num_DoG_images_per_octave+j]
                dog_image_prev = dog_images[i*self.num_DoG_images_per_octave+j-1]
                dog_image_next = dog_images[i*self.num_DoG_images_per_octave+j+1]
                for y in range(1, dog_image.shape[0]-1):
                    for x in range(1, dog_image.shape[1]-1):
                        dog_stack = np.stack([dog_image_prev[y-1:y+2, x-1:x+2], dog_image[y-1:y+2, x-1:x+2], dog_image_next[y-1:y+2, x-1:x+2]])
                        if (dog_image[y,x] == np.max(dog_stack) or dog_image[y,x] == np.min(dog_stack)) and np.abs(dog_image[y,x]) > self.threshold:
                            keypoints.append((y*(1+i),x*(1+i)))

        # Step 4: Delete duplicate keypoints
        # - Function: np.unique s
        keypoints = np.unique(keypoints, axis=0)

        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))] 
        return keypoints
