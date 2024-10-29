import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import logging  # Import logging module

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PanaromaStitcher():
    def __init__(self):
        pass  
    
    def make_panaroma_for_images_in(self, path):
        # Get all images in the specified directory
        imf = path
        all_image_paths = sorted(glob.glob(imf + os.sep + '*'))  # Use glob to find all image files
        
        images = [self.load_and_resize(image_path) for image_path in all_image_paths]

        logging.info(f'Loaded {len(images)} images for stitching')

        homography_matrixlist = []  # Initialize list to store homography matrices

        # Initialize stitching with the first image outside the loop
        stitched_image = self.BGR2RGB(images[0])  # Start with the first image in the list

        logging.info('Initialized aligned_image with the first image') 

        # Process images in pairs and stitch them together
        
        for i in range(1, len(images)):
                        
            # Current image to stitch
            current_image = self.BGR2RGB(images[i])
            logging.info(f'Stitching image {i +1} to the aligned image')

            # Obtain matching points between stitched_image (left image) and current_image (right image)
            src_pts, dst_pts, matching_output = self.obtain_src_dst_points(stitched_image, current_image)
            logging.info('Obtained source and destination points for feature matching for image pair: {}'.format(i))
            
            self.display_image(matching_output, width=20, height=10)
            logging.info('Computing homography for image pair: {}'.format(i))

            # Estimate homography with RANSAC
            H = self.estimate_homography_with_ransac(src_pts, dst_pts)
            homography_matrixlist.append(H)  # Append the computed homography matrix to list
            logging.info('Estimated homography with RANSAC')

            # Transform the stitched_image by aligning current_image using homography H
            stitched_image = self.transform_images(stitched_image, current_image, H)
            logging.info('Transformed and blended the images')

            logging.info(f'Stitched {i + 1} images')
            self.display_image(stitched_image, width=20, height=10)
            
        logging.info('Stitching completed.')
        self.display_image(stitched_image, width=20, height=10)

        return stitched_image, homography_matrixlist 

    # Function to load and resize an image
    def load_and_resize(self, image_path, width=600, height=400):
        img = cv2.imread(image_path)  # Load the image from file
        if img is None:
            raise ValueError(f"Image at {image_path} could not be loaded.")
        img = cv2.resize(img, (width, height))  # Resize to the specified dimensions
        return img  # Return the resized image
    

    # Function to display an image
    def display_image(self,img, width=0, height=0, show_axis=False, img_title=''):
        if width == 0 or height == 0:
            plt.figure()  # Create a new figure if dimensions are not provided
        else:
            plt.figure(figsize=(width, height))  # Set figure size
        plt.title(img_title)  # Add title to the figure
        plt.imshow(img)  # Render the image
        if show_axis == False: plt.axis('off')  # Hide the axis
        plt.show()  # Display the figure

    # Function to convert BGR image to RGB format
    def BGR2RGB(self,img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert color space

    # Function to extract keypoints and descriptors
    def extract_keypoints_descriptors(self,image):
        sift_detector = cv2.SIFT_create()  # Initialize SIFT detector
        keypoints, descriptors = sift_detector.detectAndCompute(image, None)  # Extract keypoints and descriptors
        descriptors = descriptors.astype(np.uint8)  # Convert descriptors to uint8
        return keypoints, descriptors  # Return keypoints and descriptors

    # Function to visualize keypoints
    def visualize_keypoints(self,image, keypoints):
        return cv2.drawKeypoints(image, keypoints, None)  # Render keypoints on the image

    # Function to obtain source and destination points
    def obtain_src_dst_points(self,image1, image2):
        gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        keypoints1, descriptors1 = self.extract_keypoints_descriptors(gray_image1)  # Get keypoints and descriptors
        keypoints2, descriptors2 = self.extract_keypoints_descriptors(gray_image2)  # Get keypoints and descriptors
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # Create BFMatcher
        matches = matcher.match(descriptors1, descriptors2)  # Match descriptors
        matches = sorted(matches, key=lambda x: x.distance)  # Sort matches by distance
        # Draw matches between the two images
        matching_output = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, outImg=None)
        # Get coordinates of matched points
        src_points = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        dst_points = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
        return src_points, dst_points, matching_output  # Return source and destination points and matching result

    # Function to calculate the homography matrix
    def calculate_homography(self,source_points, destination_points):
        num_points = source_points.shape[0]  # Retrieve the number of points
        A_matrix = []  # Initialize the matrix A for homography computation
        for i in range(num_points):
            src_x, src_y = source_points[i, 0], source_points[i, 1]
            dst_x, dst_y = destination_points[i, 0], destination_points[i, 1]
            # Constructing the rows of the A matrix
            A_matrix.append([src_x, src_y, 1, 0, 0, 0, -dst_x * src_x, -dst_x * src_y, -dst_x])
            A_matrix.append([0, 0, 0, src_x, src_y, 1, -dst_y * src_x, -dst_y * src_y, -dst_y])
        A_matrix = np.asarray(A_matrix)  # Convert the matrix to a numpy array
        U, S, Vh = np.linalg.svd(A_matrix)  # Execute Singular Value Decomposition
        L = Vh[-1, :] / Vh[-1, -1]  # Determine the homography
        H = L.reshape(3, 3)  # Reshape to obtain the homography matrix
        return H  # Return the computed homography

    def estimate_homography_with_ransac(self,src_pts, dst_pts, iterations=1000, threshold=0.5):
        num_points = src_pts.shape[0]  # Get the number of source points
        optimal_H = None  # Initialize the optimal homography matrix
        max_inlier_count = 0  # Initialize the maximum inlier count

        for _ in range(iterations):  # Iterate for the specified number of iterations
            sampled_indices = np.random.choice(num_points, 4, replace=False)  # Randomly select 4 points
            H_candidate = self.calculate_homography(src_pts[sampled_indices], dst_pts[sampled_indices])  # Compute the homography matrix from selected points

            # Convert source points to homogeneous coordinates
            src_pts_homogeneous = np.hstack((src_pts, np.ones((num_points, 1))))
            # Convert destination points to homogeneous coordinates
            dst_pts_homogeneous = np.hstack((dst_pts, np.ones((num_points, 1))))

            # Apply the homography matrix to source points
            dst_pts_transformed = np.matmul(H_candidate, src_pts_homogeneous.T).T
            # Convert back to non-homogeneous coordinates
            dst_pts_transformed = dst_pts_transformed[:, :2] / dst_pts_transformed[:, 2:]

            # Compute the difference between predicted and actual coordinates
            residuals = np.linalg.norm(dst_pts_transformed - dst_pts, axis=1)
            inlier_count = np.sum(residuals < threshold)  # Count the number of inliers

            # Update the optimal homography matrix if the current one has more inliers
            if inlier_count > max_inlier_count:
                max_inlier_count = inlier_count
                optimal_H = H_candidate

        return optimal_H  # Return the optimal homography matrix

    def image_alignment(self,image, H, scale_factor):
        height, width, _ = image.shape  # Get the dimensions of the input image
        new_height, new_width = scale_factor * height, scale_factor * width  # Create a canvas that is 4 times larger than the image
        aligned_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)  # Initialize the output image

        # Loop through each pixel in the output image
        for y in range(-height, new_height - height):
            for x in range(-width, new_width - width):
                # Apply the homography transformation to find the corresponding pixel in the input image
                transformed_point = np.dot(H, np.array([x, y, 1]))
                transformed_point /= transformed_point[2]  # Normalize the coordinates

                # Check if the transformed coordinates are within the bounds of the input image
                if 0 <= transformed_point[0] < image.shape[1] and 0 <= transformed_point[1] < image.shape[0]:
                    # Interpolate the pixel color value using bilinear interpolation
                    x0, y0 = int(transformed_point[0]), int(transformed_point[1])  # Get the top-left pixel coordinates
                    x1, y1 = x0 + 1, y0 + 1  # Get the bottom-right pixel coordinates
                    alpha = transformed_point[0] - x0  # Calculate the weight for interpolation
                    beta = transformed_point[1] - y0  # Calculate the weight for interpolation

                    # Check if the neighboring pixels are within the bounds of the input image
                    if 0 <= x0 < image.shape[1] and 0 <= x1 < image.shape[1] and \
                    0 <= y0 < image.shape[0] and 0 <= y1 < image.shape[0]:
                        # Perform bilinear interpolation
                        interpolated_color = (1 - alpha) * (1 - beta) * image[y0, x0] + \
                                            alpha * (1 - beta) * image[y0, x1] + \
                                            (1 - alpha) * beta * image[y1, x0] + \
                                            alpha * beta * image[y1, x1]
                        aligned_image[y + height, x + width] = interpolated_color.astype(np.uint8)  # Set the pixel value in the canvas

        return aligned_image, height, width  # Return the aligned image and canvas parameters

    def crop_background_from_image(self,image):
        mask = image.sum(axis=2) > 0  # Create a mask to identify non-black pixels
        y_coords, x_coords = np.where(mask)  # Get the coordinates of the non-black pixels
        x_min, x_max = x_coords.min(), x_coords.max()  # Get the minimum and maximum x coordinates
        y_min, y_max = y_coords.min(), y_coords.max()  # Get the minimum and maximum y coordinates
        cropped_image = image[y_min:y_max + 1, x_min:x_max + 1, :]  # Crop the image to remove black pixels
        return cropped_image  # Return the cropped image

    def transform_images(self,img1, img2, H, focus=2, blend=True, scale_factor=4, blend_region=5):
        height1, width1 = img1.shape[:2]  # Height and width of the first image
        height2, width2 = img2.shape[:2]  # Height and width of the second image
        corners1 = np.array([[0, 0], [0, height1], [width1, height1], [width1, 0]], dtype=np.float32)  # Corners of the first image
        corners2 = np.array([[0, 0], [0, height2], [width2, height2], [width2, 0]], dtype=np.float32)  # Corners of the second image

        # Transform the corners of the second image
        transformed_corners2 = cv2.perspectiveTransform(corners2.reshape(1, -1, 2), H).reshape(-1, 2)

        # Combine the corners of both images
        all_corners = np.concatenate((corners1, transformed_corners2), axis=0)
        x_min, y_min = np.int32(all_corners.min(axis=0).ravel() - 0.5)  # Calculate the minimum corner coordinates
        x_max, y_max = np.int32(all_corners.max(axis=0).ravel() + 0.5)  # Calculate the maximum corner coordinates

        # Ensure x_min and y_min are at least zero to prevent negative indexing
        x_min, y_min = max(0, x_min), max(0, y_min)

        translation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])  # Create a translation matrix
        inverse_H = np.linalg.inv(translation_matrix.dot(H))  # Calculate the inverse of the homography matrix

        if focus == 1:  # Wrap the second image into the first image
            transformed_img, height_, width_ = self.image_alignment(img2, H, scale_factor)  # Align the second image
            result_image = transformed_img.copy()  # Copy the transformed image

            # Define destination region with boundary checks
            dest_height = min(height_ + height1, result_image.shape[0])
            dest_width = min(width_ + width1, result_image.shape[1])

            # Paste img1 within bounds
            result_image[height_:dest_height, width_:dest_width] = img1[:dest_height - height_, :dest_width - width_]

            if blend:  # Blend around the edges if blending is enabled
                # Blend right edge
                blend_region_img = result_image[height_: height_ + height1, -blend_region + width_: width_ + blend_region]
                result_image[height_: height_ + height1, -blend_region + width_: width_ + blend_region] = cv2.GaussianBlur(blend_region_img, (3, 1), blend_region, blend_region)

                # Blend left edge
                blend_region_img = result_image[height_: height_ + height1, width2 - blend_region + width_: width_ + blend_region + width2]
                result_image[height_: height_ + height1, width2 - blend_region + width_: width_ + blend_region + width2] = cv2.GaussianBlur(blend_region_img, (3, 1), blend_region, blend_region)

                # Blend top edge
                blend_region_img = result_image[-blend_region + height_: height_ + blend_region, width_: width_ + width1]
                result_image[-blend_region + height_: height_ + blend_region, width_: width_ + width1] = cv2.GaussianBlur(blend_region_img, (1, 3), blend_region, blend_region)

                # Blend bottom edge
                blend_region_img = result_image[height2 - blend_region + height_: height_ + blend_region + height2, width_: width_ + width1]
                result_image[height2 - blend_region + height_: height_ + blend_region + height2, width_: width_ + width1] = cv2.GaussianBlur(blend_region_img, (1, 3), blend_region, blend_region)

        else:  # Wrap the first image into the second image
            transformed_img, height_, width_ = self.image_alignment(img1, inverse_H, scale_factor)  # Align the first image
            result_image = transformed_img.copy()  # Copy the transformed image

            # Define destination region with boundary checks
            dest_height = min(height_ + height2 - y_min, result_image.shape[0])
            dest_width = min(width_ + width2 - x_min, result_image.shape[1])

            # Paste img2 within bounds
            result_image[-y_min + height_:dest_height, -x_min + width_:dest_width] = img2[:dest_height - height_, :dest_width - width_]

            if blend:  # Blend around the edges if blending is enabled
                # Blend right edge
                blend_region_img = result_image[-y_min + height_: height_ + height2 - y_min, -x_min - blend_region + width_: width_ + blend_region - x_min]
                result_image[-y_min + height_: height_ + height2 - y_min, -x_min - blend_region + width_: width_ + blend_region - x_min] = cv2.GaussianBlur(blend_region_img, (3, 1), blend_region, blend_region)

                # Blend left edge
                blend_region_img = result_image[-y_min + height_: height_ + height2 - y_min, width1 - blend_region + width_: width_ + blend_region + width1]
                result_image[-y_min + height_: height_ + height2 - y_min, width1 - blend_region + width_: width_ + blend_region + width1] = cv2.GaussianBlur(blend_region_img, (3, 1), blend_region, blend_region)

                # Blend top edge
                blend_region_img = result_image[-blend_region + height_: height_ + blend_region - y_min, -x_min + width_: width_ + width1]
                result_image[-blend_region + height_: height_ + blend_region - y_min, -x_min + width_: width_ + width1] = cv2.GaussianBlur(blend_region_img, (1, 3), blend_region, blend_region)

                # Blend bottom edge
                blend_region_img = result_image[height2 - blend_region + height_: height_ + blend_region + height2 - y_min, -x_min + width_: width_ + width1]
                result_image[height2 - blend_region + height_: height_ + blend_region + height2 - y_min, -x_min + width_: width_ + width1] = cv2.GaussianBlur(blend_region_img, (1, 3), blend_region, blend_region)

        return self.crop_background_from_image(result_image)  # Return the final image after removing black pixels

