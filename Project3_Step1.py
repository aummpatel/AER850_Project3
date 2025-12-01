# %%

# Import all necessary packages
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the path to the Motherboard Image and verify it has loaded correctly by printing the shape of the image tensor
mb_img_path = "Project 3 Data/Project 3 Data/motherboard_image.jpeg"
mb_img_original = cv2.imread(mb_img_path)
print(f"The shape of the loaded image is:{mb_img_original.shape}")
#%% Step 1: Object Masking

# Image Pre-processing
# Rotating the image to match the required orientation of the extracted image
mb_img_rot = cv2.rotate(mb_img_original, cv2.ROTATE_90_CLOCKWISE)
# Converting the BGR image to Grayscale since OpenCV's threshold works with only Grayscale images 
mb_img_gray = cv2.cvtColor(mb_img_rot,cv2.COLOR_BGR2GRAY)
# Gaussian Blur to remove noise (works well with Otsu's Thresholding)
mb_img_blurred = cv2.GaussianBlur(mb_img_gray, (5,5), 0)

#-----------------------------------------------------------------------------------------------------------
# First Method - Otsu's Thresholding + Canny + Dilation

# Otsu's Thresholding
threshold_otsu, mb_img_otsu = cv2.threshold(mb_img_blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print("Canny + Otsu's threshold value =", threshold_otsu)
# Canny Edge Detection and Edge Dilation
edges_image = cv2.Canny(mb_img_blurred, 0.5*threshold_otsu, 1.5*threshold_otsu)
dilated_edges = cv2.dilate(edges_image, None, iterations = 5)

#-----------------------------------------------------------------------------------------------------------
# Second Method - Otsu's Thresholding + Morphological closing

# Otsu's Inverse Thresholding
threshold_otsu_2, mb_img_otsu_2 = cv2.threshold(mb_img_blurred,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
print("Morphology Otsu's threshold value =", threshold_otsu_2)
# Morphological closing for internal holes
kernel = np.ones((15, 15), np.uint8)  # A larger kernel closes larger gaps
closed_mask = cv2.morphologyEx(mb_img_otsu_2, cv2.MORPH_CLOSE, kernel)

# %% Contour Detections for both the methods
# Method 1
contours1, _ = cv2.findContours(dilated_edges,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Find contours in the dilated binary edge image
mask1 = np.zeros_like(mb_img_rot) # Create an empty np array of the same size as the image
largest_contour1 = max(contours1, key=cv2.contourArea) #Extracting the biggest contour assuming it is the pcb board
cv2.drawContours(mask1,[largest_contour1], -1, (255,255,255),thickness=cv2.FILLED) # Fill the empty array with the mask values
masked_image1 = cv2.bitwise_and(mb_img_rot, mask1) # bitwise_and to extract the pcb board

plt.figure()
plt.imshow(dilated_edges, cmap='gray')
plt.title("Edge Detection Image (Method 1)")
plt.axis('off')
plt.show()

plt.figure()
plt.imshow(mask1, cmap='gray')
plt.title("Mask image (Method 1)")
plt.axis('off')
plt.show()

plt.figure()
plt.imshow(cv2.cvtColor(masked_image1, cv2.COLOR_BGR2RGB))
plt.title("Masked Image (Method 1) - Final Result")
plt.axis('off')

cv2.imwrite("Project 3 Outputs/Edge Detection Images/Edge_Detection_Method1.png",dilated_edges)
cv2.imwrite("Project 3 Outputs/Contour Mask Images/Mask_Method1.png",mask1)
#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------
# Method 2
contours2, _ = cv2.findContours(closed_mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
mask2 = np.zeros_like(mb_img_rot)
largest_contour2 = max(contours2, key=cv2.contourArea)
cv2.drawContours(mask2,[largest_contour2], -1, (255,255,255),thickness=cv2.FILLED)
masked_image2 = cv2.bitwise_and(mb_img_rot, mask2)

plt.figure()
plt.imshow(closed_mask, cmap='gray')
plt.title("PCB Segmentation Image (Method 2)")
plt.axis('off')
plt.show()

plt.figure()
plt.imshow(mask2, cmap='gray')
plt.title("Mask (Method 2)")
plt.axis('off')
plt.show()

plt.figure()
plt.imshow(cv2.cvtColor(masked_image2, cv2.COLOR_BGR2RGB))
plt.title("Masked Image (Method 2) - Final Result")
plt.axis('off')

cv2.imwrite("Project 3 Outputs/Edge Detection Images/Edge_Detection_Method2.png",closed_mask)
cv2.imwrite("Project 3 Outputs/Contour Mask Images/Mask_Method2.png",mask2)
#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------
# SuperMask
super_mask = cv2.bitwise_or(mask1, mask2) # Union of both the masks
super_masked_image = cv2.bitwise_and(mb_img_rot, super_mask) # Extract the pcb board

plt.figure()
plt.imshow(super_mask, cmap='gray')
plt.title("Super Mask")
plt.axis('off')
plt.show()

plt.figure()
plt.imshow(cv2.cvtColor(super_masked_image, cv2.COLOR_BGR2RGB))
plt.title("Super Masked Image - Final Result")
plt.axis('off')

#%%
# Saving the Edge Detection Images --> Contour Masks --> Extracted PCB Image
cv2.imwrite("Project 3 Outputs/Contour Mask Images/Super_Mask.png",super_mask)
cv2.imwrite("Project 3 Outputs/Extracted PCB Images/Extracted_Motherboard_Image_Super.png",super_masked_image)

# Priting shapes of input image as well as the extracted image to ensure resolution is unchanged
print("Input Image Shape:", mb_img_rot.shape)
print("Extracted Masked Image Shape:", super_masked_image.shape)
