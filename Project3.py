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
edges_image_1 = cv2.Canny(mb_img_blurred, 0.5*threshold_otsu, 1.5*threshold_otsu)
dilated_edges_1 = cv2.dilate(edges_image_1, None, iterations = 5)
#-----------------------------------------------------------------------------------------------------------
# # First Method (b) - Otsu's Thresholding + Canny + Dilation + MorphologyEx
# # Otsu's Thresholding
# threshold_otsu_3, mb_img_otsu_3 = cv2.threshold(mb_img_blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# print("Canny(Morphology) + Otsu's threshold value =", threshold_otsu_3)
# # Canny Edge Detection and Edge Dilation
# edges_image_2 = cv2.Canny(mb_img_blurred, 0.5*threshold_otsu_3, 1.5*threshold_otsu_3)
# dilated_edges_4 = cv2.dilate(edges_image_2, None, iterations = 5)
# # MorphologyEx
# kernel1 = np.ones((15, 15), np.uint8)  # A larger kernel closes larger gaps
# closed_mask2 = cv2.morphologyEx(dilated_edges_4, cv2.MORPH_CLOSE, kernel1)
#-----------------------------------------------------------------------------------------------------------
# # Second Method - Otsu's Threshold Only
# threshold_otsu_1, mb_img_otsu_1 = cv2.threshold(mb_img_blurred,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# print("Only Otsu's threshold value =", threshold_otsu_1)
# dilated_edges_2 = mb_img_otsu_1
#-----------------------------------------------------------------------------------------------------------
# Second Method (b)
threshold_otsu_2, mb_img_otsu_2 = cv2.threshold(mb_img_blurred,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
print("Morphology Otsu's threshold value =", threshold_otsu_2)
kernel = np.ones((15, 15), np.uint8)  # A larger kernel closes larger gaps
closed_mask = cv2.morphologyEx(mb_img_otsu_2, cv2.MORPH_CLOSE, kernel)


# %% Contour Detections for all the methods
contours1, _ = cv2.findContours(dilated_edges_1,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
mask1 = np.zeros_like(mb_img_rot)
largest_contour1 = max(contours1, key=cv2.contourArea)
cv2.drawContours(mask1,[largest_contour1], -1, (255,255,255),thickness=cv2.FILLED)
masked_image1 = cv2.bitwise_and(mb_img_rot, mask1)

plt.figure()
plt.imshow(dilated_edges_1, cmap='gray')
plt.title("Edge Detection Image (Combo)")
plt.axis('off')
plt.show()

plt.figure()
plt.imshow(mask1, cmap='gray')
plt.title("Mask image (Combo)")
plt.axis('off')
plt.show()

plt.figure()
plt.imshow(cv2.cvtColor(masked_image1, cv2.COLOR_BGR2RGB))
plt.title("Masked Image (Combo) - Final Result")
plt.axis('off')

#-----------------------------------------------------------------------------------------------------------
# contours2, _ = cv2.findContours(dilated_edges_2,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# mask2 = np.zeros_like(mb_img_rot)
# largest_contour2 = max(contours2, key=cv2.contourArea)
# cv2.drawContours(mask2,[largest_contour2], -1, (255,255,255),thickness=cv2.FILLED)
# masked_image2 = cv2.bitwise_and(mb_img_rot, mask2)

# plt.figure()
# plt.imshow(dilated_edges_2, cmap='gray')
# plt.title("Edge Detection Image (Threshold Only)")
# plt.axis('off')
# plt.show()

# plt.figure()
# plt.imshow(mask2, cmap='gray')
# plt.title("Mask (Threshold Only)")
# plt.axis('off')
# plt.show()

# plt.figure()
# plt.imshow(cv2.cvtColor(masked_image2, cv2.COLOR_BGR2RGB))
# plt.title("Masked Image (Threshold Only) - Final Result")
# plt.axis('off')

#-----------------------------------------------------------------------------------------------------------
contours3, _ = cv2.findContours(closed_mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
mask3 = np.zeros_like(mb_img_rot)
largest_contour3 = max(contours3, key=cv2.contourArea)
cv2.drawContours(mask3,[largest_contour3], -1, (255,255,255),thickness=cv2.FILLED)
masked_image3 = cv2.bitwise_and(mb_img_rot, mask3)

plt.figure()
plt.imshow(closed_mask, cmap='gray')
plt.title("Edge Detection Image (Threshold + Morphology)")
plt.axis('off')
plt.show()

plt.figure()
plt.imshow(mask3, cmap='gray')
plt.title("Mask (Threshold + Morphology)")
plt.axis('off')
plt.show()

plt.figure()
plt.imshow(cv2.cvtColor(masked_image3, cv2.COLOR_BGR2RGB))
plt.title("Masked Image (Threshold + Morphology) - Final Result")
plt.axis('off')
#-----------------------------------------------------------------------------------------------------------
# contours4, _ = cv2.findContours(closed_mask2,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# mask4 = np.zeros_like(mb_img_rot)
# largest_contour4 = max(contours4, key=cv2.contourArea)
# cv2.drawContours(mask4,[largest_contour4], -1, (255,255,255),thickness=cv2.FILLED)
# masked_image4 = cv2.bitwise_and(mb_img_rot, mask4)

# plt.figure()
# plt.imshow(closed_mask2, cmap='gray')
# plt.title("Edge Detection Image (Combo + Morphology)")
# plt.axis('off')
# plt.show()

# plt.figure()
# plt.imshow(mask4, cmap='gray')
# plt.title("Mask (Combo + Morphology)")
# plt.axis('off')
# plt.show()

# plt.figure()
# plt.imshow(cv2.cvtColor(masked_image4, cv2.COLOR_BGR2RGB))
# plt.title("Masked Image (Combo + Morphology) - Final Result")
# plt.axis('off')
#-----------------------------------------------------------------------------------------------------------
# SuperMask
super_mask = cv2.bitwise_or(mask1, mask3)
super_masked_image = cv2.bitwise_and(mb_img_rot, super_mask)

plt.figure()
plt.imshow(super_mask, cmap='gray')
plt.title("Super Mask")
plt.axis('off')
plt.show()

plt.figure()
plt.imshow(cv2.cvtColor(super_masked_image, cv2.COLOR_BGR2RGB))
plt.title("Super Masked Image - Final Result", color = 'Green')
plt.axis('off')

#%%
# Saving the Edge Detection Images --> Contour Masks --> Extracted PCB Image
cv2.imwrite("Project 3 Outputs/Edge Detection Images/Edge_Detection_Canny.png",dilated_edges_1)
cv2.imwrite("Project 3 Outputs/Edge Detection Images/Edge_Detection_Morphology.png",closed_mask)
cv2.imwrite("Project 3 Outputs/Contour Mask Images/Mask_Canny.png",mask1)
cv2.imwrite("Project 3 Outputs/Contour Mask Images/Mask_Morphology.png",mask3)
cv2.imwrite("Project 3 Outputs/Contour Mask Images/Super_Mask.png",super_mask)
cv2.imwrite("Project 3 Outputs/Extracted PCB Images/Extracted_Motherboard_Image_Canny.png",masked_image1)
cv2.imwrite("Project 3 Outputs/Extracted PCB Images/Extracted_Motherboard_Image_Morphology.png",masked_image3)
cv2.imwrite("Project 3 Outputs/Extracted PCB Images/Extracted_Motherboard_Image_Super.png",super_masked_image)

# Priting shapes of input image as well as the extracted image to ensure resolution is unchanged
print("Input Image Shape:", mb_img_rot.shape)
print("Extracted Masked Image Shape:", super_masked_image.shape)
