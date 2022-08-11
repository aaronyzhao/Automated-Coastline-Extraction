from ntpath import join
import sys, cv2, random, shutil, os, glob
from tkinter import image_names
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from skimage.io import imsave, imread
import numpngw
from scipy.signal import find_peaks
from skimage.transform import resize
from skimage.morphology import skeletonize
from matplotlib import pyplot as plt

sys.path.insert(1, '../training/keras-deeplab-v3-plus')
sys.path.insert(2, '../training')
# from preprocessing import preprocess_image
# from processing import predict, calculate_mean_deviation, calculate_iou, process_mohajerani_on_calfin
# from plotting import plot_validation_results, plot_production_results, plot_troubled_ones
# from mask_to_shp import mask_to_shp
# from error_analysis import extract_front_indicators
# from ordered_line_from_unordered_points import is_outlier



def remove_small_components(image, limit=np.inf, min_size_percentage=0.00001):
    """Removes small connected regions from an image."""
    image = image.astype('uint8') # specify img type
    image = cv2.bitwise_not(image)
    #find all connected components (white blobs in image)
    # output = destination labeled img, 
    # stats = stats output for each label, may be accessed via stats(label, COLUMN) where COLUMN is one of ConnectedComponentsTypes
    # centroid = centroid of ea label
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8) # con = 8 = full con on circle of sq around px
    sizes = stats[:, cv2.CC_STAT_AREA]
    # for all labels in stats, sizes = total area in px of connected component, sort value stored @ idx of sizes from large to small
    ordering = np.argsort(-sizes)
    
    #for every component in the image, keep it only if it's above min_size
    min_size = output.size * min_size_percentage
    if len(ordering) > 1:
        min_size = max(min_size, sizes[ordering[1]] * 0.15)
    
    #Isolate large components, create a blank slate
    large_components = np.zeros((output.shape))
    
    #Store the bounding boxes of components, so they can be isolated and reprocessed further. Default box is entire image.
    bounding_boxes = [[0, 0, image.shape[0], image.shape[1]]]
    #Skip first component, since it's the background color in edge masks
    #Restrict number of components returned depending on limit
    number_returned = 0

    # iterating through all groups of px, including background
    for i in range(0, len(sizes)):
        # if sizes[ordering[i]] group of px is larger than min_size, then add it to large_components
        if sizes[ordering[i]] >= min_size:
            # mask_idx returns an output sized arr and will set True if there's a px there based on sizes[ordering[i]]
            mask_indices = output == ordering[i]
            x, y = np.nonzero(mask_indices)

            #bounding box code
            min_x = min(x) 
            delta_x = max(x) - min_x
            min_y = min(y)
            delta_y = max(y) - min_y
            bounding_boxes.append([min_x, min_y, delta_x, delta_y])

            #add mask idx components into large_components
            large_components[mask_indices] = image[mask_indices]
            number_returned += 1
            if number_returned >= limit:
                break

    #return large component image and bounding boxes for each component
    return large_components.astype(np.float32), bounding_boxes

def path_to_img(img_path):
    return imread(img_path, as_gray=True)


def remove_sc_from_all_img(input_path, dest_path, domains, dry_run = True):	
	total = 0
	for image_path in glob.glob(os.path.join(input_path, "*_mask.png")): # only take masks
		image_name = image_path.split(os.path.sep)[-1]
		image_name_base = image_name.split('.')[0]
		
		domain = image_name_base.split('_')[0]
		if domain in domains:
			image_name_mask = "_".join(image_name_base.split('_')[0:-1]) + '_mask.png'
			
			img_uint8 = imread(image_path, as_gray=True) #np.uint8 [0, 255]
			mask_uint8 = remove_small_components(img_uint8)[0]
			
			save_path_mask = os.path.join(dest_path, image_name_mask)
			total += 1
			print('Saving #' + str(total) + ' processed raw to:', save_path_mask)
			if (dry_run == False):
				# numpngw.write_png(save_path_raw, img_uint8)
				# shutil.copyfile(image_path, save_path_raw)
				imsave(save_path_mask, 255-mask_uint8)

# copy images over to dest_path function?

# create new list file
def file_path_to_lst(input_path, dest_path):
    with open(dest_path, 'w', encoding = 'utf8') as f:
        total = 0
        for image_path in glob.glob(os.path.join(input_path, "*.png")):
            # image_name = image_path.split(os.path.sep)[-3:]
            image_name = os.path.relpath(image_path, start = r"C:\User Files\Caltech\JPL\CALFIN\training\HRNet_2_OCR\HRNet-Semantic-Segmentation\data\coastlines")
            image_name.replace('\\', '/')
            if total % 2 == 0:
                # print(image_name[3::] + str(total) + "\n")
                f.write(image_name + "\t")
                total +=1
                continue
                # return 0
            if total % 2 == 1:
                # print(image_name[3::] + str(total) + "\n")
                f.write(image_name + "\n")
                total +=1
                # return 0
    return 0

#  Modify ODGT file for CSAIL
def file_path_to_lst_CSAIL(input_path, dest_path):
    with open(dest_path, 'w', encoding = 'utf8') as f:
        total = 0
        for image_path in glob.glob(os.path.join(input_path, "*.png")):
            # image_name = image_path.split(os.path.sep)[-3:]
            # image_name = os.path.relpath(image_path, start = r"C:\User Files\Caltech\JPL\CALFIN\training\data")
            image_name = image_path
            image_name.replace("\\", "/")
            if total % 2 == 0:
                # print(image_name[3::] + str(total) + "\n")
                f.write("{\"fpath_img\": ")
                f.write("\"" + image_name + "\", ")
                total +=1
                continue
                # return 0
            if total % 2 == 1:
                # print(image_name[3::] + str(total) + "\n")
                f.write("\"fpath_segm\": ")
                f.write("\"" + image_name + "\", \"width\": 512, \"height\": 512}" + "\n")
                total +=1
                # return 0
    return 0

if __name__ == "__main__":
    # CSAIL Train
    input_path = r"C:\User Files\Caltech\JPL\CALFIN\training\data\train_patched_dual_512_448_32"
    dest_path = r"C:\User Files\Caltech\JPL\CALFIN\training\semantic-segmentation-pytorch-master\data\training.odgt"
    file_path_to_lst_CSAIL(input_path, dest_path)

    # CSAIL Validation 
    input_path = r"C:\User Files\Caltech\JPL\CALFIN\training\data\validation_patched_dual_512_448_32"
    dest_path = r"C:\User Files\Caltech\JPL\CALFIN\training\semantic-segmentation-pytorch-master\data\validation.odgt"
    file_path_to_lst_CSAIL(input_path, dest_path)

    
    
    
    # # # img_path = r"D:/Caltech/JPL/CALFIN_data/CALFIN/training\data/train_nora/Crane_2007-03-31_ENVISAT_20_3_467_mask.png"
    # img_path = r"D:/Caltech/JPL/CALFIN_data/CALFIN/training\data/train_nora_masks_temp/Crane_2002-11-09_ERS_20_2_061_mask.png"
    # # img_path = r"D:/Caltech/JPL/CALFIN_data/CALFIN/training\data/train_nora/Crane_2009-09-29_PALSAR_17_1_137_mask.png"
    # img = path_to_img(img_path)

    # masked_image = remove_small_components(img)[0]
    # # large_inverse_distances = 255-remove_small_components(255-img)[0]

    # # plt.imshow(large_inverse_distances)
    # plt.imshow(masked_image)
    # plt.show()

    # domains = ["Crane", "DBE", "JAC", "Jorum", "SI"]
    # input_path = r"../training/data/train_nora"
    # dest_path = r"../training/data/train_nora_masks_temp"
    # remove_sc_from_all_img(input_path, dest_path, domains, dry_run = False)

    # domains = ["Crane", "DBE", "JAC", "Jorum", "SI"]
    # input_path = r"../training/data/train_nora"
    # dest_path = r"../training/data/train_nora_masks_temp_inverted"
    # remove_sc_from_all_img(input_path, dest_path, domains, dry_run = False)

    # test_domains = ['COL', 'Mapple']
    # input_path = r"../training/data/validation_nora"
    # dest_path = r"../training/data/validation_nora_masks_temp"
    # remove_sc_from_all_img(input_path, dest_path, domains, dry_run = False)

    # test_domains = ['COL', 'Mapple']
    # input_path = r"../training/data/validation_nora"
    # dest_path = r"../training/data/validation_nora_masks_temp_inverted"
    # remove_sc_from_all_img(input_path, dest_path, test_domains, dry_run = False)

    # input_path = r"D:\Caltech\JPL\CALFIN\training\data\train_patched_dual_512_448_32"
    
    # # Train
    # input_path = r"C:\User Files\Caltech\JPL\CALFIN\training\HRNet_2_OCR\HRNet-Semantic-Segmentation\data\coastlines\calvingFront\train"
    # dest_path = r"C:\User Files\Caltech\JPL\CALFIN\training\HRNet_2_OCR\HRNet-Semantic-Segmentation\data\list\coastlines\train.lst"
    # file_path_to_lst(input_path, dest_path)

    # # Validation 
    # input_path = r"C:\User Files\Caltech\JPL\CALFIN\training\HRNet_2_OCR\HRNet-Semantic-Segmentation\data\coastlines\calvingFront\val"
    # dest_path = r"C:\User Files\Caltech\JPL\CALFIN\training\HRNet_2_OCR\HRNet-Semantic-Segmentation\data\list\coastlines\val.lst"
    # file_path_to_lst(input_path, dest_path)

    # # TrainVal
    # input_path = r"D:\Caltech\JPL\CALFIN\training\ HRNet_2_OCR\HRNet-Semantic-Segmentation\data\coastlines\calvingFront\val"
    # dest_path = r"D:\Caltech\JPL\CALFIN\training\HRNet_2_OCR\HRNet-Semantic-Segmentation\data\list\coastlines\val.lst"
    # file_path_to_lst(input_path, dest_path)

    
    