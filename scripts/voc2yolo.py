from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm
import xmltodict
import numpy as np
import glob
import os
import random
import cv2

# VOC_class_names = ['aeroplane', 'bicycle', 'bird','boat','bottle', 'bus', 'car',
#                     'cat', 'chair', 'cow','diningtable','dog', 'horse', 'motorbike',
#                     'person','pottedplant','sheep', 'sofa', 'train', 'tvmonitor']
#VOC_class_names = ['warning', 'information', 'other', 'regulatory', 'complementary']

#VOC_class_names = ['rice', 'weed']

#VOC_class_names = [ "blossom_end_rot", "graymold","powdery_mildew","spider_mite","spotting_disease"]
VOC_class_names = [ "blossom_end_rot", "graymold","powdery_mildew","spider_mite","spotting_disease", "snails_and_slugs"]
 
images_dir = 'D:/cw_projects/paprika/paprika_processed/data_final/train_aug_no_bg/'
op_dir = 'D:/cw_projects/paprika/paprika_processed/data_final/paprika_y4/'

xml_filepaths = glob.glob( os.path.join( images_dir , '*.xml' ) )
#%%
def remove_duplicate(s):
    '''
    I created this function b/c i am looping over all the keys in the xml file
    and in case if xmldict have only one object the loop sitll consider the 
    <part> keys to be part of object so i am just looping over same object again 
    and again. So i just rmove those duplication with this
    '''
    x = s.split(' ')
    y = list(set(x))
    y = ' '.join(map(str, y))
    return y
for filepath in tqdm(xml_filepaths):
    
    full_dict = xmltodict.parse(open( filepath , 'rb' ))
    try:
        obj_boxnnames = full_dict[ 'annotation' ][ 'object' ] # names and boxes
        file_name = full_dict[ 'annotation' ][ 'filename' ] #os.path.basename(filepath)[:-4]+'.jpg'
        all_bounding_boxnind = []
        for i in range(len(obj_boxnnames)):
            # 1st get the name and indices of the class
            try:
                obj_name = obj_boxnnames[i]['name']
            except:
                obj_name = obj_boxnnames['name']  # if the xml file has only one object key
                
            # Uncomment if change in classes
            # obj_name = obj_name.split('-')
            # obj_name = obj_name[0]
            
            obj_ind = [i for i in range(len(VOC_class_names)) if obj_name == VOC_class_names[i]] # get the index of the object
            obj_ind = int(np.array(obj_ind))
            # 2nd get tht bbox coord and append the class name at the end
            try:
                obj_box = obj_boxnnames[i]['bndbox']
            except:
                obj_box = obj_boxnnames['bndbox'] # if the xml file has only one object key
            bounding_box = [0.0] * 4                    # creat empty list
            bounding_box[0] = int(float(obj_box['xmin']))# two times conversion is for handeling exceptions 
            bounding_box[1] = int(float(obj_box['ymin']))# so that if coordinates are given in float it'll
            bounding_box[2] = int(float(obj_box['xmax']))# still convert them to int
            bounding_box[3] = int(float(obj_box['ymax']))
            bounding_box.append(obj_ind)                # append the class ind in same list (YOLO format)
            bounding_box = str(bounding_box)[1:-1]      # remove square brackets
            bounding_box = "".join(bounding_box.split())# remove spaces in between **here dont give space inbetween the inverted commas "".
            all_bounding_boxnind.append(bounding_box)
        all_bounding_boxnind = ' '.join(map(str, all_bounding_boxnind))# convert list to string
        all_bounding_boxnind = remove_duplicate(all_bounding_boxnind)
        full_anotation = file_name + ' ' + all_bounding_boxnind
        
        # check if file exiscts else make new
        with open(op_dir + "train.txt", "a+") as file_object:
            # Move read cursor to the start of file.
            file_object.seek(0)
            # If file is not empty then append '\n'
            data = file_object.read(100)
            if len(data) > 0 :
                file_object.write("\n")
            # Append text at the end of file
            file_object.write(full_anotation)
    except KeyError:
            print("KeyError \n Kindly check keys")
    
#%%    
######################################################################
#                  Train_Val split
######################################################################
'''
give a .txt file containing all the images annotations and it'll split them in two
'''
full_list = []
with open('C:/Users/Talha/Desktop/paprika_label 20.08.25/paparika/full_annotations.txt', 'r') as f:
        full_list = full_list + f.readlines()
train_list = full_list
val_list = []
total_data = len(full_list)
val_data = 363

length = np.arange(total_data-val_data)
for i in range(val_data):# b/c total length is 17000-4000=13000
    x = random.choice(length)
    val_list.append(train_list[x])  # train list has all the elements of full list
    del train_list[x]               # now delete that element form train list to avoid duplicates
#%%
for i in range(len(val_list)):
    with open('C:/Users/Talha/Desktop/paprika_label 20.08.25/paparika/valid.txt', "a+") as file_object:
            # Move read cursor to the start of file.
            file_object.seek(0)
            # If file is not empty then append '\n'
            data = file_object.read(100)
            # Append text at the end of file
            file_object.write(val_list[i])
for i in range(len(train_list)):
    with open('C:/Users/Talha/Desktop/paprika_label 20.08.25/paparika/train.txt', "a+") as file_object:
            # Move read cursor to the start of file.
            file_object.seek(0)
            # If file is not empty then append '\n'
            data = file_object.read(100)
            # Append text at the end of file
            file_object.write(train_list[i])
#%%
######################################################################
#                  Train_Val Write
######################################################################
full_list = []
with open('C:/Users/Talha/Desktop/paprika_label 20.08.25/paparika/train.txt', 'r') as f:
        full_list = full_list + f.readlines()
for item in tqdm(full_list):
      item = item.replace("\n", "").split(" ")
      img_nam = item[0]
      img = cv2.imread('C:/Users/Talha/Desktop/paprika_label 20.08.25/paparika/images/'+ img_nam)
      cv2.imwrite('C:/Users/Talha/Desktop/paprika_label 20.08.25/paparika/train/'+img_nam, img)
      
full_list = []
with open('C:/Users/Talha/Desktop/paprika_label 20.08.25/paparika/valid.txt', 'r') as f:
        full_list = full_list + f.readlines()
for item in tqdm(full_list):
      item = item.replace("\n", "").split(" ")
      img_nam = item[0]
      img = cv2.imread('C:/Users/Talha/Desktop/paprika_label 20.08.25/paparika/images/'+ img_nam)
      cv2.imwrite('C:/Users/Talha/Desktop/paprika_label 20.08.25/paparika/valid/'+img_nam, img)