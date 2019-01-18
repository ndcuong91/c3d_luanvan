import cv2
import os
import numpy as np

data_folder='/home/prdcv/PycharmProjects/gvh205/c3d_luanvan/data'
new_method=True
kinect_folder='Kinect_1_clean_1_rename'
kinect_folder_augmented='Kinect_1_clean_1_augmented_padding_100'
if (new_method==True):
    kinect_folder_augmented = 'Kinect_1_clean_1_augmented_padding_new'

subjects=['Binh','Giang','Hung','Tan','Thuan']
actions=[1,2,3,4,5]
old_res=[640,480]
new_res=[560,420]
new_res_new=[256,192]
padding=[100,75]
do_padding=True

crop_set_augment_1=[]
crop_set_1=[[0,0],[80,0],[40,30],[80,60],[0,60]] #for action 1,3,5
crop_set_2=[[0,0],[20,0],[40,0],[60,0],[80,0]] # for action 2
crop_set_3=[[80,0],[80,15],[80,30],[80,45],[80,60]] #for action 4
crop_set_augment_1.append(crop_set_1)
crop_set_augment_1.append(crop_set_2)
crop_set_augment_1.append(crop_set_3)



# crop_set_1=[[0,0],[80,0],[40,30],[80,60],[0,60],[20,15],[60,15],[20,45],[60,45]] #for action 1,3,5
# crop_set_2=[[0,0],[10,0],[20,0],[30,0],[40,0],[50,0],[60,0],[70,0],[80,0]] # for action 2
# crop_set_3=[[80,0],[80,7],[80,15],[80,22],[80,30],[80,38],[80,45],[80,53],[80,60]] #for action 4

def get_list_dir_in_folder (dir):
    sub_dir=[o for o in os.listdir(dir) if os.path.isdir(os.path.join(dir, o))]
    return sub_dir
def get_list_jpg_in_folder(dir):
    included_extensions = ['jpg']
    file_names = [fn for fn in os.listdir(dir)
                  if any(fn.endswith(ext) for ext in included_extensions)]
    return file_names

def make_new_data(data_folder,kinect_folder, kinect_folder_augmented, subject,action, crop_set):
    action_folder=os.path.join(data_folder,kinect_folder,subject,str(action))
    action_folder_augmented=os.path.join(data_folder,kinect_folder_augmented,subject,str(action))
    samples=get_list_dir_in_folder(action_folder)
    for sample in samples:
        if(len(sample)>1):
            continue
        image_list=get_list_jpg_in_folder(action_folder+'/'+sample)
        for i in range(len(crop_set)):
            new_folder = sample + '_' + str(i + 1)
            if not os.path.exists(action_folder_augmented + '/' + new_folder):
                os.makedirs(action_folder_augmented + '/' + new_folder)
            for image in image_list:
                img = cv2.imread(action_folder+'/'+sample+'/'+image)
                crop_img = img[crop_set[i][1]:crop_set[i][1] + new_res[1], crop_set[i][0]:crop_set[i][0] + new_res[0]]
                if (do_padding==True):
                    padding_image = np.zeros((2*padding[1]+ new_res[1], 2*padding[0]+ new_res[0], 3), np.uint8)
                    padding_image[padding[1]:padding[1]+ new_res[1], padding[0]:padding[0]+ new_res[0]]=crop_img
                    cv2.imwrite(os.path.join(action_folder_augmented,new_folder,image),padding_image)
                else:
                    cv2.imwrite(os.path.join(action_folder_augmented,new_folder,image),crop_img)


def make_new_data_new(data_folder,kinect_folder, kinect_folder_augmented, subject,action, final_pos):

    with open(os.path.join(data_folder,kinect_folder,subject,'crop_aug_2')) as f:
        lines = f.readlines()
    crop_rect=[320,320]
    crop_pos=lines[action-1].replace('\n','').split(',')
    x = int(crop_pos[0])
    y = int(crop_pos[1])

    action_folder=os.path.join(data_folder,kinect_folder,subject,str(action))
    action_folder_augmented=os.path.join(data_folder,kinect_folder_augmented,subject,str(action))
    samples=get_list_dir_in_folder(action_folder)

    for sample in samples:
        if(len(sample)>1):
            continue
        image_list=get_list_jpg_in_folder(action_folder+'/'+sample)
        for i in range(len(final_pos)):
            new_folder = sample + '_' + str(i + 1)
            if not os.path.exists(action_folder_augmented + '/' + new_folder):
                os.makedirs(action_folder_augmented + '/' + new_folder)
            for image in image_list:
                img = cv2.imread(action_folder+'/'+sample+'/'+image)
                crop_img = img[y:y+ crop_rect[1], x:x + crop_rect[0]]
                if (do_padding==True):
                    padding_image = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
                    padding_image[final_pos[i][1]:final_pos[i][1]+ crop_rect[1], final_pos[i][0]:final_pos[i][0]+ crop_rect[0]]=crop_img
                    cv2.imwrite(os.path.join(action_folder_augmented,new_folder,image),padding_image)
                else:
                    cv2.imwrite(os.path.join(action_folder_augmented,new_folder,image),crop_img)
                kk=1


begin_pt=[110,30]
pos_set = [[0, 0], [25, 25], [50, 50], [75, 75], [100, 100],[25, 75], [75, 25], [0, 100], [100, 0]]
final_pos=[]
for pos in pos_set:
    temp_pos=[pos[0]+begin_pt[0],pos[1]+begin_pt[1]]
    final_pos.append(temp_pos)


#action 1
if(new_method==True):
    print('Begin augment data with new method')
else:
    print('Begin augment data with old method')

print('augment action 1')
for subject in subjects:
    if(new_method==True):
        make_new_data_new(data_folder,kinect_folder,kinect_folder_augmented, subject,actions[0], final_pos)
    else:
        make_new_data(data_folder,kinect_folder,kinect_folder_augmented, subject,actions[0], crop_set_1)

#action 2
print('augment action 2')
for subject in subjects:
    if(new_method==True):
        make_new_data_new(data_folder,kinect_folder,kinect_folder_augmented, subject,actions[1], final_pos)
    else:
        make_new_data(data_folder,kinect_folder,kinect_folder_augmented, subject,actions[1], crop_set_1)

#action 3
print('augment action 3')
for subject in subjects:
    if(new_method==True):
        make_new_data_new(data_folder,kinect_folder,kinect_folder_augmented, subject,actions[2], final_pos)
    else:
        make_new_data(data_folder,kinect_folder,kinect_folder_augmented, subject,actions[2], crop_set_1)

#action 4
print('augment action 4')
for subject in subjects:
    if(new_method==True):
        make_new_data_new(data_folder,kinect_folder,kinect_folder_augmented, subject,actions[3], final_pos)
    else:
        make_new_data(data_folder,kinect_folder,kinect_folder_augmented, subject,actions[3], crop_set_1)

#action 5
print('augment action 5')
for subject in subjects:
    if(new_method==True):
        make_new_data_new(data_folder,kinect_folder,kinect_folder_augmented, subject,actions[4], final_pos)
    else:
        make_new_data(data_folder,kinect_folder,kinect_folder_augmented, subject,actions[4], crop_set_1)

print('Finish.')
