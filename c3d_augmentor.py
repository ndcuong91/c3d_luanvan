import cv2
import os
import numpy as np
import c3d_params

data_folder=c3d_params.c3d_data_root
method=3
Kinects=['K1','K3','K5']
subjects=['Binh','Giang','Hung','Tan','Thuan']
actions=[1,2,3,4,5]
old_res=[640,480]
padding=[100,75]
do_padding=True

#old method
new_res=[560,420]
crop_set_augment_1=[]
crop_set_1=[[0,0],[80,0],[40,30],[80,60],[0,60]] #for action 1,3,5
crop_set_2=[[0,0],[20,0],[40,0],[60,0],[80,0]] # for action 2
crop_set_3=[[80,0],[80,15],[80,30],[80,45],[80,60]] #for action 4
crop_set_augment_1.append(crop_set_1)
crop_set_augment_1.append(crop_set_2)
crop_set_augment_1.append(crop_set_3)

#new method
new_res_new=[256,192]
begin_pt=[110,30]
#pos_set = [[0, 0], [25, 25], [50, 50], [75, 75], [100, 100],[25, 75], [75, 25], [0, 100], [100, 0]]
pos_set = [[0, 0], [50, 50], [100, 100], [0, 100], [100, 0]]
final_pos=[]
for pos in pos_set:
    temp_pos=[pos[0]+begin_pt[0],pos[1]+begin_pt[1]]
    final_pos.append(temp_pos)


def get_list_dir_in_folder (dir):
    sub_dir=[o for o in os.listdir(dir) if os.path.isdir(os.path.join(dir, o))]
    return sub_dir
def get_list_jpg_in_folder(dir):
    included_extensions = ['jpg']
    file_names = [fn for fn in os.listdir(dir)
                  if any(fn.endswith(ext) for ext in included_extensions)]
    return file_names

#method for augmented with crop size 560 x 420
def make_new_data_1(data_folder,kinect_folder, kinect_folder_augmented, subject,action, crop_set):
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


#method for augmented with crop size 320 x 320, focus on motion
def make_new_data_2(data_folder,kinect_folder, kinect_folder_augmented, subject,action, final_pos):

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
                final_image = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
                final_image[final_pos[i][1]:final_pos[i][1]+ crop_rect[1], final_pos[i][0]:final_pos[i][0]+ crop_rect[0]]=crop_img
                cv2.imwrite(os.path.join(action_folder_augmented,new_folder,image),final_image)

#method for augmented which center by hand
def make_new_data_3(data_folder,kinect_folder, kinect_folder_augmented, subject,action, subject_id):

    with open(os.path.join(data_folder,kinect_folder,'crop_aug_3')) as f:
        lines = f.readlines()
    hand_pos=lines[subject_id].replace('\n','').split(',')
    shift_x=old_res[0]/2-int(hand_pos[0])
    shift_y=old_res[1]/2-int(hand_pos[1])

    action_folder=os.path.join(data_folder,kinect_folder,subject,str(action))
    action_folder_augmented=os.path.join(data_folder,kinect_folder_augmented,subject,str(action))
    samples=get_list_dir_in_folder(action_folder)

    shift=20
    new_shift=[[-shift,-shift],[-shift,shift],[shift,-shift],[0,0],[shift,shift]]

    for sample in samples:
        if(len(sample)>1):
            continue
        image_list=get_list_jpg_in_folder(action_folder+'/'+sample)
        for i in range(len(new_shift)):
            new_folder = sample + '_' + str(i + 1)
            if not os.path.exists(action_folder_augmented + '/' + new_folder):
                os.makedirs(action_folder_augmented + '/' + new_folder)
            for image in image_list:
                img = cv2.imread(action_folder + '/' + sample + '/' + image)
                M = np.float32([[1, 0, shift_x+new_shift[i][0]], [0, 1, shift_y+new_shift[i][1]]])
                shift_image = cv2.warpAffine(img, M, (old_res[0], old_res[1]))
                cv2.imwrite(os.path.join(action_folder_augmented, new_folder, image), shift_image)

def make_new_data_3_not_aug(data_folder,kinect_folder, kinect_folder_augmented, subject,action, subject_id):

    with open(os.path.join(data_folder,kinect_folder,'crop_aug_3')) as f:
        lines = f.readlines()
    hand_pos=lines[subject_id].replace('\n','').split(',')
    shift_x=old_res[0]/2-int(hand_pos[0])
    shift_y=old_res[1]/2-int(hand_pos[1])

    action_folder=os.path.join(data_folder,kinect_folder,subject,str(action))
    action_folder_augmented=os.path.join(data_folder,kinect_folder_augmented,subject,str(action))
    samples=get_list_dir_in_folder(action_folder)


    for sample in samples:
        if(len(sample)>1):
            continue
        image_list=get_list_jpg_in_folder(action_folder+'/'+sample)
        if not os.path.exists(action_folder_augmented + '/' + sample):
            os.makedirs(action_folder_augmented + '/' + sample)
        for image in image_list:
            img = cv2.imread(action_folder + '/' + sample + '/' + image)
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            shift_image = cv2.warpAffine(img, M, (old_res[0], old_res[1]))
            cv2.imwrite(os.path.join(action_folder_augmented, sample, image), shift_image)



for kinect in Kinects:
    kinect_folder=kinect+'_original'
    kinect_folder_augmented=kinect+'_original_not_aug_'+str(method)
    print('Begin augment data with method '+str(method)+' for ' + kinect)

    for n in range(len(subjects)):
        print('Augment subject '+subjects[n])
        if (method == 1):
            for i in range(len(actions)):
                print('Action '+str(i+1))
                make_new_data_1(data_folder, kinect_folder, kinect_folder_augmented, subjects[n], actions[i])
        if (method == 2):
            for i in range(len(actions)):
                print('Action '+str(i+1))
                make_new_data_2(data_folder, kinect_folder, kinect_folder_augmented, subjects[n], actions[i], final_pos)
        if (method == 3):
            for i in range(len(actions)):
                print('Action '+str(i+1))
                make_new_data_3_not_aug(data_folder, kinect_folder, kinect_folder_augmented, subjects[n], actions[i], n)

print('Finish.')
