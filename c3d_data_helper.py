import cv2
import os
import numpy as np
import shutil
import cv2
from c3d_helper import check_gpu_ready


video_data_dir='/home/titikid/PycharmProjects/c3d_luanvan/data/Dataset_hand_gesture'
image_data_dir='/home/titikid/PycharmProjects/c3d_luanvan/data'
subjects=['Binh','Giang','Hung','Tan','Thuan']
Kinects=['Kinect_1','Kinect_3','Kinect_5']
actions=['1','2','3','4','5']
resolution=(640,480)

def get_list_dir_in_folder (dir):
    sub_dir=[o for o in os.listdir(dir) if os.path.isdir(os.path.join(dir, o))]
    return sub_dir

def get_list_jpg_in_folder(dir):
    included_extensions = ['jpg']
    file_names = [fn for fn in os.listdir(dir)
                  if any(fn.endswith(ext) for ext in included_extensions)]
    return file_names

def summary_video_data():
    for subject in subjects:
        for Kinect in Kinects:
            print('Copy samples from ' + subject + '_' + Kinect)
            summary_dir = os.path.join(video_data_dir, subject, Kinect + '_summary')
            if not os.path.exists(summary_dir):
                os.makedirs(summary_dir)
            sample_dirs = get_list_dir_in_folder(os.path.join(video_data_dir, subject, Kinect))
            for sample in sample_dirs:
                a, b = sample.split('_')
                new_sample_name = a.zfill(2) + '_' + b.zfill(2)
                shutil.copy(os.path.join(video_data_dir, subject, Kinect, sample, 'video.avi'),
                            os.path.join(summary_dir, new_sample_name + '.avi'))


def summary_image_data(data_type='clean_1_rename'):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    for Kinect in Kinects:
        for subject in subjects:
            print('Make video samples from images in ' + subject + '_' + Kinect)
            destination_dir= os.path.join(image_data_dir, Kinect +'_' +data_type+'_summary',subject)
            if not os.path.exists(destination_dir):
                os.makedirs(destination_dir)
            for action in actions:
                action_folder=os.path.join(image_data_dir,Kinect+'_'+data_type,subject,action)
                samples = get_list_dir_in_folder(action_folder)
                for sample in samples:
                    image_list = get_list_jpg_in_folder(os.path.join(action_folder, sample))
                    image_list.sort()
                    destination_name=action.zfill(2)+'_'+sample.zfill(2)+'.avi'
                    video = cv2.VideoWriter(os.path.join(destination_dir,destination_name), fourcc, 10, resolution)
                    for image in image_list:
                        image_path=os.path.join(action_folder,sample,image)
                        frame=cv2.imread(image_path)
                        video.write(frame)
                    cv2.destroyAllWindows()
                    video.release()

def rename_data_after_clean(Kinect):
    clean_name='clean_1'
    kinect_clean_dir=Kinect+'_'+clean_name
    kinect_rename_dir=Kinect + '_'+clean_name+'_rename'
    print('Rename image from '  + kinect_clean_dir)
    for subject in subjects:
        for action in actions:
            action_folder = os.path.join(image_data_dir, kinect_clean_dir, subject, action)
            samples = get_list_dir_in_folder(action_folder)
            for sample in samples:
                image_list = get_list_jpg_in_folder(os.path.join(action_folder, sample))
                image_list.sort()
                if(len(image_list)<16):
                    print('len smaller than 16 in '+os.path.join(action_folder, sample))
                for i in range(len(image_list)):
                    destination_dir = os.path.join(image_data_dir, kinect_rename_dir, subject,action,sample)
                    if not os.path.exists(destination_dir):
                        os.makedirs(destination_dir)
                    new_name=str(i+1).zfill(6)+'.jpg'
                    shutil.copy(os.path.join(action_folder,sample,image_list[i]),
                            os.path.join(destination_dir,new_name))


def remove_nois_by_copy_roi(dir, roi, src_image):
    image_list = get_list_jpg_in_folder(dir)
    image_list.sort()
    img = cv2.imread(src_image)
    crop_roi = img[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
    for image_name in image_list:
        image_path=os.path.join(dir,image_name)
        image = cv2.imread(image_path)
        image[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]] = crop_roi
        cv2.imwrite(image_path, image)
        kk=1

def shift_image(dir, shift_data): #shift image x, y
    image_list = get_list_jpg_in_folder(dir)
    image_list.sort()
    x=max(-shift_data[0],0)
    y=max(-shift_data[1],0)
    new_x=max(shift_data[0],0)
    new_y=max(shift_data[1],0)
    new_w=resolution[0]-abs(shift_data[0])
    new_h=resolution[1]-abs(shift_data[1])
    for image_name in image_list:
        image_path=os.path.join(dir,image_name)
        img = cv2.imread(image_path)
        crop_roi = img[ y:y + new_h,x:x + new_w]
        new_image = np.zeros((resolution[1], resolution[0], 3), np.uint8)
        new_image[new_y:new_y + new_h,new_x:new_x + new_w] = crop_roi
        cv2.imwrite(image_path, new_image)


check_gpu_ready(allocate_mem=1330,total_gpu_mem=2002,log_time=60)



#summary_image_data(data_type='clean_1_augmented_padding_new')
#rename_data_after_clean('Kinect_3')
#remove_nois_by_copy_roi('/home/titikid/PycharmProjects/c3d_luanvan/data/Kinect_3_clean_1/Thuan/5/1',
 #                      [211,194,200,180],
  #                     '/home/titikid/PycharmProjects/c3d_luanvan/data/Kinect_3_clean_1/Thuan/5/1/000003.jpg')

# for i in range(5):
#     shift_image('/home/titikid/PycharmProjects/c3d_luanvan/data/Kinect_1_clean_1_fix_Binh/Tan/5/'+str(i+1),(-60,0))
# print('Finish.')
