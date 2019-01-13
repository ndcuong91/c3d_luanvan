import cv2
import os
import numpy as np
import shutil
import cv2


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


def summary_image_data():
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    for Kinect in Kinects:
        for subject in subjects:
            print('Make video samples from images in ' + subject + '_' + Kinect)
            destination_dir= os.path.join(image_data_dir, Kinect + '_original_summary',subject)
            if not os.path.exists(destination_dir):
                os.makedirs(destination_dir)
            for action in actions:
                action_folder=os.path.join(image_data_dir,Kinect+'_original',subject,action)
                samples = get_list_dir_in_folder(action_folder)
                for sample in samples:
                    image_list = get_list_jpg_in_folder(os.path.join(action_folder, sample))
                    image_list.sort()
                    destination_name=action.zfill(2)+'_'+sample.zfill(2)+'.avi'
                    video = cv2.VideoWriter(os.path.join(destination_dir,destination_name), fourcc, 5, resolution)
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
                for i in range(len(image_list)):
                    destination_dir = os.path.join(image_data_dir, kinect_rename_dir, subject,action,sample)
                    if not os.path.exists(destination_dir):
                        os.makedirs(destination_dir)
                    new_name=str(i+1).zfill(6)+'.jpg'
                    shutil.copy(os.path.join(action_folder,sample,image_list[i]),
                            os.path.join(destination_dir,new_name))





#summary_image_data()
rename_data_after_clean('Kinect_1')
print('Finish.')



# height,width,layers=img[1].shape
#
# video=cv2.VideoWriter('video.avi',-1,10,(width,height))
#
# for j in range(0,5):
#     video.write(img)
#
# cv2.destroyAllWindows()
# video.release()