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

def summary_result(folder, subjects=['Binh','Giang','Hung','Tan','Thuan'], data_type=['fc6_linear','fc6_rbf','fc7_linear','fc7_rbf','prob']):
    print('Begin summarize result in '+folder)
    result=folder+'\n\n'
    subject_title=''
    field_title = ''
    final_accuracy=[]
    final_loss=[]

    for i in range(len(subjects)):
        subject_title+=subjects[i].ljust(5)+'\t\t'
        if(i==len(subjects)-1):
            subject_title+='Final accuracy'
        field_title+='loss\tacc \t'
        result_folder=os.path.join(folder,subjects[i])

        accuracy=[]
        loss=[]
        for type in data_type:
            with open(os.path.join(result_folder, type+'_test_accuracy.txt')) as f:
                lines = f.readlines()
                x = np.array(lines)
                accuracy.append(x.astype(np.float))
            with open(os.path.join(result_folder, type+'_test_loss.txt')) as f:
                lines = f.readlines()
                x = np.array(lines)
                loss.append(x.astype(np.float))

        final_accuracy.append(accuracy)
        final_loss.append(loss)

    for n in range(len(data_type)):
        result +=data_type[n]+'\n' + subject_title + '\n' + field_title + '\n'
        final_acc = []
        for j in range(len(final_accuracy[n][0])):
            sum_acc = 0.
            for k in range((len(final_accuracy[n]))):
                sum_acc += final_accuracy[k][n][j]
                res = "%.4f" % final_loss[k][n][j] + '\t' + "%.4f" % final_accuracy[k][n][j] + '\t'
                result += res
            avg_acc = sum_acc / len(final_accuracy[n])
            final_acc.append(avg_acc)
            result += '\t' + "%.4f" % avg_acc + '\n'
        result += data_type[n]+ '_accuracy:%.4f' % max(final_acc) + '\n\n'


    with open(os.path.join(folder,'summary.txt'),'w') as f:
        f.writelines(result)
    print('Save result to '+os.path.join(folder,'summary.txt'))
    return os.path.join(folder,'summary.txt')

def summary_all_results_in_folder(folder):
    sub_dir = get_list_dir_in_folder(folder)
    names=['K1_K1','K1_K3','K1_K5','K3_K1','K3_K3','K3_K5','K5_K1','K5_K3','K5_K5']
    for dir in sub_dir:
        for name in names:
            if(name in dir):
                summary_result(os.path.join(folder,dir))
    print('Finish.')



def summary_9_results(folder, Kinects=['Kinect_1','Kinect_3','Kinect_5'],data_type=['fc6_linear','fc6_rbf','fc7_linear','fc7_rbf','prob']):
    sub_dir= get_list_dir_in_folder(folder)
    final_acc=[]
    header='\t\t'
    summary_dir=os.path.join(folder, 'final_summary')
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    for kinect_train in Kinects:
        header+=' || '+kinect_train
        for kinect_test in Kinects:
            folder_prefix=kinect_train+'_test_on_'+kinect_test
            for dir in sub_dir:
                if (folder_prefix in dir):
                    result_file=summary_result(os.path.join(folder, dir))
                    new_name='K'+ kinect_train.split('_')[1]+'_K'+ kinect_test.split('_')[1]
                    shutil.copy(result_file, os.path.join(summary_dir,new_name))
                    with open(result_file) as f:
                        lines = f.readlines()
                    acc=[]
                    for i in range(len(data_type)):
                        result_line=len(lines)/len(data_type)
                        max_acc_str=lines[result_line*(i+1)].split(':')
                        max_acc=float(max_acc_str[1].replace('\n',''))
                        acc.append(max_acc)
            final_acc.append(acc)
    x = np.array(final_acc)
    final_acc=x.transpose(1,0).tolist()

    single_view =[]
    cross_view =[]
    for i in range(len(data_type)):
        single_view.append((final_acc[i][0]+final_acc[i][4]+final_acc[i][8])/3)
        cross_view.append((final_acc[i][1]+final_acc[i][2]+final_acc[i][3]+final_acc[i][5]+final_acc[i][6]+final_acc[i][7])/6)

    result=''
    header += ' ||'
    for i in range(len(data_type)):
        result+=data_type[i]+'\n'+header+'\n'
        count=0
        for kinect_train in Kinects:
            result += kinect_train + ' ||'
            for kinect_test in Kinects:
                result+= '  %.2f' % (100*final_acc[i][count])+'   ||'
                count+=1
            result +='\n'
        result += '\t\tsingle-view: %.2f' % (100 * single_view[i])+'\n'
        result += '\t\tcross-view:  %.2f' % (100 * cross_view[i])+'\n'
        result +='\n'


    with open(os.path.join(summary_dir,'final_summary.txt'),'w') as f:
        f.writelines(result)
    print('Save result to '+os.path.join(summary_dir,'final_summary.txt'))


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

summary_all_results_in_folder('output/experiments')
summary_9_results('output/result_set_table_22Jan2019_segmented')
#summary_result('output/Kinect_1_test_on_Kinect_1_22-01-2019_16.43.03')
#check_gpu_ready(allocate_mem=1330,total_gpu_mem=2002,log_time=60)



#summary_image_data(data_type='clean_1_augmented_padding_new')
#rename_data_after_clean('Kinect_3')
#remove_nois_by_copy_roi('/home/titikid/PycharmProjects/c3d_luanvan/data/Kinect_3_clean_1/Thuan/5/1',
 #                      [211,194,200,180],
  #                     '/home/titikid/PycharmProjects/c3d_luanvan/data/Kinect_3_clean_1/Thuan/5/1/000003.jpg')

# for i in range(5):
#     shift_image('/home/titikid/PycharmProjects/c3d_luanvan/data/Kinect_1_clean_1_fix_Binh/Tan/5/'+str(i+1),(-60,0))
# print('Finish.')
