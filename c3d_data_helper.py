import os
import numpy as np
import shutil
import cv2
import c3d_params
from c3d_helper import write_matlab_file

video_data_dir = os.path.join(c3d_params.c3d_data_root, 'Dataset_hand_gesture')
image_data_dir = c3d_params.c3d_data_root
Kinects = ['K1,K2,K3,K4,K5']
actions = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
resolution = (640, 480)


def get_list_dir_in_folder(dir):
    sub_dir = [o for o in os.listdir(dir) if os.path.isdir(os.path.join(dir, o))]
    return sub_dir


def get_list_file_in_folder(dir, ext='jpg'):
    included_extensions = [ext]
    file_names = [fn for fn in os.listdir(dir)
                  if any(fn.endswith(ext) for ext in included_extensions)]
    return file_names


def summary_result(folder, subjects, suffix='', max_in_each_subject=True, data_type='fc6_linear,fc6_rbf,fc7_linear,fc7_rbf,prob'):
    print('Begin summarize result in ' + folder)
    result = folder + '\n\n'
    subject_title = ''
    field_title = ''
    final_accuracy = dict()
    final_loss = dict()
    subjects = subjects.split(',')
    data_type = data_type.split(',')

    for i in range(len(subjects)):
        subject_title += subjects[i].ljust(5) + '\t\t'
        if (i == len(subjects) - 1):
            subject_title += 'Final accuracy'
        field_title += 'loss\tacc \t'
        result_folder = os.path.join(folder, subjects[i])

        final_accuracy[subjects[i]] = dict()

        accuracy = dict()
        loss = dict()
        for type in data_type:
            with open(os.path.join(result_folder, type + '_test_accuracy.txt')) as f:
                lines = f.readlines()
                x = np.array(lines)
                accuracy[type] = x.astype(np.float)
            with open(os.path.join(result_folder, type + '_test_loss.txt')) as f:
                lines = f.readlines()
                x = np.array(lines)
                loss[type] = x.astype(np.float)

        final_accuracy[subjects[i]] = accuracy
        final_loss[subjects[i]] = loss

    number_of_snapshot = len(final_accuracy[subjects[0]][data_type[0]])

    for type in data_type:
        result += type + '\n' + subject_title + '\n' + field_title + '\n'
        final_acc = []
        max_subject=dict()
        for subject in subjects:
            max_subject[subject]=0
        for i in range(number_of_snapshot):
            sum_acc = 0.
            for subject in subjects:
                if(final_accuracy[subject][type][i]>max_subject[subject]):
                    max_subject[subject]=final_accuracy[subject][type][i]

                sum_acc += final_accuracy[subject][type][i]
                res = "%.4f" % final_loss[subject][type][i] + '\t' + "%.4f" % final_accuracy[subject][type][i] + '\t'
                result += res
            avg_acc = sum_acc / len(subjects)
            final_acc.append(avg_acc)
            result += '\t' + "%.4f" % avg_acc + '\n'

        final_max_acc=0
        for subject in subjects:
            final_max_acc+=max_subject[subject]
        final_max_avg_acc=final_max_acc/len(subjects)
        if(max_in_each_subject==False):
            result += type + '_accuracy:%.4f' % max(final_acc) + '\n\n'
        else:
            result += type + '_accuracy:%.4f' % final_max_avg_acc + '\n\n'

    file_name = 'summary_' + suffix + '.txt'
    with open(os.path.join(folder, file_name), 'w') as f:
        f.writelines(result)
    print('Save result to ' + os.path.join(folder, file_name))
    return os.path.join(folder, file_name)


def summary_all_results_in_folder(folder, Kinects, subjects):
    sub_dir = get_list_dir_in_folder(folder)
    names = []
    Kinects = Kinects.split(',')
    for kinect_train in Kinects:
        for kinect_test in Kinects:
            name = kinect_train + '_' + kinect_test
            names.append(name)
    for dir in sub_dir:
        for name in names:
            if (name in dir):
                summary_result(os.path.join(folder, dir), subjects=subjects, suffix=name)
    print('Finish. Summary_all_results_in_folder ' + folder)


def summary_all_results(folder, subjects, Kinects, data_type='fc6_linear,fc6_rbf,fc7_linear,fc7_rbf,prob'):
    sub_dir = get_list_dir_in_folder(folder)
    final_acc = dict()
    header = ''
    Kinects = Kinects.split(',')
    data_type = data_type.split(',')
    summary_dir = os.path.join(folder, 'final_summary')
    num_kinect = len(Kinects)
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    for kinect_train in Kinects:
        header += '\t' + kinect_train+'   '
        for kinect_test in Kinects:
            folder_prefix = kinect_train + '_' + kinect_test
            is_folder_exist = False
            acc = []
            for dir in sub_dir:
                if (folder_prefix in dir):
                    is_folder_exist = True
                    result_file = summary_result(os.path.join(folder, dir), subjects=subjects, suffix=folder_prefix)
                    shutil.copy(result_file, os.path.join(summary_dir, folder_prefix))
                    with open(result_file) as f:
                        lines = f.readlines()
                    for i in range(len(data_type)):
                        result_line = len(lines) / len(data_type)
                        max_acc_str = lines[result_line * (i + 1)].split(':')
                        max_acc = float(max_acc_str[1].replace('\n', ''))
                        acc.append(max_acc)
            if (is_folder_exist == True):
                final_acc[folder_prefix] = acc
            else:
                final_acc[folder_prefix] = None

    single_view = dict()
    cross_view = dict()
    total_view = dict()

    num_total_view =len(data_type)* num_kinect * num_kinect
    num_single_view =len(data_type)* num_kinect
    for i in range(len(data_type)):
        t_view = 0
        s_view = 0
        for kinect_train in Kinects:
            for kinect_test in Kinects:
                folder_prefix = kinect_train + '_' + kinect_test
                if (final_acc[folder_prefix] != None):
                    t_view += final_acc[folder_prefix][i]
                    if (kinect_test == kinect_train):
                        s_view += final_acc[folder_prefix][i]
                else:
                    num_total_view=num_total_view-1
                    if (kinect_test == kinect_train):
                        num_single_view =num_single_view-1

        total_view[data_type[i]] = t_view
        single_view[data_type[i]] = s_view
        cross_view[data_type[i]] = total_view[data_type[i]] - single_view[data_type[i]]

    num_total_view=num_total_view/len(data_type)
    num_single_view=num_single_view/len(data_type)
    num_cross_view = num_total_view - num_single_view

    result = ''
    #header
    for i in range(len(data_type)):
        result += data_type[i] + '\n' + header + '\n'
        for kinect_train in Kinects:
            result += kinect_train + '\t'
            for kinect_test in Kinects:
                folder_prefix = kinect_train + '_' + kinect_test
                if(final_acc[folder_prefix]!=None):
                    result += '%.2f' % (100 * final_acc[folder_prefix][i]) + '\t'
                else:
                    result += '    \t'
            result += '\n'
        result += 'single-view: %.2f' % (100 * single_view[data_type[i]] / num_single_view) + '\n'
        result += 'cross-view:  %.2f' % (100 * cross_view[data_type[i]] / num_cross_view) + '\n'
        result += 'total-view:  %.2f' % (100 * total_view[data_type[i]] / num_total_view) + '\n'
        result += '\n'

    with open(os.path.join(summary_dir, 'final_summary.txt'), 'w') as f:
        f.writelines(result)
    print('Save result to ' + os.path.join(summary_dir, 'final_summary.txt'))


def summary_video_data(subjects, Kinects):
    subjects = subjects.split(',')
    Kinects = Kinects.split(',')
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


def summary_image_data(subjects, Kinects, data_type='clean_1_rename'):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    subjects = subjects.split(',')
    Kinects = Kinects.split(',')
    for Kinect in Kinects:
        for subject in subjects:
            print('Make video samples from images in ' + subject + '_' + Kinect)
            destination_dir = os.path.join(image_data_dir, Kinect + '_' + data_type + '_video_from_images', subject)
            if not os.path.exists(destination_dir):
                os.makedirs(destination_dir)
            for action in actions:
                action_folder = os.path.join(image_data_dir, Kinect + '_' + data_type, subject, action)
                samples = get_list_dir_in_folder(action_folder)
                for sample in samples:
                    image_list = get_list_file_in_folder(os.path.join(action_folder, sample))
                    image_list.sort()
                    destination_name = action.zfill(2) + '_' + sample.zfill(2) + '.avi'
                    video = cv2.VideoWriter(os.path.join(destination_dir, destination_name), fourcc, 10, resolution)
                    for image in image_list:
                        image_path = os.path.join(action_folder, sample, image)
                        frame = cv2.imread(image_path)
                        video.write(frame)
                    cv2.destroyAllWindows()
                    video.release()


def rename_data_after_clean(Kinect, subjects, leng=8):
    clean_name = 'segmented'
    kinect_clean_dir = Kinect + '_' + clean_name
    kinect_rename_dir = Kinect + '_' + clean_name + '_rename'
    print('Rename image from ' + kinect_clean_dir)
    subjects = subjects.split(',')
    for subject in subjects:
        for action in actions:
            action_folder = os.path.join(image_data_dir, kinect_clean_dir, subject, action)
            samples = get_list_dir_in_folder(action_folder)
            for sample in samples:
                image_list = get_list_file_in_folder(os.path.join(action_folder, sample))
                image_list.sort()
                if (len(image_list) < leng):
                    print('len smaller than ' + str(leng) + ' in ' + os.path.join(action_folder, sample))
                for i in range(len(image_list)):
                    destination_dir = os.path.join(image_data_dir, kinect_rename_dir, subject, action, sample)
                    if not os.path.exists(destination_dir):
                        os.makedirs(destination_dir)
                    new_name = str(i + 1).zfill(6) + '.jpg'
                    shutil.copy(os.path.join(action_folder, sample, image_list[i]),
                                os.path.join(destination_dir, new_name))


def remove_nois_by_copy_roi(dir, roi, src_image):
    image_list = get_list_file_in_folder(dir)
    image_list.sort()
    img = cv2.imread(src_image)
    crop_roi = img[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
    for image_name in image_list:
        image_path = os.path.join(dir, image_name)
        image = cv2.imread(image_path)
        image[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]] = crop_roi
        cv2.imwrite(image_path, image)


def shift_image(dir, shift_data):  # shift image x, y
    image_list = get_list_file_in_folder(dir)
    image_list.sort()
    x = max(-shift_data[0], 0)
    y = max(-shift_data[1], 0)
    new_x = max(shift_data[0], 0)
    new_y = max(shift_data[1], 0)
    new_w = resolution[0] - abs(shift_data[0])
    new_h = resolution[1] - abs(shift_data[1])
    for image_name in image_list:
        image_path = os.path.join(dir, image_name)
        img = cv2.imread(image_path)
        crop_roi = img[y:y + new_h, x:x + new_w]
        new_image = np.zeros((resolution[1], resolution[0], 3), np.uint8)
        new_image[new_y:new_y + new_h, new_x:new_x + new_w] = crop_roi
        cv2.imwrite(image_path, new_image)


def count_number_of_frame(video_file):
    vidcap = cv2.VideoCapture(video_file)
    success, image = vidcap.read()
    count = 1
    success = True
    while success:
        success, image = vidcap.read()
        # print 'Read a new frame: ', success
        count += 1
    return count


def count_number_of_frame_and_save_to_file(video_dir):
    subjects = get_list_dir_in_folder(video_dir)
    subjects.sort()
    for subject in subjects:
        print subject
        kinects = get_list_dir_in_folder(os.path.join(video_dir, subject))
        kinects.sort()
        for kinect in kinects:
            print kinect
            samples = get_list_dir_in_folder(os.path.join(video_dir, subject, kinect))
            samples.sort()
            for sample in samples:
                num_frm = count_number_of_frame(os.path.join(video_dir, subject, kinect, sample, 'video.avi'))
                shutil.move(os.path.join(video_dir, subject, kinect, sample, 'video.avi'),
                            os.path.join(video_dir, subject, kinect, sample, 'video_' + str(num_frm) + '.avi'))


def get_number_of_frame_from_video_name(video_file_dir):
    video_file = get_list_file_in_folder(video_file_dir, 'avi')
    name, num_frm = video_file[0].split('_')
    return int(num_frm.replace('.avi', ''))


def extract_frames_from_video(video_file_dir, output_dir='', begin_frame=1, frame_to_extract=16, ext='jpg'):
    video_file = get_list_file_in_folder(video_file_dir, 'avi')
    vidcap = cv2.VideoCapture(os.path.join(video_file_dir, video_file[0]))
    success, image = vidcap.read()
    count = 1
    success = True

    num_frm = get_number_of_frame_from_video_name(os.path.join(video_file_dir))
    sampling_rate = (num_frm - begin_frame) / frame_to_extract
    frame_id_extract = []
    for i in range(int(frame_to_extract)):
        id = round(begin_frame + i * sampling_rate)
        frame_id_extract.append(id)

    while success:
        for i in range(int(frame_to_extract)):
            if (count == frame_id_extract[i]):
                cv2.imwrite(os.path.join(output_dir, (str(i + 1)).zfill(6)) + '.' + ext,
                            image)  # save frame as JPEG file
        success, image = vidcap.read()
        # print 'Read a new frame: ', success
        count += 1


def convert_video_dataset_to_images(video_dir, image_dir, base_num_frame=16):
    drop_rate = dict()
    drop_rate['Giang'] = 4
    drop_rate['Hai'] = 3
    drop_rate['Long'] = 4
    drop_rate['Minh'] = 8
    drop_rate['Thuan'] = 4
    drop_rate['Thuy'] = 9
    drop_rate['Tuyen'] = 4
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    subjects = get_list_dir_in_folder(video_dir)
    subjects.sort()

    # chuan hoa thoi gian thuc hien tung hanh dong
    total_frame_normalize_for_each_action = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for subject in subjects:
        print subject
        kinects = get_list_dir_in_folder(os.path.join(video_dir, subject))
        kinects.sort()
        for kinect in kinects:
            print kinect
            old_samples = get_list_dir_in_folder(os.path.join(video_dir, subject, kinect))
            old_samples.sort()
            first = old_samples[:9]
            second = old_samples[9:36]
            samples = []
            samples[:27] = second
            samples[28:36] = first
            for sample in samples:
                action, id = sample.split('_')
                num_frm = get_number_of_frame_from_video_name(os.path.join(video_dir, subject, kinect, sample))
                total_frame_normalize_for_each_action[int(action) - 1] += num_frm / drop_rate[subject]

    # action 1 always fastest
    base_value = total_frame_normalize_for_each_action[0] / base_num_frame
    frame_to_extract = []
    for i in range(12):
        round_frame = round(total_frame_normalize_for_each_action[i] / base_value)
        frame_to_extract.append(round_frame)
        print('total frame normalize for action ' + str(i) + ': ' + str(
            total_frame_normalize_for_each_action[i]) + ', round frame: ' + str(round_frame))

    for subject in subjects:
        print '\nExtract frame from: ' + subject + ', drop_rate: ' + str(drop_rate[subject]),
        kinects = get_list_dir_in_folder(os.path.join(video_dir, subject))
        kinects.sort()
        for kinect in kinects:
            print '\n', kinect,
            old_samples = get_list_dir_in_folder(os.path.join(video_dir, subject, kinect))
            old_samples.sort()
            first = old_samples[:9]
            second = old_samples[9:36]
            samples = []
            samples[:27] = second
            samples[28:36] = first
            count = 1
            print 'Num_frame: ',
            for sample in samples:
                action, id = sample.split('_')
                destination_dir = os.path.join(image_dir, subject, kinect, action, str(count))

                if not os.path.exists(destination_dir):
                    os.makedirs(destination_dir)
                begin_frm = 1
                if (subject == 'Thuan' and action == '3' and count == 3):
                    begin_frm = 6
                if (subject == 'Thuy'):
                    if (action == '3'):
                        if (count == 2):
                            begin_frm = 10
                        if (count == 3):
                            begin_frm = 20
                    if (action == '6' and count == 3):
                        begin_frm = 10
                    if (action == '7' and count == 3):
                        begin_frm = 10
                print int(frame_to_extract[int(action) - 1]),
                extract_frames_from_video(os.path.join(video_dir, subject, kinect, sample), destination_dir, begin_frm,
                                          frame_to_extract=frame_to_extract[int(action) - 1])
                if (count % 3 == 0):
                    print ',', 26
                    count = 1
                else:
                    count += 1
    print('\nFinish.')


def reorder_dataset(video_dir, new_video_dir):
    subjects = get_list_dir_in_folder(video_dir)
    subjects.sort()
    for subject in subjects:
        print(subject)
        kinects = get_list_dir_in_folder(os.path.join(video_dir, subject))
        kinects.sort()
        for kinect in kinects:
            print(kinect)
            actions = get_list_dir_in_folder(os.path.join(video_dir, subject, kinect))
            actions.sort()
            for action in actions:
                print(action)
                action_dir = os.path.join(video_dir, subject, kinect, action)
                samples = get_list_dir_in_folder(action_dir)
                samples.sort()
                for sample in samples:
                    destination_dir = os.path.join(new_video_dir, kinect + '_original', subject, action, sample)
                    if not os.path.exists(destination_dir):
                        os.makedirs(destination_dir)
                    images = get_list_file_in_folder(os.path.join(action_dir, sample))
                    for image in images:
                        shutil.copy(os.path.join(action_dir, sample, image), os.path.join(destination_dir, image))

def summary_matlab_result(table_set_result_folder,subjects, Kinects, data_type='fc6', max_iter=700, snapshot=100):
    sub_dir = get_list_dir_in_folder(table_set_result_folder)
    Kinects = Kinects.split(',')
    Subjects = subjects.split(',')
    data_type = data_type.split(',')
    num_kinect = len(Kinects)

    feature_vector=dict()
    label_vector=dict()
    feature_init=dict()
    label_init=dict()
    for kinect_test in Kinects:
        feature_init[kinect_test]=False
        label_init[kinect_test]=False

    import scipy.io
    for kinect_train in Kinects:
        for kinect_test in Kinects:
            if(kinect_test!=kinect_train):
                continue
            folder_prefix = kinect_train + '_' + kinect_test
            is_folder_exist = False
            acc = []
            for dir in sub_dir:
                if (folder_prefix in dir):
                    print(dir)
                    for subject in Subjects:
                        for iter in range(snapshot,max_iter+1,snapshot):
                            matlab_file_dir=os.path.join(table_set_result_folder,dir,subject,'iter_'+str(iter),'output_vector')
                            if not os.path.exists(matlab_file_dir):
                                continue
                            matlab_file=get_list_file_in_folder(matlab_file_dir, ext='mat')
                            for file in matlab_file:
                                if 'label' in file:
                                    label=scipy.io.loadmat(os.path.join(matlab_file_dir,file))
                                    if (label_init[kinect_test] == True):
                                        label_vector[kinect_test] = np.concatenate(
                                            (label_vector[kinect_test], label[file.replace('.mat', '')]), axis=1)
                                    else:
                                        label_vector[kinect_test] = label[file.replace('.mat', '')]
                                        label_init[kinect_test] = True
                                else:
                                    feature = scipy.io.loadmat(os.path.join(matlab_file_dir, file))
                                    if (feature_init[kinect_test] == True):
                                        feature_vector[kinect_test] = np.concatenate(
                                            (feature_vector[kinect_test], feature[file.replace('.mat', '')]), axis=1)
                                    else:
                                        feature_vector[kinect_test] = feature[file.replace('.mat', '')]
                                        feature_init[kinect_test] = True

    for kinect in Kinects:
        print('Save matlab file for: ' +kinect)

        label_transpose = label_vector[kinect].transpose(1, 0)
        feature_transpose = feature_vector[kinect].transpose(1, 0)

        label_vector_sorted_idx=np.argsort(label_transpose,axis=0)

        label_vector_sorted=label_transpose[label_vector_sorted_idx]
        feature_vector_sorted=feature_transpose[label_vector_sorted_idx]
        label_vector_sorted_write=label_vector_sorted.reshape(216,1)
        feature_vector_sorted_write=feature_vector_sorted.reshape(216,4096)
        write_matlab_file(feature_vector_sorted_write, label_vector_sorted_write, table_set_result_folder, kinect)
    print('Finish.')

def extract_segmented_image():
    import cv2
    input_RGB = '/home/prdcv/PycharmProjects/c3d_luanvan/data/12gestures_images_RGB'
    input_mask = '/home/prdcv/PycharmProjects/c3d_luanvan/data/12gestures_images_mask'
    output_segmented = '/home/prdcv/PycharmProjects/c3d_luanvan/data/12gestures_images_segmented'
    subject_list = 'Giang,Hai,Long,Minh,Thuan,Thuy,Tuyen'
    subjects = subject_list.split(',')
    kinects = 'K2'
    kinects = kinects.split(',')
    gestures = 12
    sample_per_ges = 3
    data_type = 'original'

    success=0
    failed=0
    for kinect in kinects:
        print(kinect)
        for subject in subjects:
            print(subject)
            for gesture in range(gestures):
                for sample in range(sample_per_ges):
                    sample_path = os.path.join(kinect + '_' + data_type, subject, str(gesture + 1), str(sample + 1))
                    sample_RGB_dir = os.path.join(input_RGB, sample_path)
                    sample_mask_dir = os.path.join(input_mask, sample_path)
                    sample_segemented_dir = os.path.join(output_segmented, sample_path)

                    if not os.path.exists(sample_segemented_dir):
                        os.makedirs(sample_segemented_dir)

                    images_RGB = get_list_file_in_folder(sample_RGB_dir)
                    images_RGB.sort()
                    images_mask = get_list_file_in_folder(sample_mask_dir)
                    images_mask.sort()
                    for idx in range (len(images_mask)):
                        statinfo = os.stat(os.path.join(sample_mask_dir,images_mask[idx]))
                        sz=statinfo.st_size
                        if(sz==0):
                            failed+=1
                            continue
                        else:
                            success+=1
                            img_rgb=cv2.imread(os.path.join(sample_RGB_dir,images_RGB[idx]), -1)
                            mask=cv2.imread(os.path.join(sample_mask_dir,images_mask[idx]))
                            final =cv2.bitwise_and(img_rgb, mask)
                            cv2.imwrite(os.path.join(sample_segemented_dir,images_mask[idx]),final)

    print(str(success))
    print(str(failed))
    print(str(failed+success))
    success_rate=(float)(success)/(float)(failed+success)
    print('Finish. success rate = '+str(success_rate))








if __name__ == "__main__":
    # reorder_dataset('/home/titikid/PycharmProjects/12gestures_images',
    #                '/home/titikid/PycharmProjects/12gestures_images_new')
    # for i in range (5):
    #    rename_data_after_clean('K' +str(i+1))
    # summary_all_results_in_folder('output/result_26Jan')
    # summary_9_results('output/result/new')
    # extract_frames_from_video('/home/prdcv/PycharmProjects/gvh205/UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi','/home/prdcv/PycharmProjects/gvh205/test')
    # data_dir='/home/prdcv/PycharmProjects/c3d_luanvan/data/'
    # count_number_of_frame_and_save_to_file(data_dir +'12gestures_video')
    # convert_video_dataset_to_images(data_dir +'12gestures_video',data_dir +'12gestures_images')
    #extract_segmented_image()
    #summary_matlab_result('output/result_backup/table_2019_02_16_original_pre_3_12gestures_6_subjects', Kinects=c3d_params.Kinects, subjects=c3d_params.subject_list)
    #summary_all_results('output/result_backup/table_2019_02_27_original_pre_3_12gestures_6_subjects_fc6_1024_fc7_1024', Kinects=c3d_params.Kinects, subjects=c3d_params.subject_list)
    summary_result('output/result/K1_K1_2019-05-13_13.30',max_in_each_subject=True, subjects=c3d_params.subject_list)
    #summary_image_data( subjects=c3d_params.subject_list,  Kinects=c3d_params.Kinects, data_type='original')
    # rename_data_after_clean('Kinect_3')
    # remove_nois_by_copy_roi('/home/titikid/PycharmProjects/c3d_luanvan/data/Kinect_3_clean_1/Thuan/5/1',
    #                      [211,194,200,180],
    #                     '/home/titikid/PycharmProjects/c3d_luanvan/data/Kinect_3_clean_1/Thuan/5/1/000003.jpg')

    # for i in range(5):
    #     shift_image('/home/titikid/PycharmProjects/c3d_luanvan/data/Kinect_1_clean_1_fix_Binh/Tan/5/'+str(i+1),(-60,0))
    # print('Finish.')
