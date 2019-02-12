import cv2
import os
import numpy as np
import c3d_params
from c3d_data_helper import get_list_dir_in_folder, get_list_file_in_folder

method = 3
data_type='original'

data_folder = c3d_params.c3d_data_root
old_res = [640, 480]
padding = [100, 75]
do_padding = True

# method for augmented with crop size 560 x 420
new_res = [560, 420]
crop_set_augment_1 = []
crop_set_1 = [[0, 0], [80, 0], [40, 30], [80, 60], [0, 60]]  # for action 1,3,5
crop_set_2 = [[0, 0], [20, 0], [40, 0], [60, 0], [80, 0]]  # for action 2
crop_set_3 = [[80, 0], [80, 15], [80, 30], [80, 45], [80, 60]]  # for action 4
crop_set_augment_1.append(crop_set_1)
crop_set_augment_1.append(crop_set_2)
crop_set_augment_1.append(crop_set_3)


def aug_1(data_folder, kinect_folder, target_folder, subject, action, crop_set):
    action_folder = os.path.join(data_folder, kinect_folder, subject, str(action))
    action_folder_augmented = os.path.join(data_folder, target_folder, subject, str(action))
    samples = get_list_dir_in_folder(action_folder)
    for sample in samples:
        if (len(sample) > 1):
            continue
        image_list = get_list_file_in_folder(action_folder + '/' + sample)
        for i in range(len(crop_set)):
            new_folder = sample + '_' + str(i + 1)
            if not os.path.exists(action_folder_augmented + '/' + new_folder):
                os.makedirs(action_folder_augmented + '/' + new_folder)
            for image in image_list:
                img = cv2.imread(action_folder + '/' + sample + '/' + image)
                crop_img = img[crop_set[i][1]:crop_set[i][1] + new_res[1], crop_set[i][0]:crop_set[i][0] + new_res[0]]
                if (do_padding == True):
                    padding_image = np.zeros((2 * padding[1] + new_res[1], 2 * padding[0] + new_res[0], 3), np.uint8)
                    padding_image[padding[1]:padding[1] + new_res[1], padding[0]:padding[0] + new_res[0]] = crop_img
                    cv2.imwrite(os.path.join(action_folder_augmented, new_folder, image), padding_image)
                else:
                    cv2.imwrite(os.path.join(action_folder_augmented, new_folder, image), crop_img)


# method for augmented with crop size 320 x 320, focus on motion
new_res_new = [256, 192]
begin_pt = [110, 30]
pos_set = [[0, 0], [50, 50], [100, 100], [0, 100], [100, 0]]
final_pos = []
for pos in pos_set:
    temp_pos = [pos[0] + begin_pt[0], pos[1] + begin_pt[1]]
    final_pos.append(temp_pos)


def pre_aug_2(data_folder, kinect_folder, target_folder, subject, action, final_pos):
    with open(os.path.join(data_folder, kinect_folder, subject, 'pre_aug_2')) as f:
        lines = f.readlines()
    crop_rect = [320, 320]
    crop_pos = lines[action - 1].replace('\n', '').split(',')
    x = int(crop_pos[0])
    y = int(crop_pos[1])

    action_folder = os.path.join(data_folder, kinect_folder, subject, str(action))
    action_folder_augmented = os.path.join(data_folder, target_folder, subject, str(action))
    samples = get_list_dir_in_folder(action_folder)

    for sample in samples:
        if (len(sample) > 1):
            continue
        image_list = get_list_file_in_folder(action_folder + '/' + sample)
        for i in range(len(final_pos)):
            new_folder = sample + '_' + str(i + 1)
            if not os.path.exists(action_folder_augmented + '/' + new_folder):
                os.makedirs(action_folder_augmented + '/' + new_folder)
            for image in image_list:
                img = cv2.imread(action_folder + '/' + sample + '/' + image)
                crop_img = img[y:y + crop_rect[1], x:x + crop_rect[0]]
                final_image = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
                final_image[final_pos[i][1]:final_pos[i][1] + crop_rect[1],
                final_pos[i][0]:final_pos[i][0] + crop_rect[0]] = crop_img
                cv2.imwrite(os.path.join(action_folder_augmented, new_folder, image), final_image)


# method for augmented which center by hand
def pre_3_aug(data_folder, kinect_folder, target_folder, subject, action, subject_id):
    with open(os.path.join(data_folder, kinect_folder, 'pre_3_' + kinect_folder)) as f:
        lines = f.readlines()
    hand_pos = lines[subject_id].replace('\n', '').split(',')
    shift_x = old_res[0] / 2 - int(hand_pos[0])
    shift_y = old_res[1] / 2 - int(hand_pos[1])

    action_folder = os.path.join(data_folder, kinect_folder, subject, str(action))
    action_folder_augmented = os.path.join(data_folder, target_folder, subject, str(action))
    samples = get_list_dir_in_folder(action_folder)

    shift = 20
    new_shift = [[-shift, -shift], [-shift, shift], [shift, -shift], [0, 0], [shift, shift]]

    for sample in samples:
        if (len(sample) > 1):
            continue
        image_list = get_list_file_in_folder(action_folder + '/' + sample)
        for i in range(len(new_shift)):
            new_folder = sample + '_' + str(i + 1)
            if not os.path.exists(action_folder_augmented + '/' + new_folder):
                os.makedirs(action_folder_augmented + '/' + new_folder)
            for image in image_list:
                img = cv2.imread(action_folder + '/' + sample + '/' + image)
                M = np.float32([[1, 0, shift_x + new_shift[i][0]], [0, 1, shift_y + new_shift[i][1]]])
                shift_image = cv2.warpAffine(img, M, (old_res[0], old_res[1]))
                cv2.imwrite(os.path.join(action_folder_augmented, new_folder, image), shift_image)


def pre_3(data_folder, kinect_folder, target_folder, subject, action, subject_id):
    with open(os.path.join(data_folder, kinect_folder, 'pre_3_' + kinect_folder)) as f:
        lines = f.readlines()
    hand_pos = lines[subject_id].replace('\n', '').split(',')
    shift_x = old_res[0] / 2 - int(hand_pos[0])
    shift_y = old_res[1] / 2 - int(hand_pos[1])

    action_folder = os.path.join(data_folder, kinect_folder, subject, str(action))
    action_folder_augmented = os.path.join(data_folder, target_folder, subject, str(action))
    samples = get_list_dir_in_folder(action_folder)

    for sample in samples:
        if (len(sample) > 1):
            continue
        image_list = get_list_file_in_folder(action_folder + '/' + sample)
        if not os.path.exists(action_folder_augmented + '/' + sample):
            os.makedirs(action_folder_augmented + '/' + sample)
        for image in image_list:
            img = cv2.imread(action_folder + '/' + sample + '/' + image)
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            shift_image = cv2.warpAffine(img, M, (old_res[0], old_res[1]))
            cv2.imwrite(os.path.join(action_folder_augmented, sample, image), shift_image)



if __name__ == "__main__":
    Kinects = c3d_params.Kinects.split(',')
    subjects = c3d_params.subject_list.split(',')
    actions = c3d_params.actions

    for kinect in Kinects:
        kinect_folder = kinect + '_'+data_type
        print('\nBegin preprocessing data with method ' + str(method) + ' for ' + kinect_folder)
        for n in range(len(subjects)):
            print('\nAugment subject ' + subjects[n]+', action: '),
            if (method == 1):
                suffix='aug_1'
                target_folder = kinect + '_'+data_type+'_'+suffix
                for i in range(len(actions)):
                    print(str(i + 1)),
                    aug_1(data_folder, kinect_folder, target_folder, subjects[n], actions[i])
            if (method == 2):
                suffix='pre_aug_2'
                target_folder = kinect + '_'+data_type+'_'+suffix
                for i in range(len(actions)):
                    print(str(i + 1)),
                    pre_aug_2(data_folder, kinect_folder, target_folder, subjects[n], actions[i],
                                    final_pos)
            if (method == 3):
                suffix='pre_3'
                target_folder = kinect + '_'+data_type+'_'+suffix
                for i in range(len(actions)):
                    print(str(i + 1)),
                    pre_3(data_folder, kinect_folder, target_folder, subjects[n],
                          actions[i], n)

    print('Finish.')
