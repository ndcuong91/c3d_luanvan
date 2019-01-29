from crossview import c3d_train_and_test, Result
import crossview
import argparse
import os, shutil
from python_utils import list_all_folders_in_a_directory, create_folder
import numpy as np
import random
from datetime import datetime
from c3d_helper import delete_files_with_extension_in_folder, dump_plot_to_image_file, get_params_info
import c3d_params
from c3d_data_helper import summary_result

def parse_args():
    """
    Written by by CuongND (nguyenduycuong2004@gmail.com)
    """
    parser = argparse.ArgumentParser(description='Train C3D hand-gesture networks by CuongND')
    parser.add_argument('--num_action', type=int, default=12,
                        help="Number of action.")
    parser.add_argument('--step_size', type=int, default=500,  # coco
                        help='Step to decrease learning rate.')
    parser.add_argument('--max_iter', type=int, default=500,
                        help='Maximum iteration to stop. ')
    parser.add_argument('--snapshot', type=int, default=100,
                        help='Number of periodic save params')
    parser.add_argument('--kinect_train', type=str, default='K1',
                        help='trainning set')
    parser.add_argument('--kinect_test_list', type=str, default='K1',
                        help='List of Kinect view for test. e.g: "K1,K3,K5"')
    parser.add_argument('--average_feature', type=bool, default=False,
                        help='average feature or use only first x frames')

    #optional params (already in c3d_params.py)
    parser.add_argument('--data_type_train', type=str, default='',
                        help='original, segmented...')
    parser.add_argument('--data_type_test', type=str, default='',
                        help='original, segmented...')
    parser.add_argument('--base_lr', type=float, default=0.0,
                        help="Base learning rate for fine-tuning.")
    parser.add_argument('--gamma', type=float, default=0.0,
                        help='gamma for learning rate')
    parser.add_argument('--subject_list', type=str, default='',
                        help='subject to training and test')
    parser.add_argument('--subject_test', type=str, default='',
                        help='subject to training and test')
    parser.add_argument('--batch_size_test', type=int, default=0,
                        help='batch size for feature extraction.')
    parser.add_argument('--batch_size_finetune', type=int, default=0,
                        help='batch size for fine-tuning.')

    # 23Jan. CuongND. Add parameters for modify c3d
    parser.add_argument('--resize', type=str, default='171,128',
                        help='input size of C3D')
    parser.add_argument('--crop', type=int, default=112,
                        help='crop size of C3D')
    parser.add_argument('--num_frame', type=int, default=16,
                        help='number of frame for input')
    parser.add_argument('--conv1a', type=int, default=64,
                        help='Number of filter in conv1a')
    parser.add_argument('--conv2a', type=int, default=128,
                        help='Number of filter in conv2a')
    parser.add_argument('--conv3a', type=int, default=256,
                        help='Number of filter in conv3a')
    parser.add_argument('--conv3b', type=int, default=256,
                        help='Number of filter in conv3b')
    parser.add_argument('--conv4a', type=int, default=512,
                        help='Number of filter in conv4a')
    parser.add_argument('--conv4b', type=int, default=512,
                        help='Number of filter in conv4b')
    parser.add_argument('--conv5a', type=int, default=512,
                        help='Number of filter in conv5a')
    parser.add_argument('--conv5b', type=int, default=512,
                        help='Number of filter in conv5b')
    parser.add_argument('--fc6', type=int, default=4096,
                        help='Number of neural in fc6')
    parser.add_argument('--fc7', type=int, default=4096,
                        help='Number of neural in fc7')
    args = parser.parse_args()
    return args


class ConfigParams(object):
    """
    Written by Dang Manh Truong (dangmanhtruong@gmail.com), modified by CuongND (nguyenduycuong2004@gmail.com)
    """
    args = parse_args()
    image_type = "jpg"
    num_of_actions = args.num_action
    base_lr = args.base_lr
    gamma = args.gamma
    step_size = args.step_size
    max_iter = args.max_iter
    snapshot = args.snapshot

    if(args.base_lr!=0.0):
        base_lr = args.base_lr
    else:
        base_lr = c3d_params.base_lr
    if(args.gamma!=0.0):
        gamma = args.gamma
    else:
        gamma = c3d_params.gamma
    if(args.batch_size_test!=0):
        batch_size = args.batch_size_test  # For feature extraction
    else:
        batch_size = c3d_params.batch_size_test  # For feature extraction
    if(args.batch_size_finetune!=0):
        batch_size_finetune = args.batch_size_finetune  # For finetuning
    else:
        batch_size_finetune = c3d_params.batch_size_finetune  # For finetuning

    if(args.subject_list!=''):
        subject_list = [x.strip() for x in args.subject_list.split(',')]
    else:
        subject_list = [x.strip() for x in c3d_params.subject_list.split(',')]

    if(args.subject_test!=''):
        subject_test = [x.strip() for x in args.subject_test.split(',')]
    else:
        subject_test = [x.strip() for x in c3d_params.subject_test.split(',')]
    if(args.data_type_train!=''):
        data_type_train = args.data_type_train
    else:
        data_type_train = c3d_params.data_type_train
    if(args.data_type_test!=''):
        data_type_test = args.data_type_test
    else:
        data_type_test = c3d_params.data_type_test

    kinect_train = args.kinect_train
    kinect_test_list = [x.strip() for x in args.kinect_test_list.split(',')]
    average_feature = args.average_feature

    # CuongND. Modify C3D structure
    c3d_default = dict()
    c3d_default['resize_w'] = 171
    c3d_default['resize_h'] = 128
    c3d_default['crop'] = 112
    c3d_default['num_frame'] = 16
    c3d_default['conv1a'] = 64
    c3d_default['conv2a'] = 128
    c3d_default['conv3a'] = 256
    c3d_default['conv3b'] = 256
    c3d_default['conv4a'] = 512
    c3d_default['conv4b'] = 512
    c3d_default['conv5a'] = 512
    c3d_default['conv5b'] = 512
    c3d_default['fc6'] = 4096
    c3d_default['fc7'] = 4096
    resize_w = int(args.resize.split(',')[0])
    resize_h = int(args.resize.split(',')[1])
    crop = args.crop
    num_frame = args.num_frame
    conv1a = args.conv1a
    conv2a = args.conv2a
    conv3a = args.conv3a
    conv3b = args.conv3b
    conv4a = args.conv4a
    conv4b = args.conv4b
    conv5a = args.conv5a
    conv5b = args.conv5b
    fc6 = args.fc6
    fc7 = args.fc7

    output_dir = c3d_params.output_dir
    template_dir = c3d_params.template_dir
    c3d_data_root = c3d_params.c3d_data_root
    c3d_files_dir = c3d_params.c3d_files_dir
    tool_dir = c3d_params.tool_dir

    output_vector = 'intermediate_output_vector'
    date_time = datetime.now().strftime('%Y-%m-%d_%H.%M')  # date time when start training

    # c3d_feature_dir = "/home/dangmanhtruong95/Truong_Python_run_scripts/C3D_twostream_finetuning_with_confusion_matrix_and_loss_for_train_and_test/C3D_feature_dir"
    c3d_template_dir = os.path.join(template_dir, "Original")

    c3d_pretrained_model_and_volume_mean_dir = os.path.join(c3d_files_dir, "pretrained_model_and_volume_mean")
    c3d_finetuning_dir = os.path.join(c3d_files_dir, "finetuning")
    c3d_feature_extraction_for_train_dir = os.path.join(c3d_files_dir, "feature_extraction_for_train")
    c3d_feature_extraction_for_test_dir = os.path.join(c3d_files_dir, "feature_extraction_for_test")
    c3d_intermediate_model_snapshot_dir = os.path.join(output_dir, "intermediate_model_snapshot")

    c3d_compute_volume_mean_sh = os.path.join(c3d_pretrained_model_and_volume_mean_dir,
                                              "c3d_sport1m_compute_volume_mean.sh")
    c3d_pretrained_model = os.path.join(c3d_pretrained_model_and_volume_mean_dir,
                                        "conv3d_deepnetA_sport1m_iter_1900000")
    c3d_volume_mean_file = os.path.join(c3d_pretrained_model_and_volume_mean_dir, "c3d_sport1m_volume_mean.binaryproto")

    c3d_finetuning_solver = os.path.join(c3d_finetuning_dir, "c3d_sport1m_finetuning_solver.prototxt")
    c3d_finetuning_train = os.path.join(c3d_finetuning_dir, "c3d_sport1m_finetuning_train.prototxt")
    c3d_finetuning_sh = os.path.join(c3d_finetuning_dir, "c3d_sport1m_finetuning.sh")

    c3d_feature_extraction_train_prototxt = os.path.join(c3d_feature_extraction_for_train_dir,
                                                         "c3d_sport1m_feature_extractor_frm_train.prototxt")
    c3d_feature_extraction_train_sh = os.path.join(c3d_feature_extraction_for_train_dir,
                                                   "c3d_sport1m_feature_extractor_train.sh")
    c3d_feature_extraction_test_prototxt = os.path.join(c3d_feature_extraction_for_test_dir,
                                                        "c3d_sport1m_feature_extractor_frm_test.prototxt")
    c3d_feature_extraction_test_sh = os.path.join(c3d_feature_extraction_for_test_dir,
                                                  "c3d_sport1m_feature_extractor_test.sh")

    snapshot_prefix = os.path.join(c3d_intermediate_model_snapshot_dir, "c3d_sport1m_finetune_whole")
    c3d_intermediate_model_snapshot = snapshot_prefix + "_iter_" + str(snapshot)
    c3d_intermediate_model_snapshot_solverstate = c3d_intermediate_model_snapshot + ".solverstate"

    c3d_train_01 = os.path.join(c3d_files_dir, "train_01.lst")
    c3d_train_01_output = os.path.join(c3d_files_dir, "train_01_output.lst")
    c3d_test_01 = os.path.join(c3d_files_dir, "test_01.lst")
    c3d_test_01_output = os.path.join(c3d_files_dir, "test_01_output.lst")

    c3d_template_dir = os.path.join(template_dir, "Original")
    c3d_template_compute_volume_mean_dir = os.path.join(c3d_template_dir, "compute_volume_mean")
    c3d_template_finetuning_dir = os.path.join(c3d_template_dir, "finetuning")
    c3d_template_training_dir = os.path.join(c3d_template_dir, "training")
    c3d_template_feature_extraction_dir = os.path.join(c3d_template_dir, "feature_extraction")
    c3d_template_visualization_dir = os.path.join(c3d_template_dir, "visualization")

    c3d_template_compute_volume_mean_sh = os.path.join(c3d_template_compute_volume_mean_dir, \
                                                       "c3d_sport1m_compute_volume_mean.sh")

    c3d_template_finetuning_solver = os.path.join(c3d_template_finetuning_dir, \
                                                  "c3d_sport1m_finetuning_solver.prototxt")
    c3d_template_finetuning_train = os.path.join(c3d_template_finetuning_dir, \
                                                 "c3d_sport1m_finetuning_train.prototxt")
    c3d_template_finetuning_sh = os.path.join(c3d_template_finetuning_dir, \
                                              "c3d_sport1m_finetuning.sh")

    c3d_template_feature_extractor_frm = os.path.join(c3d_template_feature_extraction_dir, \
                                                      "c3d_sport1m_feature_extractor_frm.prototxt")
    c3d_template_feature_extractor_sh = os.path.join(c3d_template_feature_extraction_dir, \
                                                     "c3d_sport1m_feature_extractor.sh")


if __name__ == "__main__":
    config_params = ConfigParams
    num_of_iters = config_params.max_iter / config_params.snapshot

    # CuongND. delete all .fc6, .fc7, .prob in old training
    print "\nPROGRAM BEGIN!\n"

    result_dir = os.path.join(config_params.output_dir,
                              config_params.output_vector)
    delete_files_with_extension_in_folder(result_dir, '.fc6')
    delete_files_with_extension_in_folder(result_dir, '.fc7')
    delete_files_with_extension_in_folder(result_dir, '.prob')

    avg_result = {}
    for kinect_test in config_params.kinect_test_list:
        avg_result[kinect_test] = {}
        avg_result[kinect_test]["fc6_linear"] = {}
        avg_result[kinect_test]["fc6_rbf"] = {}
        avg_result[kinect_test]["fc7_linear"] = {}
        avg_result[kinect_test]["fc7_rbf"] = {}
        avg_result[kinect_test]["prob"] = {}
        avg_result[kinect_test]["fc6_linear"]["train"] = np.zeros(num_of_iters)
        avg_result[kinect_test]["fc6_linear"]["test"] = np.zeros(num_of_iters)
        avg_result[kinect_test]["fc6_rbf"]["train"] = np.zeros(num_of_iters)
        avg_result[kinect_test]["fc6_rbf"]["test"] = np.zeros(num_of_iters)
        avg_result[kinect_test]["fc7_linear"]["train"] = np.zeros(num_of_iters)
        avg_result[kinect_test]["fc7_linear"]["test"] = np.zeros(num_of_iters)
        avg_result[kinect_test]["fc7_rbf"]["train"] = np.zeros(num_of_iters)
        avg_result[kinect_test]["fc7_rbf"]["test"] = np.zeros(num_of_iters)
        avg_result[kinect_test]["prob"]["train"] = np.zeros(num_of_iters)
        avg_result[kinect_test]["prob"]["test"] = np.zeros(num_of_iters)

    # Leave one out
    # subject_list = list_all_folders_in_a_directory(config_params.c3d_data_dir)

    subject_list = config_params.subject_list
    subject_test = config_params.subject_test

    subject_list = list(subject_list)
    num_of_subjects = len(subject_list)
    kinect_train = config_params.kinect_train
    snapshot = config_params.snapshot
    max_iter = config_params.max_iter

    params_info = get_params_info(config_params)

    for kinect_test in config_params.kinect_test_list:
        output_result_dir = os.path.join('result',"%s_%s_%s" % (kinect_train, kinect_test, config_params.date_time))
        create_folder(os.path.join(config_params.output_dir, output_result_dir))

        # CuongND. Save arguments.
        with open(os.path.join(config_params.output_dir, output_result_dir, "params.txt"), 'a') as the_file:
            the_file.writelines(params_info)
        print "Save parameters to file " + os.path.join(config_params.output_dir, output_result_dir, "params.txt")
        for subject in subject_list:
            continue_create_folder = False
            for test_subject in subject_test:
                if (test_subject == subject):
                    continue_create_folder = True
                    break
            if (continue_create_folder == False):
                continue
            create_folder(
                os.path.join(
                    config_params.output_dir,
                    output_result_dir,
                    subject))
            for iter_ in range(snapshot, max_iter + 1, snapshot):
                create_folder(
                    os.path.join(
                        config_params.output_dir,
                        output_result_dir,
                        subject,
                        "iter_%d" % (iter_)))

    # random.shuffle(subject_list)
    count = 1
    for test_subject in subject_list:
        train_list = []
        test_list = [test_subject]

        # CuongND
        continue_test = False
        for subject in subject_test:
            if (test_subject == subject):
                continue_test = True
                break
        if (continue_test == False):
            continue

        for train_subject in subject_list:
            if train_subject == test_subject:
                continue
            train_list.append(train_subject)


        print "\n\nTEST SUBJECT " + str(
            count) + ': ' + test_subject + ' ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
        print "BEGINS\n\n"
        count += 1
        # Train and Classify
        (result_list) = c3d_train_and_test(train_list, test_list, config_params)

        # Output to text files
        output_result_dir = os.path.join('result',"%s_%s_%s" % (kinect_train, kinect_test, config_params.date_time))
        print "\n\nWrite result to folder output/" + output_result_dir +'/'+test_subject +'\n\n'
        for kinect_test, r0_ in result_list.iteritems():
            for classification_method, r1_ in r0_.iteritems():
                # Plot loss
                file_name = "%s_loss.png" % (
                    classification_method)
                dump_plot_to_image_file(
                    r1_["train"].loss_list,
                    r1_["test"].loss_list,
                    config_params.max_iter,
                    config_params.snapshot,
                    "Train",
                    "Test",
                    "Loss",
                    os.path.join(
                        config_params.output_dir,
                        output_result_dir,
                        test_subject,
                        file_name))

                # Plot accuracy
                file_name = "%s_accuracy.png" % (
                    classification_method)
                dump_plot_to_image_file(
                    r1_["train"].acc_list,
                    r1_["test"].acc_list,
                    config_params.max_iter,
                    config_params.snapshot,
                    "Train",
                    "Test",
                    "Accuracy",
                    os.path.join(
                        config_params.output_dir,
                        output_result_dir,
                        test_subject,
                        file_name))

                # Save accuracy and loss in text files
                np.savetxt(
                    os.path.join(
                        config_params.output_dir,
                        output_result_dir,
                        test_subject,
                        "%s_train_accuracy.txt" % (classification_method)),
                    r1_["train"].acc_list,
                    delimiter=' ',
                    fmt="%f")
                np.savetxt(
                    os.path.join(
                        config_params.output_dir,
                        output_result_dir,
                        test_subject,
                        "%s_test_accuracy.txt" % (classification_method)),
                    r1_["test"].acc_list,
                    delimiter=' ',
                    fmt="%f")
                np.savetxt(
                    os.path.join(
                        config_params.output_dir,
                        output_result_dir,
                        test_subject,
                        "%s_train_loss.txt" % (classification_method)),
                    r1_["train"].loss_list,
                    delimiter=' ',
                    fmt="%f")
                np.savetxt(
                    os.path.join(
                        config_params.output_dir,
                        output_result_dir,
                        test_subject,
                        "%s_test_loss.txt" % (classification_method)),
                    r1_["test"].loss_list,
                    delimiter=' ',
                    fmt="%f")

                # Aggregate average results
                avg_result[kinect_test][classification_method]["train"] += np.array(r1_["train"].acc_list)
                avg_result[kinect_test][classification_method]["test"] += np.array(r1_["test"].acc_list)

    # Output average accuracy
    print ""
    print ""
    print "OUTPUT AVERAGE ACCURACY"
    print ""
    print ""
    # pdb.set_trace()
    for kinect_test, r0_ in avg_result.iteritems():
        output_result_dir = os.path.join('result',"%s_%s_%s" % (kinect_train, kinect_test, config_params.date_time))
        for classification_method, r1_ in r0_.iteritems():
            file_name = "%s_avg_acc.png" % (classification_method)
            # pdb.set_trace()
            dump_plot_to_image_file(
                list(r1_["train"] / (1.0 * num_of_subjects)),
                list(r1_["test"] / (1.0 * num_of_subjects)),
                config_params.max_iter,
                config_params.snapshot,
                "Train",
                "Test",
                "Average accuracy",
                os.path.join(
                    config_params.output_dir,
                    output_result_dir,
                    file_name))
            np.savetxt(
                os.path.join(
                    config_params.output_dir,
                    output_result_dir,
                    "%s_train_accuracy.txt" % (classification_method)),
                list(r1_["train"] / (1.0 * num_of_subjects)),
                delimiter=' ',
                fmt="%f")
            np.savetxt(
                os.path.join(
                    config_params.output_dir,
                    output_result_dir,
                    "%s_test_accuracy.txt" % (classification_method)),
                list(r1_["test"] / (1.0 * num_of_subjects)),
                delimiter=' ',
                fmt="%f")

        summary_result(os.path.join(c3d_params.output_dir, output_result_dir), subjects=c3d_params.subject_list)