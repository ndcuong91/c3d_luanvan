import os
import time
from sklearn.externals import joblib
from glob import glob
from python_utils import list_all_folders_in_a_directory
import subprocess
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('agg')
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import log_loss, confusion_matrix
from c3d_helper import find_files_to_read, load_data_for_classification, check_gpu_ready

import c3d_params


class C3DNetwork(object):
    """
    Written by Dang Manh Truong (dangmanhtruong@gmail.com)

    This is a wrapper class to the C3D network.

    Supported functions:

    - Finetuning using a pretrained model

    - Feature extraction

    TODO:

    - Train

    - Test

    - Visualization using deconvnet3D

    Also note that this class uses only absolute paths
    """

    def __print_params_to_file__(self, input_params, template_file, output_file):
        """
        Utility function

        input_params: A tuple, the number of which must match with that of template_file

        template_file: Absoulte path of the template file, which is a formatted text file (e.g: "Hello world year %d" )

        output_file: Absolute path of the output
        """
        with open(output_file, "wt") as f, open(template_file, "rt") as f_read:
            formatted_content = f_read.read()
            print >> f, formatted_content % input_params

    def __init__(self):
        pass

    def set_template_compute_volume_mean(self, template_compute_volume_mean_sh_file):
        self.template_compute_volume_mean_sh_file = template_compute_volume_mean_sh_file
        pass

    def set_template_train(self, \
                           template_train_solver_file, \
                           template_train_prototxt_file, \
                           template_train_sh_file):
        self.template_train_solver_file = template_train_solver_file
        self.template_train_prototxt_file = template_train_prototxt_file
        self.template_train_sh_file = template_train_sh_file

    def set_template_test(self, \
                          template_test_prototxt_file, \
                          template_test_sh_file):
        self.template_test_prototxt_file = template_test_prototxt_file
        self.template_test_sh_file = template_test_sh_file

    def set_template_finetuning(self, \
                                template_finetuning_solver_file, \
                                template_finetuning_prototxt_file, \
                                template_finetuning_sh_file):
        self.template_finetuning_solver_file = template_finetuning_solver_file
        self.template_finetuning_prototxt_file = template_finetuning_prototxt_file
        self.template_finetuning_sh_file = template_finetuning_sh_file

    def set_template_feature_extraction(self, \
                                        template_feature_extraction_prototxt_file, \
                                        template_feature_extraction_sh_file):
        self.template_feature_extraction_prototxt_file = template_feature_extraction_prototxt_file
        self.template_feature_extraction_sh_file = template_feature_extraction_sh_file

    def set_template_visualization(self):
        """
        TODO
        """
        pass

    def compute_volume_mean(self,
                            tool_dir,
                            input_file,
                            output_volume_mean_file,
                            num_frame,
                            resize_h,
                            resize_w,
                            compute_volume_mean_sh_file):
        """
        input_file is assumed to be in the format:

        <string_path> <starting_frame> <label> (according to C3D user guide)
        """

        # Sh file
        input_params = ( os.path.join(tool_dir,'compute_volume_mean_from_list.bin') , input_file, num_frame, resize_h, resize_w, output_volume_mean_file)
        self.__print_params_to_file__(input_params, self.template_compute_volume_mean_sh_file, \
                                      compute_volume_mean_sh_file)

        # CuongND. Check for GPU
        check_gpu_ready(allocate_mem=500)
        # Run
        print subprocess.check_output(['sh', compute_volume_mean_sh_file])

    def train(self,
              input_file,
              volume_mean_file,
              train_solver_file,
              train_prototxt_file,
              train_sh_file,
              snapshot_prefix,
              num_of_iters_until_stop,
              num_of_iters_until_periodic_snapshot,
              num_of_classes):
        """
        TODO
        """
        # Solver file
        input_params = (train_prototxt_file, num_of_iters_until_stop,
                        num_of_iters_until_periodic_snapshot, snapshot_prefix)
        self.__print_params_to_file__(input_params, self.template_train_solver_file, train_solver_file)

        # Prototxt file
        input_params = (input_file, volume_mean_file, num_of_classes)
        self.__print_params_to_file__(input_params, self.template_train_prototxt_file, train_prototxt_file)

        # Sh file
        input_params = (train_solver_file)
        self.__print_params_to_file__(input_params, self.template_train_sh_file, train_sh_file)

        # Run
        print subprocess.check_output(['sh', train_sh_file])

    def test(self):
        """
        TODO
        """
        pass

    def finetune(self,
                 tool_dir,
                 input_file,
                 volume_mean_file,
                 pretrained_model_file,
                 finetuning_solver_file,
                 finetuning_sh_file,
                 finetuning_prototxt_file,
                 batch_size_finetune,
                 base_lr,
                 gamma,
                 step_size,
                 max_iter,
                 snapshot,
                 snapshot_prefix,
                 crop,
                 resize_h,
                 resize_w,
                 num_frame,
                 conv1a, conv2a, conv3a, conv3b, conv4a, conv4b, conv5a, conv5b, fc6, fc7,
                 num_of_classes):
        """
        This method performs finetuning for a number of iterations then stop and save the model

        Useful when you need to plot the accuracy,... as a function of iterations

        Note that all file directories must be absolute paths

        --------

        input_file is assumed to be in the format:

        <string_path> <starting_frame> <label> (according to C3D user guide)
        """

        # Solver file
        input_params = (finetuning_prototxt_file, base_lr, gamma, step_size, max_iter, \
                        snapshot, snapshot_prefix)
        self.__print_params_to_file__(input_params, self.template_finetuning_solver_file, finetuning_solver_file)

        # Prototxt file
        input_params = (input_file, volume_mean_file, batch_size_finetune, crop, resize_h, resize_w, num_frame,
                        conv1a, conv2a, conv3a, conv3b, conv4a, conv4b, conv5a, conv5b, fc6, fc7,
                        num_of_classes)
        self.__print_params_to_file__(input_params, self.template_finetuning_prototxt_file, finetuning_prototxt_file)

        # Sh file
        input_params = (os.path.join(tool_dir,'finetune_net.bin'), finetuning_solver_file, pretrained_model_file)
        self.__print_params_to_file__(input_params, self.template_finetuning_sh_file, finetuning_sh_file)

        # CuongND. Check for GPU
        check_gpu_ready(allocate_mem=7000)
        # Run
        print subprocess.check_output(['sh', finetuning_sh_file])

    def feature_extraction(self,
                           tool_dir,
                           input_file,
                           output_file,
                           volume_mean_file,
                           pretrained_model_file,
                           feature_extraction_prototxt_file,
                           feature_extraction_sh_file,
                           batch_size,
                           num_of_batches,
                           crop,
                           resize_h,
                           resize_w,
                           num_frame,
                           conv1a, conv2a, conv3a, conv3b, conv4a, conv4b, conv5a, conv5b, fc6, fc7,
                           num_of_classes):
        """
        Performs feature extraction

        input_file is assumed to be in the format:

        <string_path> <starting_frame> <label> (according to C3D user guide)
        """
        # Prototxt file
        input_params = (input_file, volume_mean_file, batch_size, crop, resize_h, resize_w, num_frame,
                        conv1a, conv2a, conv3a, conv3b, conv4a, conv4b, conv5a, conv5b, fc6, fc7,
                        num_of_classes)
        self.__print_params_to_file__(input_params, self.template_feature_extraction_prototxt_file, \
                                      feature_extraction_prototxt_file)

        # Sh file
        input_params = (os.path.join(tool_dir,'extract_image_features.bin'), feature_extraction_prototxt_file, pretrained_model_file, batch_size, num_of_batches, \
                        output_file)
        self.__print_params_to_file__(input_params, self.template_feature_extraction_sh_file,
                                      feature_extraction_sh_file)

        # CuongND. Check for GPU
        check_gpu_ready(allocate_mem=3500)
        # Run
        print subprocess.check_output(['sh', feature_extraction_sh_file])

    def visualization(self):
        """
        TODO
        """
        pass


class Result(object):
    """
    Placeholder object for the results

    acc: Accuracy

    confmat: Confusion matrix

    loss: Loss (here we use multinomial logistic loss)

    misclassified_dict: Dictionary of the form

    """

    def __init__(self):
        self.acc = 0
        self.confmat = 0
        self.loss = 0
        self.misclassified_dict = 0

    def set_value(self, acc, confmat, loss, misclassified_dict):
        self.acc = acc
        self.confmat = confmat
        self.loss = loss
        self.misclassified_dict = misclassified_dict


class ResultList(object):
    """
    Placeholder object for the list of results

    acc_list: List of accuracies

    loss_list: List of losses (here we use multinomial logistic loss)

    """

    def __init__(self):
        self.acc_list = []
        self.loss_list = []

    def add_element(self, result):
        self.acc_list.append(result.acc)
        self.loss_list.append(result.loss)


class KinectRunner(object):
    def __init__(self):
        self.test_numlines = 0
        self.num_of_test_batches = 0
        self.c3d_test_data_dir = 0


def create_lst_files(config_params, c3d_files_dir, data_dir, subject_list, name, num_of_actions, image_type):
    in_name = name + ".lst"
    out_name = name + "_output.lst"
    in_fullpath = os.path.join(c3d_files_dir, in_name)
    out_fullpath = os.path.join(c3d_files_dir, out_name)

    num_of_lines = 0

    with open(in_fullpath, "wt") as f_in, open(out_fullpath, "wt") as f_out:
        for subject_name in subject_list:
            fullpath_subject = os.path.join(data_dir, subject_name)
            new_output_subject = os.path.join(config_params.output_dir, config_params.output_vector,
                                              subject_name)  # CuongND. New folder for output
            for action_id in range(num_of_actions):
                action_id = action_id + 1  # 0,1,2,... -> 1,2,3....
                fullpath_subject_action = os.path.join(fullpath_subject, str(action_id))
                new_output_dir_action = os.path.join(new_output_subject,
                                                     str(action_id))  # CuongND. New folder for output
                epoch_list = list_all_folders_in_a_directory(fullpath_subject_action)
                for epoch_id in epoch_list:
                    fullpath_subject_action_epoch = os.path.join(fullpath_subject_action, epoch_id)
                    # CuongND. New folder for output
                    new_output_dir_action_epoch = os.path.join(new_output_dir_action, epoch_id)
                    if not os.path.exists(new_output_dir_action_epoch):
                        os.makedirs(new_output_dir_action_epoch)
                    # CuongND.end

                    # Find number of images
                    image_list = list(glob(os.path.join(fullpath_subject_action_epoch, "*." + image_type)))
                    num_of_images = len(image_list)
                    num_of_batches_of_16_images = num_of_images / 16

                    # Write (note that already padded to 16n !)
                    # counter = 1
                    # for _ in range(num_of_batches_of_16_images):
                    #     # Input
                    #     in_text = "%s/ %d %d\n" % (fullpath_subject_action_epoch, counter, action_id - 1)
                    #     f_in.write(in_text)
                    #     # Output
                    #     out_text = "%s/%06d\n" % (fullpath_subject_action_epoch, counter)
                    #     f_out.write(out_text)

                    #     num_of_lines = num_of_lines + 1
                    #     counter = counter + 16

                    # num_of_batches_of_16_images = num_of_images / 16

                    # Assumption: if smaller than 16, then padded to 16
                    # But if bigger than 16 then no padding .
                    # Example: 23 images, then it would be 1-16 and 8-23
                    counter = 1
                    for _ in range(num_of_batches_of_16_images):
                        # Input
                        in_text = "%s/ %d %d\n" % (
                        fullpath_subject_action_epoch, counter, action_id - 1)  # CuongND. New folder for input
                        f_in.write(in_text)
                        # Output
                        # out_text = "%s/%06d\n" % (fullpath_subject_action_epoch, counter)
                        out_text = "%s/%06d\n" % (
                        new_output_dir_action_epoch, counter)  # CuongND. New folder for output
                        f_out.write(out_text)

                        num_of_lines = num_of_lines + 1
                        counter = counter + 16

                        # Last batch: last 16 images

                    if (num_of_batches_of_16_images * 16) < num_of_images:
                        counter = num_of_images - 15
                        # Input
                        in_text = "%s/ %d %d\n" % (
                        fullpath_subject_action_epoch, counter, action_id - 1)  # CuongND. New folder for input
                        f_in.write(in_text)
                        # Output
                        # out_text = "%s/%06d\n" % (fullpath_subject_action_epoch, counter)
                        out_text = "%s/%06d\n" % (
                        new_output_dir_action_epoch, counter)  # CuongND. New folder for output
                        f_out.write(out_text)

                        num_of_lines = num_of_lines + 1
                        counter = counter + 16
                    pass
                pass

            pass
        pass
    return num_of_lines


def classification_routine(X_train, Y_train, X_test, Y_test, feature_type, svm_type,
                           train_list_mappings_dir_to_label, test_list_mappings_dir_to_label):
    """
    Input:

    train_list_mappings_dir_to_label: Dictionary of the form (train_dir, label)

    where train_dir_original is a lists

    test_list_mappings_dir_to_label: The same as train_list_mappings_dir_to_label
    """
    svm_train_time = 0
    svm_test_time = 0

    if feature_type != "prob":
        scaler = preprocessing.MinMaxScaler()

        train_start = time.time()

        X_train = scaler.fit_transform(X_train)

        train_elapsed = time.time() - train_start
        svm_train_time += train_elapsed

        test_start = time.time()

        X_test = scaler.transform(X_test)

        test_elapsed = time.time() - test_start
        svm_test_time += test_elapsed

        # Train and test
        train_start = time.time()
        classifier = OneVsRestClassifier(SVC(kernel=svm_type, probability=True))
        classifier.fit(X_train, Y_train)
        if (feature_type == "fc6") and (svm_type == "linear"):
            joblib.dump(classifier, 'c3d_svm_trained_model.pkl')
            joblib.dump(scaler, 'c3d_svm_scaler.pkl')
        train_elapsed = time.time() - train_start
        svm_train_time += train_elapsed

        train_start = time.time()
        prob_train = classifier.predict_proba(X_train)
        argmax_train = np.argmax(prob_train, axis=1)
        prediction_train = classifier.classes_[argmax_train]
        Y_train = Y_train.flatten()
        confmat_train = confusion_matrix(Y_train, prediction_train)
        loss_train = log_loss(Y_train, prob_train[:, classifier.classes_])
        train_elapsed = time.time() - train_start
        svm_train_time += train_elapsed

        test_start = time.time()
        prob_test = classifier.predict_proba(X_test)
        argmax_test = np.argmax(prob_test, axis=1)
        prediction_test = classifier.classes_[argmax_test]
        Y_test = Y_test.flatten()
        confmat_test = confusion_matrix(Y_test, prediction_test)
        loss_test = log_loss(Y_test, prob_test[:, classifier.classes_])
        test_elapsed = time.time() - test_start
        svm_test_time += test_elapsed
    else:
        # Use prob layer to predict directly
        prob_train = X_train
        prob_test = X_test
        prediction_train = np.argmax(prob_train, axis=1)
        prediction_test = np.argmax(prob_test, axis=1)

        Y_train = Y_train.flatten()
        Y_test = Y_test.flatten()

        confmat_train = confusion_matrix(Y_train, prediction_train)
        confmat_test = confusion_matrix(Y_test, prediction_test)
        loss_train = log_loss(Y_train, prob_train)
        loss_test = log_loss(Y_test, prob_test)

    num_of_train_corrects = np.sum(prediction_train == Y_train)
    num_of_test_corrects = np.sum(prediction_test == Y_test)
    acc_train = num_of_train_corrects / (Y_train.size * 1.0)
    acc_test = num_of_test_corrects / (Y_test.size * 1.0)

    dict_train_misclassified = {}
    for i in range(Y_train.size):
        if prediction_train[i] != Y_train[i]:
            dict_train_misclassified[train_list_mappings_dir_to_label[i]] = (Y_train[i], prediction_train[i])
    dict_test_misclassified = {}
    for i in range(Y_test.size):
        if prediction_test[i] != Y_test[i]:
            dict_test_misclassified[test_list_mappings_dir_to_label[i]] = (Y_test[i], prediction_test[i])
    return (acc_train, acc_test, loss_train, loss_test, confmat_train, confmat_test,
            dict_train_misclassified, dict_test_misclassified, svm_train_time, svm_test_time)


# ----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X, y, output_dir, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.savefig(output_dir)
    plt.close(fig)


def dump_plot_to_image_file(train_list, test_list, max_iter, snapshot, train_label, test_label, plot_title, out_dir):
    t = np.arange(snapshot, max_iter + 1, snapshot)
    fig = plt.figure()
    plt.plot(t, train_list, color='b', label=train_label)
    plt.plot(t, test_list, color='orange', label=test_label)
    plt.legend(loc='upper right')
    fig.suptitle(plot_title, fontsize=20)
    plt.savefig(out_dir)
    plt.close(fig)


def dump_list_of_misclassified_examples(dict_misclassified_examples, output_file_dir):
    with open(output_file_dir, "wt") as f:
        for example_dir, (ground_truth, prediction_result) in dict_misclassified_examples.iteritems():
            # 0,1,2,... -> 1,2.3,...
            f.write("%s %s %s\n" % (example_dir, ground_truth + 1, prediction_result + 1))


def c3d_train_and_test(train_list, test_list, config_params):
    test_subject = test_list[0]
    result_list = {}
    kinect_run_list = {}
    for kinect_test in config_params.kinect_test_list:
        result_list[kinect_test] = {}
        result_list[kinect_test]["fc6_linear"] = {}
        result_list[kinect_test]["fc6_rbf"] = {}
        result_list[kinect_test]["fc7_linear"] = {}
        result_list[kinect_test]["fc7_rbf"] = {}
        result_list[kinect_test]["prob"] = {}
        result_list[kinect_test]["fc6_linear"]["train"] = ResultList()
        result_list[kinect_test]["fc6_linear"]["test"] = ResultList()
        result_list[kinect_test]["fc6_rbf"]["train"] = ResultList()
        result_list[kinect_test]["fc6_rbf"]["test"] = ResultList()
        result_list[kinect_test]["fc7_linear"]["train"] = ResultList()
        result_list[kinect_test]["fc7_linear"]["test"] = ResultList()
        result_list[kinect_test]["fc7_rbf"]["train"] = ResultList()
        result_list[kinect_test]["fc7_rbf"]["test"] = ResultList()
        result_list[kinect_test]["prob"]["train"] = ResultList()
        result_list[kinect_test]["prob"]["test"] = ResultList()
        kinect_run_list[kinect_test] = KinectRunner()

    num_of_actions = config_params.num_of_actions
    max_iter = config_params.max_iter
    snapshot = config_params.snapshot
    image_type = config_params.image_type
    batch_size = config_params.batch_size
    c3d_files_dir = config_params.c3d_files_dir

    kinect_train = config_params.kinect_train
    kinect_test_list = config_params.kinect_test_list
    c3d_data_root = config_params.c3d_data_root
    c3d_feature_extraction_for_test_dir = config_params.c3d_feature_extraction_for_test_dir
    output_dir = config_params.output_dir

    # Create train_01.lst5, train_01_output.lst, test_01.lst, test_01_output.lst
    c3d_train_data_dir = os.path.join(c3d_data_root, kinect_train + "_" + config_params.data_type_train)
    train_numlines = create_lst_files(config_params, c3d_files_dir, c3d_train_data_dir, train_list, "train_01",
                                      num_of_actions, image_type)
    num_of_train_batches = train_numlines / batch_size
    if (num_of_train_batches * batch_size < train_numlines):
        num_of_train_batches = num_of_train_batches + 1

    for kinect_test in kinect_test_list:
        kinect_run_list[kinect_test].c3d_test_data_dir = os.path.join(c3d_data_root,
                                                                      kinect_test + "_" + config_params.data_type_test)
        kinect_run_list[kinect_test].test_numlines = create_lst_files(config_params,
                                                                      c3d_files_dir,
                                                                      kinect_run_list[kinect_test].c3d_test_data_dir,
                                                                      test_list,
                                                                      "test_01_%s" % (kinect_test),
                                                                      num_of_actions,
                                                                      image_type)
        # pdb.set_trace()
        kinect_run_list[kinect_test].num_of_test_batches = kinect_run_list[kinect_test].test_numlines / batch_size
        if (kinect_run_list[kinect_test].num_of_test_batches * batch_size < kinect_run_list[kinect_test].test_numlines):
            kinect_run_list[kinect_test].num_of_test_batches += 1

    # Initialize networks
    original_network = C3DNetwork()
    original_network.set_template_compute_volume_mean(config_params.c3d_template_compute_volume_mean_sh)
    original_network.set_template_finetuning(
        config_params.c3d_template_finetuning_solver,
        config_params.c3d_template_finetuning_train,
        config_params.c3d_template_finetuning_sh
    )
    original_network.set_template_feature_extraction(
        config_params.c3d_template_feature_extractor_frm,
        config_params.c3d_template_feature_extractor_sh
    )

    # Compute volume mean
    if (c3d_params.compute_volume_mean == True):
        original_network.compute_volume_mean(config_params.tool_dir,
                                             config_params.c3d_train_01,
                                             config_params.c3d_volume_mean_file,
                                             config_params.num_frame,
                                             config_params.resize_h,
                                             config_params.resize_w,
                                             config_params.c3d_compute_volume_mean_sh)

    # Finetune on train_01.lst. Output (e.g): c3d_sport1m_finetune_whole_iter_20000
    # Periodic finetuning on train_01.lst: After each snapshot iterations, save model to file,
    # then extract features from train and test files (including prob layer)
    # Then predict train and test and output loss and confusion matrix using features (fc6, fc7, and prob)
    pretrained_model_fulldir = os.path.join(config_params.c3d_pretrained_model_and_volume_mean_dir,
                                            "conv3d_deepnetA_sport1m_iter_1900000")
    # This file is created after each snapshot iterations
    # intermediate_model_fulldir = config_params.c3d_snapshot_prefix + "_iter_" + str(snapshot)
    # temp = intermediate_model_fulldir + "_temp"
    input_model_fulldir = pretrained_model_fulldir
    num_of_iters = max_iter / snapshot

    c3d_pretrained_model = config_params.c3d_pretrained_model
    if (c3d_params.finetuning == True):
        start_train = time.time()
        # Perform finetuning on train set
        print "CuongND. Finetune on train set. Move outside loop to increase performance"
        original_network.finetune(
            config_params.tool_dir,
            config_params.c3d_train_01,
            config_params.c3d_volume_mean_file,
            c3d_pretrained_model,
            config_params.c3d_finetuning_solver,
            config_params.c3d_finetuning_sh,
            config_params.c3d_finetuning_train,
            config_params.batch_size_finetune,
            config_params.base_lr,
            config_params.gamma,
            config_params.step_size,
            config_params.max_iter,
            config_params.snapshot,
            config_params.snapshot_prefix,
            config_params.crop,
            config_params.resize_h,
            config_params.resize_w,
            config_params.num_frame,
            config_params.conv1a,
            config_params.conv2a,
            config_params.conv3a,
            config_params.conv3b,
            config_params.conv4a,
            config_params.conv4b,
            config_params.conv5a,
            config_params.conv5b,
            config_params.fc6,
            config_params.fc7,
            num_of_actions)
        elapsed_train = time.time() - start_train
        print "CuongND. Finetuning time: %d" % (elapsed_train)

    start_test = time.time()
    for iter_ in range(snapshot, max_iter + 1, snapshot):
        result = {}
        for kinect_test in config_params.kinect_test_list:
            result[kinect_test] = {}
            result[kinect_test]["fc6_linear"] = {}
            result[kinect_test]["fc6_rbf"] = {}
            result[kinect_test]["fc7_linear"] = {}
            result[kinect_test]["fc7_rbf"] = {}
            result[kinect_test]["prob"] = {}
            result[kinect_test]["fc6_linear"]["train"] = Result()
            result[kinect_test]["fc6_linear"]["test"] = Result()
            result[kinect_test]["fc6_rbf"]["train"] = Result()
            result[kinect_test]["fc6_rbf"]["test"] = Result()
            result[kinect_test]["fc7_linear"]["train"] = Result()
            result[kinect_test]["fc7_linear"]["test"] = Result()
            result[kinect_test]["fc7_rbf"]["train"] = Result()
            result[kinect_test]["fc7_rbf"]["test"] = Result()
            result[kinect_test]["prob"]["train"] = Result()
            result[kinect_test]["prob"]["test"] = Result()

        # os.makedirs(os.path.join(config_params.output_dir, "iter_%d" % (iter_)))
        # output_iter_dir = os.path.join(config_params.output_dir, "iter_%d" % (iter_))

        # subprocess.check_output(['python','/home/dangmanhtruong95/Truong_Python_run_scripts/Useful_code/delete_all_files_except_images.py',
        #     c3d_train_data_dir])
        # for kinect_test in kinect_test_list:
        # subprocess.check_output(['python','/home/dangmanhtruong95/Truong_Python_run_scripts/Useful_code/delete_all_files_except_images.py',
        # kinect_run_list[kinect_test].c3d_test_data_dir])
        c3d_pretrained_model = config_params.snapshot_prefix + "_iter_" + str(iter_)

        if (c3d_params.feature_extract  == True):
            print "EXTRACT FEATURES ON TRAIN SET"
            print "Model %s" % (c3d_pretrained_model)
            original_network.feature_extraction(
                config_params.tool_dir,
                config_params.c3d_train_01,
                config_params.c3d_train_01_output,
                config_params.c3d_volume_mean_file,
                c3d_pretrained_model,
                config_params.c3d_feature_extraction_train_prototxt,
                config_params.c3d_feature_extraction_train_sh,
                batch_size,
                num_of_train_batches,
                config_params.crop,
                config_params.resize_h,
                config_params.resize_w,
                config_params.num_frame,
                config_params.conv1a,
                config_params.conv2a,
                config_params.conv3a,
                config_params.conv3b,
                config_params.conv4a,
                config_params.conv4b,
                config_params.conv5a,
                config_params.conv5b,
                config_params.fc6,
                config_params.fc7,
                num_of_actions,
            )

        for kinect_test in kinect_test_list:
            if (c3d_params.feature_extract == True):
                print "EXTRACT FEATURE ON TEST SET (view: %s)" % (kinect_test)
                print "Model %s" % (c3d_pretrained_model)
                original_network.feature_extraction(
                    config_params.tool_dir,
                    os.path.join(c3d_files_dir, "test_01_%s.lst" % (kinect_test)),
                    os.path.join(c3d_files_dir, "test_01_%s_output.lst" % (kinect_test)),
                    config_params.c3d_volume_mean_file,
                    c3d_pretrained_model,
                    os.path.join(os.path.join(c3d_feature_extraction_for_test_dir,
                                              "c3d_sport1m_feature_extractor_frm_test_%s.prototxt" % (kinect_test))),
                    os.path.join(os.path.join(c3d_feature_extraction_for_test_dir,
                                              "c3d_sport1m_feature_extractor_test_%s.sh" % (kinect_test))),
                    batch_size,
                    kinect_run_list[kinect_test].num_of_test_batches,
                    config_params.crop,
                    config_params.resize_h,
                    config_params.resize_w,
                    config_params.num_frame,
                    config_params.conv1a,
                    config_params.conv2a,
                    config_params.conv3a,
                    config_params.conv3b,
                    config_params.conv4a,
                    config_params.conv4b,
                    config_params.conv5a,
                    config_params.conv5b,
                    config_params.fc6,
                    config_params.fc7,
                    num_of_actions)

            print "CLASSIFICATION!"
            # Use the extracted features (fc6, fc7, prob) to calculate loss, confusion matrix and accuracy
            train_01_fulldir = os.path.join(c3d_files_dir, "train_01.lst")
            test_01_fulldir = os.path.join(c3d_files_dir, "test_01_%s.lst" % (kinect_test))
            find_files_to_read_func = find_files_to_read

            # pdb.set_trace()

            (X_train, Y_train, X_test, Y_test, train_mapped_to_dir, test_mapped_to_dir) = \
                load_data_for_classification(
                    config_params.fc6,
                    os.path.join(config_params.output_dir, config_params.output_vector),
                    train_01_fulldir,
                    test_01_fulldir,
                    "fc6",
                    num_of_actions,
                    find_files_to_read_func,
                    average_feature=config_params.average_feature)

            # pdb.set_trace()

            (acc_train, acc_test, loss_train, loss_test, confmat_train, confmat_test, \
             dict_train_instances_misclassified, dict_test_instances_misclassified, svm_train_time, svm_test_time) = \
                classification_routine(
                    X_train,
                    Y_train,
                    X_test,
                    Y_test,
                    "fc6",
                    "linear",
                    train_mapped_to_dir,
                    test_mapped_to_dir)

            result[kinect_test]["fc6_linear"]["train"].set_value(
                acc_train,
                confmat_train,
                loss_train,
                dict_train_instances_misclassified)
            result[kinect_test]["fc6_linear"]["test"].set_value(
                acc_test,
                confmat_test,
                loss_test,
                dict_test_instances_misclassified)
            (acc_train, acc_test, loss_train, loss_test, confmat_train, confmat_test, \
             dict_train_instances_misclassified, dict_test_instances_misclassified, _, _) = \
                classification_routine(
                    X_train,
                    Y_train,
                    X_test,
                    Y_test,
                    "fc6",
                    "rbf",
                    train_mapped_to_dir,
                    test_mapped_to_dir)
            result[kinect_test]["fc6_rbf"]["train"].set_value(
                acc_train,
                confmat_train,
                loss_train,
                dict_train_instances_misclassified)
            result[kinect_test]["fc6_rbf"]["test"].set_value(
                acc_test,
                confmat_test,
                loss_test,
                dict_test_instances_misclassified)

            (X_train, Y_train, X_test, Y_test, train_mapped_to_dir, test_mapped_to_dir) = \
                load_data_for_classification(
                    config_params.fc7,
                    os.path.join(config_params.output_dir, config_params.output_vector),
                    train_01_fulldir,
                    test_01_fulldir,
                    "fc7",
                    num_of_actions,
                    find_files_to_read_func,
                    average_feature=config_params.average_feature)
            (acc_train, acc_test, loss_train, loss_test, confmat_train, confmat_test, \
             dict_train_instances_misclassified, dict_test_instances_misclassified, _, _) = \
                classification_routine(
                    X_train,
                    Y_train,
                    X_test,
                    Y_test,
                    "fc7",
                    "linear",
                    train_mapped_to_dir,
                    test_mapped_to_dir)
            # pdb.set_trace()
            result[kinect_test]["fc7_linear"]["train"].set_value(
                acc_train,
                confmat_train,
                loss_train,
                dict_train_instances_misclassified)
            result[kinect_test]["fc7_linear"]["test"].set_value(
                acc_test,
                confmat_test,
                loss_test,
                dict_test_instances_misclassified)
            (acc_train, acc_test, loss_train, loss_test, confmat_train, confmat_test, \
             dict_train_instances_misclassified, dict_test_instances_misclassified, _, _) = \
                classification_routine(
                    X_train,
                    Y_train,
                    X_test,
                    Y_test,
                    "fc7",
                    "rbf",
                    train_mapped_to_dir,
                    test_mapped_to_dir)
            result[kinect_test]["fc7_rbf"]["train"].set_value(
                acc_train,
                confmat_train,
                loss_train,
                dict_train_instances_misclassified)
            result[kinect_test]["fc7_rbf"]["test"].set_value(
                acc_test,
                confmat_test,
                loss_test,
                dict_test_instances_misclassified)

            (X_train, Y_train, X_test, Y_test, train_mapped_to_dir, test_mapped_to_dir) = \
                load_data_for_classification(
                    config_params.fc7,
                    os.path.join(config_params.output_dir, config_params.output_vector),
                    train_01_fulldir,
                    test_01_fulldir,
                    "prob",
                    num_of_actions,
                    find_files_to_read_func)
            (acc_train, acc_test, loss_train, loss_test, confmat_train, confmat_test, \
             dict_train_instances_misclassified, dict_test_instances_misclassified, _, _) = \
                classification_routine(
                    X_train,
                    Y_train,
                    X_test,
                    Y_test,
                    "prob",
                    "",
                    train_mapped_to_dir,
                    test_mapped_to_dir)
            result[kinect_test]["prob"]["train"].set_value(
                acc_train,
                confmat_train,
                loss_train,
                dict_train_instances_misclassified)
            result[kinect_test]["prob"]["test"].set_value(
                acc_test,
                confmat_test,
                loss_test,
                dict_test_instances_misclassified)

            print "Print confusion matrix and list of misclassified examples"
            # Print confusion matrix and list of misclassified examples for each snapshot iterations
            # for kinect_test, r0_ in result.iteritems():
            for classification_method, r1_ in result[kinect_test].iteritems():
                for train_or_test, r2_ in r1_.iteritems():
                    # Output confusion matrix
                    file_name = "%s_%s_confmat_iter_%d.txt" % (
                        classification_method,
                        train_or_test,
                        iter_)
                    # pdb.set_trace()
                    np.savetxt(
                        # os.path.join(output_iter_dir,file_name),
                        os.path.join(
                            output_dir,
                            "%s_%s_%s" % (kinect_train, kinect_test, config_params.date_time),
                            test_subject,
                            "iter_%d" % (iter_),
                            file_name),
                        r2_.confmat,
                        delimiter=' ',
                        fmt="%6d")

                    # Output misclassified examples
                    file_name = "%s_%s_misclassified_examples_iter_%d.txt" % (
                        classification_method,
                        train_or_test,
                        iter_)
                    dump_list_of_misclassified_examples(
                        r2_.misclassified_dict,
                        os.path.join(
                            output_dir,
                            "%s_%s_%s" % (kinect_train, kinect_test, config_params.date_time),
                            test_subject,
                            "iter_%d" % (iter_),
                            file_name))
                    # for kinect_test, r0_ in result.iteritems():
            for classification_method, r1_ in result[kinect_test].iteritems():
                # print classification_method
                # pdb.set_trace()
                for train_or_test, r2_ in r1_.iteritems():
                    # pdb.set_trace()
                    result_list[kinect_test][classification_method][train_or_test].add_element(
                        result[kinect_test][classification_method][train_or_test])
    print "\n\nTOTAL TEST TIME: %f" % (time.time() - start_test)
    # raw_input("Press enter")
    return result_list

