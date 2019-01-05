from crossview import c3d_train_and_test, Result
import crossview
import os, shutil
from python_utils import list_all_folders_in_a_directory, create_folder
import numpy as np
import random
#import matplotlib as mpl
#mpl.use('Agg') # To use on server
#import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from datetime import datetime
from c3d_helper import delete_files_with_extension_in_folder
import pdb

def dump_plot_to_image_file(train_list, test_list, max_iter, snapshot, train_label, test_label, plot_title, out_dir):
    t = np.arange(snapshot, max_iter+1, snapshot)
    fig = plt.figure()    
    plt.plot(t, train_list,color =  'b',label = train_label)
    plt.plot(t, test_list, color = 'orange', label = test_label)
    plt.legend(loc='upper right')
    fig.suptitle(plot_title, fontsize=20)
    plt.savefig(out_dir)
    plt.close(fig)     

class ConfigParams(object):
    """
    Written by Dang Manh Truong (dangmanhtruong@gmail.com), modified by CuongND (nguyenduycuong2004@gmail.com)
    """
    num_of_actions = 5 
    image_type = "jpg"
    base_lr= 0.0001
    gamma= 0.1
    step_size= 500
    max_iter = 800
    snapshot = 100
    batch_size = 20 # For feature extraction
    server=False
    crossview.server=server

    # Cuong thay doi rieng thu muc ouput va template
    output_dir = "/home/titikid/PycharmProjects/c3d_luanvan/output"
    template_dir = "/home/titikid/PycharmProjects/c3d_luanvan/template"
    c3d_data_root = "/home/titikid/PycharmProjects/c3d_luanvan"
    c3d_files_dir = "/home/titikid/PycharmProjects/c3d_luanvan/c3d_files"
    if(server==True):
        #Cuong thay doi rieng thu muc ouput va template
        output_dir = "/home/dangmanhtruong95/Cuong/c3d_luanvan/output"
        template_dir = "/home/dangmanhtruong95/Cuong/c3d_luanvan/template"
        c3d_data_root = "/media/data2/users/dangmanhtruong95"
        c3d_files_dir = "/home/dangmanhtruong95/Cuong/c3d_luanvan/c3d_files"

    #thay doi Kinect can train hoac test
    kinect_train = "Kinect_1"
    #kinect_test_list = ["Kinect_1","Kinect_3", "Kinect_5"]
    kinect_test_list = ["Kinect_1"]
    #data_type = "segmented" 
    data_type = "original"
    output_result_ext='output'
    date_time=datetime.now().strftime('%d-%m-%Y_%H.%M.%S') #date time when start training

    # c3d_feature_dir = "/home/dangmanhtruong95/Truong_Python_run_scripts/C3D_twostream_finetuning_with_confusion_matrix_and_loss_for_train_and_test/C3D_feature_dir"
    c3d_template_dir = os.path.join(template_dir, "Original")
    
    c3d_pretrained_model_and_volume_mean_dir = os.path.join(c3d_files_dir, "pretrained_model_and_volume_mean")
    c3d_finetuning_dir = os.path.join(c3d_files_dir, "finetuning")
    c3d_feature_extraction_for_train_dir = os.path.join(c3d_files_dir, "feature_extraction_for_train")
    c3d_feature_extraction_for_test_dir = os.path.join(c3d_files_dir, "feature_extraction_for_test")
    c3d_intermediate_model_snapshot_dir = os.path.join(c3d_files_dir, "intermediate_model_snapshot")


    c3d_compute_volume_mean_sh = os.path.join(c3d_pretrained_model_and_volume_mean_dir, "c3d_sport1m_compute_volume_mean.sh")
    c3d_pretrained_model = os.path.join(c3d_pretrained_model_and_volume_mean_dir, "conv3d_deepnetA_sport1m_iter_1900000")
    c3d_volume_mean_file = os.path.join(c3d_pretrained_model_and_volume_mean_dir, "c3d_sport1m_volume_mean.binaryproto")

    c3d_finetuning_solver = os.path.join(c3d_finetuning_dir, "c3d_sport1m_finetuning_solver.prototxt")
    c3d_finetuning_train = os.path.join(c3d_finetuning_dir, "c3d_sport1m_finetuning_train.prototxt")
    c3d_finetuning_sh = os.path.join(c3d_finetuning_dir, "c3d_sport1m_finetuning.sh")

    c3d_feature_extraction_train_prototxt = os.path.join(c3d_feature_extraction_for_train_dir, "c3d_sport1m_feature_extractor_frm_train.prototxt")
    c3d_feature_extraction_train_sh =  os.path.join(c3d_feature_extraction_for_train_dir, "c3d_sport1m_feature_extractor_train.sh")
    c3d_feature_extraction_test_prototxt = os.path.join(c3d_feature_extraction_for_test_dir,"c3d_sport1m_feature_extractor_frm_test.prototxt")
    c3d_feature_extraction_test_sh =  os.path.join(c3d_feature_extraction_for_test_dir,"c3d_sport1m_feature_extractor_test.sh")

    snapshot_prefix =  os.path.join(c3d_intermediate_model_snapshot_dir,"c3d_sport1m_finetune_whole")
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

    c3d_template_compute_volume_mean_sh = os.path.join(c3d_template_compute_volume_mean_dir,\
         "c3d_sport1m_compute_volume_mean.sh")

    c3d_template_finetuning_solver = os.path.join(c3d_template_finetuning_dir, \
        "c3d_sport1m_finetuning_solver.prototxt")
    c3d_template_finetuning_train= os.path.join(c3d_template_finetuning_dir, \
        "c3d_sport1m_finetuning_train.prototxt")
    c3d_template_finetuning_sh =  os.path.join(c3d_template_finetuning_dir, \
        "c3d_sport1m_finetuning.sh")
    
    c3d_template_feature_extractor_frm = os.path.join(c3d_template_feature_extraction_dir, \
        "c3d_sport1m_feature_extractor_frm.prototxt")
    c3d_template_feature_extractor_sh = os.path.join(c3d_template_feature_extraction_dir, \
        "c3d_sport1m_feature_extractor.sh")

if __name__ == "__main__":
    config_params = ConfigParams
    num_of_iters = config_params.max_iter / config_params.snapshot

    #CuongND. Don't clean up output folder beforehand
    # if os.path.exists(config_params.output_dir):
    #     shutil.rmtree(config_params.output_dir)
    # os.makedirs(config_params.output_dir)

    #CuongND. delete all .fc6, .fc7, .prob in old training

    result_dir= os.path.join(config_params.c3d_data_root,
                            config_params.kinect_train +'_'+config_params.data_type,
                            config_params.output_result_ext)
    delete_files_with_extension_in_folder(result_dir,'.fc6')
    delete_files_with_extension_in_folder(result_dir,'.fc7')
    delete_files_with_extension_in_folder(result_dir,'.prob')

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
   
    subject_list = ['Binh', 'Giang', 'Hung', 'Tan','Thuan']
    #subject_test=['Binh', 'Giang', 'Hung', 'Tan','Thuan'] #default for test
    subject_test=['Hung']

    subject_list = list(subject_list) 
    num_of_subjects = len(subject_list)
    kinect_train = config_params.kinect_train 
    snapshot = config_params.snapshot
    max_iter = config_params.max_iter
    
    for kinect_test in config_params.kinect_test_list:
        create_folder(
            os.path.join(
                config_params.output_dir, 
                "%s_test_on_%s_%s" % (kinect_train, kinect_test, config_params.date_time)))
        for subject in subject_list:
            create_folder(
                os.path.join(
                    config_params.output_dir, 
                    "%s_test_on_%s_%s" % (kinect_train, kinect_test, config_params.date_time),
                    subject))   
            for iter_ in range(snapshot, max_iter+1, snapshot):
                create_folder(
                    os.path.join(
                        config_params.output_dir, 
                        "%s_test_on_%s_%s" % (kinect_train, kinect_test, config_params.date_time),
                        subject,
                        "iter_%d" % (iter_)))

    #random.shuffle(subject_list)
    count=1
    for test_subject in subject_list:
        train_list = []
        test_list = [test_subject]
        #CuongND
        continue_test=False
        for subject in subject_test:
            if(test_subject==subject):
                continue_test=True
                break
        if (continue_test==False):
            continue

        for train_subject in subject_list:
            if train_subject == test_subject:
                continue
            train_list.append(train_subject)
            
        print "\n\nTEST SUBJECT " + str(count)+': '+test_subject +' ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
        print "BEGINS\n\n"
        count+=1
        # Train and Classify
        (result_list) = c3d_train_and_test(train_list, test_list, config_params) 

        # Output to text files   
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
                        "%s_test_on_%s_%s" % (kinect_train, kinect_test, config_params.date_time),
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
                        "%s_test_on_%s_%s" % (kinect_train, kinect_test, config_params.date_time),
                        test_subject,                        
                        file_name))
                
                # Save accuracy and loss in text files
                np.savetxt(
                    os.path.join(
                        config_params.output_dir,
                        "%s_test_on_%s_%s" % (kinect_train, kinect_test, config_params.date_time),
                        test_subject,
                        "%s_train_accuracy.txt" % (classification_method)),
                    r1_["train"].acc_list, 
                    delimiter = ' ', 
                    fmt = "%f" )    
                np.savetxt(
                    os.path.join(
                        config_params.output_dir,
                        "%s_test_on_%s_%s" % (kinect_train, kinect_test, config_params.date_time),
                        test_subject,
                        "%s_test_accuracy.txt" % (classification_method)),
                    r1_["test"].acc_list, 
                    delimiter = ' ', 
                    fmt = "%f" )  
                np.savetxt(
                    os.path.join(
                        config_params.output_dir,
                        "%s_test_on_%s_%s" % (kinect_train, kinect_test, config_params.date_time),
                        test_subject,
                        "%s_train_loss.txt" % (classification_method)),
                    r1_["train"].loss_list, 
                    delimiter = ' ', 
                    fmt = "%f" )    
                np.savetxt(
                    os.path.join(
                        config_params.output_dir,
                        "%s_test_on_%s_%s" % (kinect_train, kinect_test, config_params.date_time),
                        test_subject,
                        "%s_test_loss.txt" % (classification_method)),
                    r1_["test"].loss_list, 
                    delimiter = ' ', 
                    fmt = "%f" )     
                
                # Aggregate average results
                avg_result[kinect_test][classification_method]["train"] += np.array(r1_["train"].acc_list)
                avg_result[kinect_test][classification_method]["test"] += np.array(r1_["test"].acc_list)
                    
        
    # Output average accuracy 
    print ""
    print ""
    print ""
    print "OUTPUT AVERAGE ACCURACY"
    print ""
    print ""
    print ""
    # pdb.set_trace()
    for kinect_test, r0_ in avg_result.iteritems():
        for classification_method, r1_ in r0_.iteritems():
            file_name = "%s_avg_acc.png" % (classification_method)
            # pdb.set_trace()
            dump_plot_to_image_file(
                list(r1_["train"]/ (1.0 * num_of_subjects)), 
                list(r1_["test"] / (1.0 * num_of_subjects)), 
                config_params.max_iter,
                config_params.snapshot,
                "Train",
                "Test",
                "Average accuracy",
                os.path.join(
                    config_params.output_dir,
                    "%s_test_on_%s_%s" % (kinect_train, kinect_test, config_params.date_time),
                    file_name))
            np.savetxt(
                os.path.join(
                    config_params.output_dir,
                    "%s_test_on_%s_%s" % (kinect_train, kinect_test, config_params.datesnapsho_time),
                    "%s_train_accuracy.txt" % (classification_method)),
                list(r1_["train"] / (1.0 * num_of_subjects)), 
                delimiter = ' ', 
                fmt = "%f" ) 
            np.savetxt(
                os.path.join(
                    config_params.output_dir,
                    "%s_test_on_%s_%s" % (kinect_train, kinect_test, config_params.date_time),
                    "%s_test_accuracy.txt" % (classification_method)),
                list(r1_["test"] / (1.0 * num_of_subjects)), 
                delimiter = ' ', 
                fmt = "%f" ) 

                   

    

