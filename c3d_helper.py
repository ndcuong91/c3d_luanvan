import numpy as np
import array
from glob import glob
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import log_loss, confusion_matrix
from sklearn import datasets
from pprint import pprint
import pdb

def read_binary_blob(fn):
    """
    Read binary data in C3D format

    Shamelessly copied from: 

    https://github.com/facebook/C3D/blob/master/C3D-v1.0/examples/c3d_feature_extraction/extract_C3D_feature.py

    Input: 

    fn: File name (e.g: 000001.fc6-1)

    Output: 

    feature_vec: numpy array (size 1 x 4096 if layer is fc6-1 or fc7-1, or 1 x num_of_classes if layer is prob)

    Reference: 

    D. Tran, L. Bourdev, R. Fergus, L. Torresani, and M. Paluri, 
    Learning Spatiotemporal Features with 3D Convolutional Networks, ICCV 2015
    """

    f = open(fn, "rb")
    # read all bytes into a string
    s = f.read()
    f.close()
    (n, c, l, h, w) = array.array("i", s[:20])
    feature_vec = np.array(array.array("f", s[20:]))
    return feature_vec        

def get_average_of_all_features_in_a_directory(directory, feature_type):
    """
    Written by Dang Manh Truong (dangmanhtruong@gmail.com)

    This function retrieves the average of all C3D features in a directory

    Information on the structure of C3D feature files can be found at: 
    https://github.com/facebook/C3D/blob/master/C3D-v1.0/examples/c3d_feature_extraction/extract_C3D_feature.py

    or :

    https://github.com/facebook/C3D/tree/master/C3D-v1.0/examples/c3d_feature_extraction/script   
    E.g: directory = "/data/Hoctap/BASH" and feature_type = "fc6-1"    

    Input:

    directory: Directory of the feature files

    feature_type: Type of feature that has been extracted by C3D 

    Output:

    avg_features: Average of features, returned as a numpy array 

    Reference: 

    D. Tran, L. Bourdev, R. Fergus, L. Torresani, and M. Paluri, 
    Learning Spatiotemporal Features with 3D Convolutional Networks, ICCV 2015
    """    
    file_list = glob(directory + "/*." + feature_type)
    num_of_files = len(file_list)       
    feature_vec_list = []
    for file_name in file_list:       
        
        feature_vec = read_binary_blob(file_name)        
        feature_vec_list.append(feature_vec)           
    avg_features = np.zeros(feature_vec_list[0].shape)
    for feature_vec in feature_vec_list:
        avg_features += feature_vec
    avg_features = avg_features / num_of_files    
    return avg_features

# def classification_routine(X_train, Y_train, X_test, Y_test, feature_type, svm_type):     
#     """  
#     Written by Dang Manh Truong (dangmanhtruong@gmail.com)

#     Input: 

#     X_train, X_test: numpy array, size (num_of_examples x num_of_features)

#     Y_train, Y_test: 0,1,2,.....

#     feature_type: "fc6", "fc7" or "prob"

#     svm_type: "linear" or "rbf" (if feature_type == "prob" then svm_type should be "")

#     Output:

#     (acc_train, acc_test, loss_train, loss_test, confmat_train, confmat_test)

#     acc_train, acc_test: Accuracy (from 0 to 1)

#     loss_train, loss_test: Multinomial logistic loss (natural logarithm)

#     confmat_train, confmat_test: Confusion matrix as a numpy array, size (num_of_classes x num_of_classes)

#     Reference: 

#     D. Tran, L. Bourdev, R. Fergus, L. Torresani, and M. Paluri, 
#     Learning Spatiotemporal Features with 3D Convolutional Networks, ICCV 2015
#     """
    

#     if feature_type != "prob":
#         # Preprocessing
#         scaler = preprocessing.MinMaxScaler()
#         X_train = scaler.fit_transform(X_train)
#         X_test = scaler.transform(X_test)

#         # Train and test
#         classifier = OneVsRestClassifier(SVC(kernel = svm_type, probability=True))    
#         classifier.fit(X_train, Y_train) 

#         prob_train = classifier.predict_proba(X_train)
#         prob_test = classifier.predict_proba(X_test)
#         argmax_train = np.argmax(prob_train, axis = 1)       
#         argmax_test = np.argmax(prob_test, axis = 1)
#         prediction_train = classifier.classes_[argmax_train]
#         prediction_test = classifier.classes_[argmax_test]

#         Y_train = Y_train.flatten()
#         Y_test = Y_test.flatten()  
#         confmat_train = confusion_matrix(Y_train, prediction_train)
#         confmat_test = confusion_matrix(Y_test, prediction_test)
#         loss_train = log_loss(Y_train, prob_train[:, classifier.classes_])
#         loss_test = log_loss(Y_test, prob_test[:, classifier.classes_])
#     else:
#         # Use prob layer to predict directly
#         prob_train = X_train
#         prob_test = X_test      
#         prediction_train = np.argmax(prob_train, axis = 1)       
#         prediction_test = np.argmax(prob_test, axis = 1)
        
#         Y_train = Y_train.flatten()
#         Y_test = Y_test.flatten()   
        
#         confmat_train = confusion_matrix(Y_train, prediction_train)
#         confmat_test = confusion_matrix(Y_test, prediction_test)
#         loss_train = log_loss(Y_train, prob_train)
#         loss_test = log_loss(Y_test, prob_test)

#     num_of_train_corrects = np.sum(prediction_train == Y_train)       
#     num_of_test_corrects = np.sum(prediction_test == Y_test)
#     acc_train = num_of_train_corrects / float(Y_train.size) 
#     acc_test = num_of_test_corrects / float(Y_test.size)    
#     return (acc_train, acc_test, loss_train, loss_test, confmat_train, confmat_test)

def classification_routine(X_train, Y_train, X_test, Y_test, feature_type, svm_type, 
        train_list_mappings_dir_to_label, test_list_mappings_dir_to_label):
    """
    Input:
    
    train_list_mappings_dir_to_label: Dictionary of the form (train_dir_original, train_dir_optical, label)

    where train_dir_original and train_dir_optical are lists 

    test_list_mappings_dir_to_label: The same as train_list_mappings_dir_to_label    
    """
    if feature_type != "prob":
        scaler = preprocessing.MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train and test
        classifier = OneVsRestClassifier(SVC(kernel = svm_type, probability=True))    
        classifier.fit(X_train, Y_train) 

        prob_train = classifier.predict_proba(X_train)
        prob_test = classifier.predict_proba(X_test)
        argmax_train = np.argmax(prob_train, axis = 1)       
        argmax_test = np.argmax(prob_test, axis = 1)
        prediction_train = classifier.classes_[argmax_train]
        prediction_test = classifier.classes_[argmax_test]

        Y_train = Y_train.flatten()
        Y_test = Y_test.flatten()  
        confmat_train = confusion_matrix(Y_train, prediction_train)
        confmat_test = confusion_matrix(Y_test, prediction_test)
        loss_train = log_loss(Y_train, prob_train[:, classifier.classes_])
        loss_test = log_loss(Y_test, prob_test[:, classifier.classes_])  
    else:
        # Use prob layer to predict directly
        prob_train = X_train
        prob_test = X_test      
        prediction_train = np.argmax(prob_train, axis = 1)       
        prediction_test = np.argmax(prob_test, axis = 1)
        
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
            dict_train_misclassified[train_list_mappings_dir_to_label[i][0]] = (Y_train[i], prediction_train[i])
    dict_test_misclassified = {}
    for i in range(Y_test.size):
        if prediction_test[i] != Y_test[i]:
            dict_test_misclassified[test_list_mappings_dir_to_label[i][0]] = (Y_test[i], prediction_test[i])
    return (acc_train, acc_test, loss_train, loss_test, confmat_train, confmat_test, 
        dict_train_misclassified, dict_test_misclassified)    
    

def find_files_to_read(input_dir):
    """
    Written by Dang Manh Truong (dangmanhtruong@gmail.com)

    Find files to read using C3D input text file, whose structure is:

    <string_path> <starting_frame> <label>

    More information on the C3D input text file can be found at:

    https://github.com/facebook/C3D/blob/master/C3D-v1.0/examples/c3d_feature_extraction/prototxt/input_list_frm.txt

    or:

    https://docs.google.com/document/d/1-QqZ3JHd76JfimY4QKqOojcEaf5g3JS0lNh-FHTxLag/edit?usp=sharing

    Input: 

    input_dir: Absolute path of C3D input text file

    Output:

    directory_list: List of unique directories in the input text file, returned as a set

    dict_dir_to_label: A dictionary which maps a directory to the corresponding class

    (here we assume that the class information can be found from the directory)

    Reference: 

    D. Tran, L. Bourdev, R. Fergus, L. Torresani, and M. Paluri, 
    Learning Spatiotemporal Features with 3D Convolutional Networks, ICCV 2015
    """

    with open(input_dir) as f:
        dict_dir_to_label={}
        directory_list = set()
        for line in f:
            line = line.rstrip() # Remove trailing \n           
            line_splitted = line.split(' ')            
            directory = line_splitted[0]
            label = line_splitted[2]
            dict_dir_to_label[directory] = label
            directory_list.add(directory)           
    return (directory_list, dict_dir_to_label)

def load_data_for_classification(train_01_fulldir, test_01_fulldir, feature_type, num_of_classes, find_files_to_read_func):
    """
    Written by Dang Manh Truong (dangmanhtruong@gmail.com)

    This function loads C3D features into Python.
    
    Input:
    
    train_01_fulldir: Absolute path to C3D input file for the train set
    
    test_01_fulldir: Absolute path to C3D input file for the test set
    
    feature_type: "fc6", "fc7" or "prob"
    
    num_of_classes: Number of classes
    
    find_files_to_read_func: A function defines as follows
    
              (directory_list, dict_dir_to_label) = find_files_to_read(input_dir)
    
                  - input_dir: Absolute path of C3D input text file
    
                  - directory_list: List of unique directories in the input text file, returned as a set
    
                  - dict_dir_to_label: A dictionary which maps a directory to the corresponding class
    
    This allows for customization by users
    
    Output:

    X_train, X_test: numpy array, size (num_of_examples x num_of_features)

    Y_train, Y_test: 0,1,2,.....
    """
    

    
    if feature_type != "prob":
        X_train = np.zeros((0, 4096))
        # Y_train = np.zeros((0, 1))
        X_test = np.zeros((0, 4096))
        # Y_test = np.zeros((0, 1))

        Y_train = []
        Y_test = []
    else:
        X_train = np.zeros((0, num_of_classes))
        # Y_train = np.zeros((0, 1))
        X_test = np.zeros((0, num_of_classes))
        # Y_test = np.zeros((0, 1))
        Y_train = []
        Y_test = []
        pass
    
    line_num_id = 0
    train_mapped_to_dir = []
    # Get train data
    (train_dir_list, train_dict_dir_to_label) = find_files_to_read_func(train_01_fulldir)
    for train_dir in train_dir_list:    
        train_instance = get_average_of_all_features_in_a_directory(train_dir[:-1:], feature_type)
        label = train_dict_dir_to_label[train_dir]
        label = int(label)  
        X_train = np.vstack((X_train, train_instance))
        # Y_train = np.vstack((Y_train, label))  
        Y_train.append(label)
        train_mapped_to_dir.append(train_dir)
    Y_train = np.array(Y_train)

    # Get test data   
    test_mapped_to_dir = []
    (test_dir_list, test_dict_dir_to_label) = find_files_to_read_func(test_01_fulldir)
    for test_dir in test_dir_list:
        
        # pdb.set_trace()
        
        test_instance = get_average_of_all_features_in_a_directory(test_dir[:-1:], feature_type)
        label = test_dict_dir_to_label[test_dir]        
        label = int(label)
        X_test = np.vstack((X_test, test_instance))
        # Y_test = np.vstack((Y_test, label))
        Y_test.append(label)
        test_mapped_to_dir.append(test_dir)
    Y_test = np.array(Y_test)
    return (X_train, Y_train, X_test, Y_test, train_mapped_to_dir, test_mapped_to_dir)


if __name__ == "__main__":
    # Test modules
    # iris = datasets.load_iris()
    # X = iris.data[:, :2]  # we only take the first two features.
    # y = iris.target
    # X_train = X 
    # Y_train = y
    # X_test = X
    # Y_test = y
    # feature_type = "fc6"
    # svm_type = "linear"
    # (acc_train, acc_test, loss_train, loss_test, confmat_train, confmat_test) = \
    #     classification_routine(X_train, Y_train, X_test, Y_test, feature_type, svm_type)
    # print acc_train
    # print acc_test
    # print loss_train
    # print loss_test
    # print confmat_train
    # print confmat_test

    # # Prob layer
    # scaler = preprocessing.MinMaxScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    # classifier = OneVsRestClassifier(SVC(kernel = svm_type, probability=True))    
    # classifier.fit(X_train, Y_train) 
    # prob_train = classifier.predict_proba(X_train)
    # prob_test = classifier.predict_proba(X_test)
    # X_train = prob_train
    # X_test = prob_test
    # feature_type = "prob"
    # svm_type = ""
    # (acc_train, acc_test, loss_train, loss_test, confmat_train, confmat_test) = \
    #     classification_routine(X_train, Y_train, X_test, Y_test, feature_type, svm_type)
    # print "PROB layer:"
    # print acc_train
    # print acc_test
    # print loss_train
    # print loss_test
    # print confmat_train
    # print confmat_test    

    # test_01_fulldir = "/data/Hoctap/MICA/CODE/C3D_code_finetuning_with_confusion_matrix_and_loss_for_train_and_test/test_01.lst"
    # test_01_fulldir = "/home/dangmanhtruong95/Truong_Python_run_scripts/test_01.lst"
    # train_01_fulldir = test_01_fulldir
    # num_of_actions = 5
    # (test_dir_list, test_dict_dir_to_label) = find_files_to_read(test_01_fulldir)
    # pprint(test_dir_list)
    # pprint(test_dict_dir_to_label)

    # find_files_to_read_func = find_files_to_read
    # (X_train, Y_train, X_test, Y_test) = load_data_for_classification(train_01_fulldir, test_01_fulldir, "fc7", \
    #     num_of_actions, find_files_to_read_func) 
    # print X_train
    try:
        1 / 0
        pass
    except :
        print "hahaha"
        pass
    print "fdssfsdf"
    pass


