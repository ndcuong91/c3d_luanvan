import csv
import glob
import os
import os.path
from subprocess import call
from os import listdir
from os.path import isfile, isdir, join

C3D_HOME='/home/prdcv/PycharmProjects/luanvan/C3D'

def extract_files(vid_folders):
    """After we have all of our videos split between train and test, and
    all nested within folders representing their classes, we need to
    make a data file that we can reference when training our RNN(s).
    This will let us keep track of image sequences and other parts
    of the training process.

    We'll first need to extract images from each of the videos. We'll
    need to record the following data in the file:

    [train|test], class, filename, nb frames

    Extracting can be done with ffmpeg:
    `ffmpeg -i video.mpg image-%04d.jpg`
    """

    class_folders = [f for f in listdir(vid_folders) if isdir(join(vid_folders, f))]

    for vid_class in class_folders:

        class_files = [f for f in listdir(vid_folders+'/'+vid_class) if isfile(join(vid_folders+'/'+vid_class, f))]

        for vid_file in class_files:

            # Now extract it.
            src = vid_folders + '/' + vid_class + '/' + vid_file
            dest_dir = vid_folders.replace('/vids','') + '/frm/' + vid_class + '/' + vid_file.replace('.avi','')

            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)

            files = [f for f in listdir(dest_dir) if isfile(join(dest_dir, f))]
            if(len(files)>0):
                continue

            dest = dest_dir + '/%06d.jpg'
            call(["ffmpeg", "-i", src, dest])

             # Now get how many frames it is.
            total_files=[f for f in listdir(dest_dir) if isfile(join(dest_dir, f))]
            print("Generated %d frames for %s" % (len(total_files), vid_file))

    print("Extracted and wrote %d video files." % (len(vid_class)))

def mod_file_list(frm_file):
    with open(frm_file, 'r') as myfile:
        data = myfile.read().replace('/data/users/trandu/datasets/ucf101',C3D_HOME+'/C3D-v1.1/data/UCF-101')
        myfile.close()
    with open(frm_file, 'w') as myfile:
        myfile.write(data)
        myfile.close()
    return

def mod_file_prefix(frm_file):
    with open(frm_file, 'r') as myfile:
        data = myfile.read().replace('/data/users/trandu/datasets/ucf101/c3d_resnet18_r2',C3D_HOME+'/C3D-v1.1/data/UCF-101/frm')
        myfile.close()
    with open(frm_file, 'w') as myfile:
        myfile.write(data)
        myfile.close()
    return

def main():
    #extract frame from UCF101 videos
    extract_files(C3D_HOME+'/C3D-v1.1/data/UCF-101/vids')

    #modify file ucf101_video_frame.list and ucf101_video_frame.list
    mod_file_list(C3D_HOME+'/C3D-v1.1/examples/c3d_ucf101_feature_extraction/ucf101_video_frame.list')
    mod_file_prefix(C3D_HOME+'/C3D-v1.1/examples/c3d_ucf101_feature_extraction/ucf101_video_frame.prefix')
    mod_file_list(C3D_HOME+'/C3D-v1.0/examples/c3d_finetuning/test_01.lst')
    mod_file_list(C3D_HOME+'/C3D-v1.0/examples/c3d_finetuning/train_01.lst')
if __name__ == '__main__':
    main()
