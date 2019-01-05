#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 19:38:43 2018

@author: dangmanhtruong
"""
import os, shutil
from matplotlib import pyplot as plt 
import cv2
# from glob import glob

def list_all_folders_in_a_directory(directory):
    """
    Returns relative paths    
    Usage:
        folder_list = list_all_folder_in_a_directory(directory)
        for folder_name in folder_list:
            print folder_name
    """    
    # return glob(os.path.join(directory, "*/"))
    folder_list = (folder_name for folder_name in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, folder_name)))
    return folder_list
    pass

def list_all_folders_in_current_directory():
    """
    Returns relative paths
    Usage:
        folder_list = list_all_folders_in_current_directory()
        for folder_name in folder_list:
            print folder_name
    """ 
    # return glob("*/")
    directory = "./"
    folder_list = (folder_name for folder_name in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, folder_name)))
    return folder_list
    pass

def list_all_files_in_a_directory(directory):
    """
    Returns relative paths    
    Usage:
        file_list = list_all_files_in_a_directory(directory)
        for file_name in file_list:
            print file_name
    """
    file_list = (file_name for file_name in os.listdir(directory) 
         if os.path.isfile(os.path.join(directory, file_name)))
    return file_list
    pass

def list_all_files_in_current_directory():
    """
    Returns relative paths
    Usage:
        file_list = list_all_files_in_current_directory()
        for file_name in file_list:
            print file_name
    """
    directory = "./"
    file_list = (file_name for file_name in os.listdir(directory) 
         if os.path.isfile(os.path.join(directory, file_name)))
    return file_list
    pass

def copy_file(source, dest):
    shutil.copy2(source, dest)
    pass

def create_folder(folder_fullpath):
    """
    Create folder if not exists
    
    Input:
        folder_fullpath: Fullpath to the folder
    
    Output:
        None
    """
    if os.path.exists(folder_fullpath) is False:
        os.makedirs(folder_fullpath)
        
def copy_folder_structure(source_dir, dest_dir):
    """
    Copy folder structure from source_dir to dest_dir without copying any file    
    Note that source_dir and dest_dir should be absolute paths
    """
    # Delete dest_dir if exists!
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir)
    
    # Recursion
    source_folder_list = list_all_folders_in_a_directory(source_dir)
    for source_subfolder in source_folder_list:        
        os.makedirs(os.path.join(dest_dir, source_subfolder))
        copy_folder_structure(os.path.join(source_dir, source_subfolder), os.path.join(dest_dir, source_subfolder))
    pass

def remove_file(file_path):
    """
    Remove a file
    Note that file_path should be the absolute path to the file
    """
    os.remove(file_path)

def remove_folder(folder_path):    
    """
    Remove a folder and all of its content
    Note that folder_path should be the absolute path to the folder    
    """
    shutil.rmtree(folder_path)

def file_exists(file_path):
    """
    Check if file exists
    Note that file_path should be the absolute path to the file
    """    
    return os.path.isfile(file_path)

def folder_exists(folder_path):
    """
    Check if folder exists
    Note that folder_path should be the absolute path to the file    
    """    
    return os.path.isdir(folder_path)

def mirror_directory_tree(source_dir, dest_dir):
    """
    Mirror an entire directory tree with dummy files
    Useful when you need to code something quickly
    Written by Dang Manh Truong (dangmanhtruong@gmail.com)
    Please note that if dest_dir already exists when this function is called, it
    will be deleted completely!
    Example:
        source_dir = "/home/user/folder_1"
        dest_dir = "/home/user/folder_1_dummy"
    """
    
    # Delete dest_dir if exists!
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir)
    
    # Mirror files in source_dir's immediate subdirectory
    source_file_list = list_all_files_in_a_directory(source_dir)
    for source_file_name in source_file_list:        
        # Create dummy text file
        fid = open(os.path.join(dest_dir, source_file_name), "wt")
        fid.close()    

    # Recursion
    source_folder_list = list_all_folders_in_a_directory(source_dir)
    for source_sub_folder in source_folder_list:        
        os.makedirs(os.path.join(dest_dir, source_sub_folder))
        mirror_directory_tree(os.path.join(source_dir, source_sub_folder), os.path.join(dest_dir, source_sub_folder))
        
def display_image_in_actual_size(im_data):
    """
    Utility function. Shamelessly copied from:
        
    https://stackoverflow.com/questions/28816046/displaying-different-images-with-actual-size-in-matplotlib-subplot
    
    Input:
        
    im_data: Loaded from cv2.imread
    
    Output: None        
    
    """
    
    dpi = 80    
    height, width, depth = im_data.shape

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(cv2.cvtColor(im_data, cv2.COLOR_BGR2RGB), cmap='gray')

    plt.show()
        
