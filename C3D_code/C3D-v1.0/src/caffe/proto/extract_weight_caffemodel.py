import caffe_pb2
import numpy as np
import os

caffe_model='/home/prdcv/PycharmProjects/c3d_luanvan/c3d_files/pretrained_model_and_volume_mean/c3d_ucf101_finetune_whole_iter_20000_fc6_1024_77.26%'
target_model='c3d_ucf101_finetune_whole_fc6_1024_fc7_1024'
temp_model='c3d_ucf101_finetune_whole_iter_100'
origin_weight_folder='origin_weight_bin'
modify_weight_folder='modify_weight_bin'
layer_names = ['conv1a', 'conv2a', 'conv3a', 'conv3b', 'conv4a', 'conv4b', 'conv5a', 'conv5b']
weight_shapes=dict()
weight_shapes['conv1a']=(64,3,3,3,3)
weight_shapes['conv2a']=(128,64,3,3,3)
weight_shapes['conv3a']=(256,128,3,3,3)
weight_shapes['conv3b']=(256,256,3,3,3)
weight_shapes['conv4a']=(512,256,3,3,3)
weight_shapes['conv4b']=(512,512,3,3,3)
weight_shapes['conv5a']=(512,512,3,3,3)
weight_shapes['conv5b']=(512,512,3,3,3)



def read_data(filename, len_to_read=-1, dtype='float32'):  # read .bin or .txt data file

    if ('.txt' in filename):
        with open(filename) as f:
            lines = []
            if (len_to_read > -1):
                count = 0
                for line in f:
                    if (count < len_to_read):
                        lines.append(line)
                    count = count + 1
            else:
                lines = f.readlines()
            x = np.array(lines)
            data = x.astype(np.float)

    if ('.bin' in filename):
        data = np.fromfile(os.path.join(filename), dtype=dtype)

    return data

def modify_caffemodel(layers_to_modify=['conv4b'], save_data=True):

    with open('final_c3d_ucf101_finetune_whole_iter_100', 'r') as f:
        cq2 = caffe_pb2.NetParameter()
        cq2.ParseFromString(f.read())

    layers = cq2.layers
    save_model=target_model
    for lc in layers:
        name = lc.name
        if (name == 'pre_pool4'):
            print lc
        print name
        for layer_name in layer_names:
            if (name == layer_name):
                weight = np.float32(np.array(lc.blobs[0].data))
                bias=np.float32(np.array(lc.blobs[0].data))
                if(save_data==True):
                    weight.tofile(os.path.join(origin_weight_folder,name+'_weight.bin'))
                    bias.tofile(os.path.join(origin_weight_folder,name+'_bias.bin'))
                    print 'Finish save data to layer '+name
                else:
                    for layer in layers_to_modify:
                        if (name == 'pre_pool4'):
                            print lc
                            save_model+='_'+name
                            lc.blobs[0].width = 1
                            lc.blobs[0].height = 1
                            lc.convolution_param.kernel_size = 1
                            data=read_data(os.path.join(modify_weight_folder,name+'_weight.bin'))
                            lc.blobs[0].data[:] = data
                            print 'Finish assign new data to layer '+name

    print 'Save all parameters to file '+save_model
    with open(save_model, 'wb') as f:
        f.write(cq2.SerializeToString())


def modify_weight(layers_to_modify=['conv2a']):
    for layer in layers_to_modify:
        print layer
        weight=read_data(os.path.join(origin_weight_folder,layer+'_weight.bin')).reshape(weight_shapes[layer])

        shape=weight.shape
        new_weight=np.zeros((shape[0],shape[1],shape[2],1,1))
        for i in range(shape[0]):
            if((i+1)%32==0):
                print i
            else:
                print i ,
            for j in range(shape[1]):
                for k in range(shape[2]):
                    sum=0
                    for kx in range(3):
                        for ky in range(3):
                            sum+=weight[i][j][k][kx][ky]
                    vl=sum/9
                    new_weight[i][j][k][0][0]=vl

        new_weight.tofile(os.path.join(modify_weight_folder,layer+'_weight.bin'))

def assign_data():
    print 'src data ' + target_model
    with open(target_model, 'r') as f:
        src_data = caffe_pb2.NetParameter()
        src_data.ParseFromString(f.read())
    src_layers = src_data.layers

    print 'dst data ' + temp_model
    with open(temp_model, 'r') as f:
        dst_data = caffe_pb2.NetParameter()
        dst_data.ParseFromString(f.read())
    dst_layers = dst_data.layers

    for layer_name in layer_names:
        for src_layer in src_layers:
            if (src_layer.name==layer_name):
                src_layer_data=src_layer
        for dst_layer in dst_layers:
            if (dst_layer.name==layer_name):
                dst_layer_data=dst_layer

        dst_layer_data.blobs[0].data[:]=np.float32(np.array(src_layer_data.blobs[0].data))
        dst_layer_data.blobs[1].data[:]=np.float32(np.array(src_layer_data.blobs[1].data))
        print 'Finish assign data from src for layer '+layer_name

    print 'Save all parameters to file '+'final_'+temp_model
    with open('final_'+temp_model, 'wb') as f:
        f.write(dst_data.SerializeToString())

if __name__ == "__main__":
    #assign_data()
    modify_caffemodel(save_data=False)
    #modify_weight()
    print('\nFinish.')