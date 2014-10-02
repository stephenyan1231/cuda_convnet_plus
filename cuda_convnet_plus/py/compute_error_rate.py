import numpy as np
import os

from util import *


if __name__=='__main__':
    data_file_dir  = \
    r'D:\yzc\proj\cuda-convnet-plus\cuda-convnet-data\imagenet\12_challenge\convnet_checkpoint\ConvNet__2013-10-26_07.58.01yzyu-server2_summary\logprob\43.99'
    data_batch_name = 'data_batch_'
    data_batch_range = range(501,521)
    multiview_test=True
    num_views=10
    
    correct_top1, correct_top5 = [], []
    for batch_num in data_batch_range:
        data_batch_file= os.path.join(data_file_dir,data_batch_name+str(batch_num))
        data = unpickle(data_batch_file)
        logprobs = data['data'] # shape=(num_views*num_imgs,num_classes)
        labels = data['labels'] # shape=(1,num_views*num_imgs)
        probs = np.exp(logprobs)
        num_classes = probs.shape[1]
        num_imgs = probs.shape[0] / num_views
        assert labels.shape[1] == num_imgs * num_views
        probs = probs.reshape((num_views, num_imgs,num_classes))
        
        labels = labels[:,:num_imgs]
        mean_probs = np.mean(probs, axis=0)
        # top 1 error
        sort_idx = np.argsort(mean_probs,axis=1)
        correct_top1 += list(sort_idx[:,num_classes-1] == labels[0,:])
        
        top_1_error = 1.0 - np.sum(sort_idx[:,num_classes-1] == labels[0,:])/np.single(num_imgs)
        # top 5 error
        correct = np.zeros((num_imgs))
        for i in range(5):
            correct += (sort_idx[:,num_classes-1-i] == labels[0,:])
        correct_top5 += list(correct)
        top_5_error = 1.0 - np.sum(correct)/np.single(num_imgs)
        
        print 'batch_num:%d num_imgs:%d num_views:%d num_classes:%d top-1 error:%f top-5 error:%f' % \
        (batch_num, num_imgs,num_views,num_classes,top_1_error,top_5_error)
    
    all_top1_error = 1.0 - np.sum(correct_top1) / np.single(len(correct_top1))
    all_top5_error = 1.0 - np.sum(correct_top5) / np.single(len(correct_top5))
    print 'In summary, top 1 error:%f top 5 error:%f' % (all_top1_error,all_top5_error)