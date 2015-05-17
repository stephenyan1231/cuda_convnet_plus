# Copyright (c) 2011, Alex Krizhevsky (akrizhevsky@gmail.com)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# 
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distributionp.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys
import os
import numpy.random as nr
import numpy as np
import random
import time

from options import *
from util import *
from util_image import *
from data import *

# import skimage.io
# import skimage.color

class DataProviderException(Exception):
    pass

class CIFARDataProvider(LabeledMemoryDataProvider):
    def __init__(self, libModel, data_dir, batch_range, init_epoch=1, init_batch_idx=None, epochBatchPerm=None, dp_params={}, test=False):
        LabeledMemoryDataProvider.__init__(self, libModel, data_dir, batch_range, init_epoch, init_batch_idx, epochBatchPerm, dp_params, test)
        self.data_mean = self.batch_meta['data_mean']
        self.num_colors = 3
        self.img_size = 32
        
        # Subtract the mean from the data and make sure that both data and
        # labels are in single-precision floating point.
        for d in self.data_dic:
            # This converts the data matrix to single precision and makes sure that it is C-ordered
            d['data'] = np.require((d['data'] - self.data_mean), dtype=np.single, requirements='C')
            d['labels'] = np.require(d['labels'].reshape((1, d['data'].shape[1])), dtype=np.single, requirements='C')

    def get_next_batch(self):
        epoch, batch_idx, batchnum, epochBatchPerm, datadic = LabeledMemoryDataProvider.get_next_batch(self)
        return epoch, batch_idx, batchnum, epochBatchPerm, [datadic['data'], datadic['labels']]

    # Returns the dimensionality of the two data matrices returned by get_next_batch
    # idx is the index of the matrix. 
    def get_data_dims(self, idx=0):
        return self.img_size ** 2 * self.num_colors if idx == 0 else 1
    
    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return np.require((data + self.data_mean).T.reshape(data.shape[1], 3, self.img_size, self.img_size).swapaxes(1, 3).swapaxes(1, 2) / 255.0, dtype=np.single)
    
class CroppedCIFARDataProvider(LabeledMemoryDataProvider):
    def __init__(self, libModel, data_dir, batch_range=None, init_epoch=1, init_batch_idx=None, epochBatchPerm=None, dp_params=None, test=False):
        LabeledMemoryDataProvider.__init__(self, libModel, data_dir, batch_range, init_epoch, init_batch_idx, epochBatchPerm, dp_params, test)

        self.border_size = dp_params['crop_border']
        self.inner_size = 32 - self.border_size * 2
        self.multiview = dp_params['multiview_test'] and test
        self.num_views = 5 * 2
        self.data_mult = self.num_views if self.multiview else 1
        self.num_colors = 3
        
        for d in self.data_dic:
            d['data'] = np.require(d['data'], requirements='C')
            d['labels'] = np.require(np.tile(d['labels'].reshape((1, d['data'].shape[1])), (1, self.data_mult)), requirements='C')
        
        self.cropped_data = [np.zeros((self.get_data_dims(), self.data_dic[0]['data'].shape[1] * self.data_mult), dtype=np.single) for x in xrange(2)] 
        # two times for horizontal reflection?

        self.batches_generated = 0
        self.data_mean = self.batch_meta['data_mean'].reshape((3, 32, 32))[:, self.border_size:self.border_size + self.inner_size, self.border_size:self.border_size + self.inner_size].reshape((self.get_data_dims(), 1))

    def get_next_batch(self):
        epoch, batch_idx, batchnum, epochBatchPerm, datadic = LabeledMemoryDataProvider.get_next_batch(self)

        cropped = self.cropped_data[self.batches_generated % 2]

        self.__trim_borders(datadic['data'], cropped)
        cropped -= self.data_mean
        self.batches_generated += 1
        return epoch, batch_idx, batchnum, epochBatchPerm, [cropped, datadic['labels']]
        
    def get_data_dims(self, idx=0):
        return self.inner_size ** 2 * 3 if idx == 0 else 1

    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return np.require((data + self.data_mean).T.reshape(data.shape[1], 3, self.inner_size, self.inner_size).swapaxes(1, 3).swapaxes(1, 2) / 255.0, dtype=np.single)
    
    def __trim_borders(self, x, target):
        y = x.reshape(3, 32, 32, x.shape[1])

        if self.test:  # don't need to loop over cases
            if self.multiview:
                start_positions = [(0, 0), (0, self.border_size * 2),
                                   (self.border_size, self.border_size),
                                  (self.border_size * 2, 0), (self.border_size * 2, self.border_size * 2)]
                end_positions = [(sy + self.inner_size, sx + self.inner_size) for (sy, sx) in start_positions]
                for i in xrange(self.num_views / 2):
                    pic = y[:, start_positions[i][0]:end_positions[i][0], start_positions[i][1]:end_positions[i][1], :]
                    target[:, i * x.shape[1]:(i + 1) * x.shape[1]] = pic.reshape((self.get_data_dims(), x.shape[1]))
                    target[:, (self.num_views / 2 + i) * x.shape[1]:(self.num_views / 2 + i + 1) * x.shape[1]] = pic[:, :, ::-1, :].reshape((self.get_data_dims(), x.shape[1]))
            else:
                pic = y[:, self.border_size:self.border_size + self.inner_size, self.border_size:self.border_size + self.inner_size, :]  # just take the center for now
                target[:, :] = pic.reshape((self.get_data_dims(), x.shape[1]))
        else:
            for c in xrange(x.shape[1]):  # loop over cases
                startY, startX = nr.randint(0, self.border_size * 2 + 1), nr.randint(0, self.border_size * 2 + 1)
                endY, endX = startY + self.inner_size, startX + self.inner_size
                pic = y[:, startY:endY, startX:endX, c]
                if nr.randint(2) == 0:  # also flip the image with 50% probability
                    pic = pic[:, :, ::-1]
                target[:, c] = pic.reshape((self.get_data_dims(),))
    
class DummyConvNetDataProvider(LabeledDummyDataProvider):
    def __init__(self, data_dim):
        LabeledDummyDataProvider.__init__(self, data_dim)
        
    def get_next_batch(self):
        epoch, batch_idx, batchnum, epochBatchPerm, dic = LabeledDummyDataProvider.get_next_batch(self)
        
        dic['data'] = np.require(dic['data'].T, requirements='C')
        dic['labels'] = np.require(dic['labels'].T, requirements='C')
        
        return epoch, batch_idx, batchnum, epochBatchPerm, [dic['data'], dic['labels']]
    
    # Returns the dimensionality of the two data matrices returned by get_next_batch
    def get_data_dims(self, idx=0):
        return self.batch_meta['num_vis'] if idx == 0 else 1
    
    
    
    
class LabeledMemoryBatchDataProvider(LabeledDataProvider):
    def __init__(self, libModel, data_dir, batch_range, init_epoch=1, init_batch_idx=None, epochBatchPerm=None, dp_params={}, test=False):
        LabeledDataProvider.__init__(self, libModel, data_dir, batch_range, init_epoch,
                                     init_batch_idx, epochBatchPerm, dp_params, test)
        self.curr_bat_dat_dic = []
        self.data_mean = self.batch_meta['data_mean']  # data_mean shape: (channel*height*width,1)
        
        self.num_colors = 3
        self.img_size = 256
        self.PCA_pixel_alter = dp_params['PCA_pixel_alter']
    def get_next_batch(self):
        self.load_batch_data()
        epoch, batch_idx, batchnum, epochBatchPerm = self.curr_epoch, self.batch_idx, self.curr_batchnum, self.epochBatchPerm
        self.advance_batch()
        return epoch, batch_idx, batchnum, epochBatchPerm, self.curr_bat_dat_dic
            
    def load_batch_data(self, zero_mean_data=1):
        oaStart = time.time()
        
        epoch, batchnum = self.curr_epoch, self.curr_batchnum
#         print '\nLabeledMemoryBatchDataProvider::load batch_Data batchnum:%d fn:%s' % (batchnum,self.get_data_file_name(batchnum))
        start = time.time()
        self.curr_bat_dat_dic = unpickle(self.get_data_file_name(batchnum))
        elapsed1 = time.time() - start
        
        start = time.time()
        
        if zero_mean_data == 1:
            self.curr_bat_dat_dic['data'] = np.require((self.curr_bat_dat_dic['data'] - self.data_mean), \
                                                    dtype=np.single, requirements='C')        
        else:
#             print 'no zero mean data'
            self.curr_bat_dat_dic['data'] = np.require(self.curr_bat_dat_dic['data'],
                                                  dtype=np.single, requirements='C')            
        elapsed2 = time.time() - start
        
        if not self.test == DataProvider.DP_PREDICT:
            if len(self.curr_bat_dat_dic['labels'].shape) == 1 :
                self.curr_bat_dat_dic['labels'] = np.require(self.curr_bat_dat_dic['labels'].reshape((1, self.curr_bat_dat_dic['data'].shape[1])),
                                                       dtype=np.single, requirements='C')
            elif len(self.curr_bat_dat_dic['labels'].shape) == 2 :
                self.curr_bat_dat_dic['labels'] = np.require(self.curr_bat_dat_dic['labels'],
                                                       dtype=np.single, requirements='C')            
            else:
                raise DataProviderException("the dim of label data array at most is 2-D ")
        else:
            dummy_labels = np.zeros((1, self.curr_bat_dat_dic['data'].shape[1]), dtype=np.single)
            self.curr_bat_dat_dic['labels'] = dummy_labels
            
#         print "elapsed1,elapsed2:%f %f\n" % (elapsed1, elapsed2)
        
        if self.PCA_pixel_alter == 1:
#             print 'do PCA pixel altering'
            
            # PCA_evecs shape: (height, width,channel,channel)
#             PCA_evecs=self.batch_meta['PCA_evecs']
#             PCA_scaled_evecs = np.zeros_like(PCA_evecs)
            # PCA_evals shape: (height, width,channel)
#             PCA_evals=self.batch_meta['PCA_evals']
         

            # PCA_scaled_evecs shape: (channel* height *width, channel)
            PCA_scaled_evecs = self.batch_meta['PCA_scaled_evecs']

            
            start = time.time()
            # sample scale for each image
            sigma = 0.2
            batchImgNum = self.curr_bat_dat_dic['data'].shape[1]
            randScales = sigma * np.random.randn(self.num_colors, batchImgNum)
            elapsed3 = time.time() - start
            
#             randScales=sigma * np.random.randn(batchImgNum,self.num_colors)
            # use broadcasting 
            start = time.time()
            delta = np.sum(PCA_scaled_evecs[:, :, np.newaxis] * randScales, 1)
            elapsed4 = time.time() - start
            
            start = time.time()
            self.curr_bat_dat_dic['data'] += delta
            elapsed5 = time.time() - start
#             print "elapsed3,4,5:%.4f %.4f %.4f" % (elapsed3,elapsed4,elapsed5)
#             for i in range(self.num_colors):
#                 PCA_scaled_evecs_col=PCA_scaled_evecs[:,i]
#                 randScalesRow=randScales[:,i]
#                 # compute outer product using broadcasting
#                 # delta shape:  (channel*height*width,batchImgNum)
#                 delta = PCA_scaled_evecs_col[:,np.newaxis]*randScalesRow
#                 self.curr_bat_dat_dic['data'] += delta
            
#             for i in xrange(0,batchImgNum):
#                 # delta shape: (height*width*channel,)
#                 delta=np.sum(PCA_scaled_evecs_view*randScales[i,:],1)
#                 delta=np.reshape(delta,(self.img_size,self.img_size,self.num_colors))
#                 # delta_view shape: (channel,height,width)
#                 delta_view=delta.swapaxes(0,2).swapaxes(1,2)
#                 imgData=self.curr_bat_dat_dic['data'][:,i]
#                 imgData=np.reshape(imgData,(self.num_colors,self.img_size,self.img_size))
#                 imgData+=delta_view
#                 self.curr_bat_dat_dic['data'][:,i]=imgData.flatten()
        oaElapsed = time.time() - oaStart
#         print "oaElapsed:%f" % oaElapsed
#         print 'load_batch_data el_1,el_2:%f %f' % (elapsed1,elapsed2)
#         print 'data shape'
#         print self.curr_bat_dat_dic['data'].shape
#         print 'labels shape'
#         print self.curr_bat_dat_dic['labels'].shape

# only current batch data is stored in main memory
class ImagenetDataProvider(LabeledMemoryBatchDataProvider):
    def __init__(self, libModel, data_dir, batch_range, init_epoch=1,
                 init_batch_idx=None, epochBatchPerm=None, dp_params={}, test=False):
        LabeledMemoryBatchDataProvider.__init__(self, libModel, data_dir, batch_range, init_epoch,
                 init_batch_idx, epochBatchPerm, dp_params, test)

        
    def get_next_batch(self):
        epoch, batch_idx, batchnum, epochBatchPerm, curr_bat_dat_dic = LabeledMemoryBatchDataProvider.get_next_batch(self)
        return epoch, batch_idx, batchnum, epochBatchPerm, [curr_bat_dat_dic['data'], curr_bat_dat_dic['labels']]
               
    def get_data_dims(self, idx=0):
        return self.img_size ** 2 * self.num_colors if idx == 0 else 1
        
    def get_plottable_data(self, data):
        return np.require((data + self.data_mean).T.reshape(data.shape[1], 3, self.img_size, self.img_size).swapaxes(1, 3).swapaxes(1, 2) / 255.0, dtype=np.single)
    
class CroppedImagenetDataProvider(LabeledMemoryBatchDataProvider):
    def __init__(self, libModel, data_dir, batch_range, init_epoch=1,
                 init_batch_idx=None, epochBatchPerm=None, dp_params={}, test=False):
        LabeledMemoryBatchDataProvider.__init__(self, libModel, data_dir, batch_range, init_epoch,
                                     init_batch_idx, epochBatchPerm, dp_params, test)
        
        self.border_size = dp_params['crop_border']
        self.inner_size = self.img_size - self.border_size * 2
        self.multiview = dp_params['multiview_test'] and test
        self.num_views = 5 * 2
        self.data_mult = self.num_views if self.multiview else 1
#         self.num_colors = 3
           
        # self.cropped_data = [np.zeros((self.get_data_dims(), self.data_dic[0]['data'].shape[1]*self.data_mult), dtype=np.single) for x in xrange(2)] 
        # two times for horizontal reflection?
        
        self.batches_generated = 0
        # self.data_mean = self.batch_meta['data_mean'].reshape((3,self.img_size,self.img_size))[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size].reshape((self.get_data_dims(), 1))

#     def import_model(self):
#         lib_name = "pyconvnet" if is_windows_machine() else "_ConvNet"
#         print "========================="
#         print "Importing %s C++ module" % lib_name
#         self.libmodel = __import__(lib_name) 
        
    def get_next_batch(self):
        start = time.time()
        epoch, batch_idx, batchnum, epochBatchPerm, curr_bat_dat_dic = \
        LabeledMemoryBatchDataProvider.get_next_batch(self)
        elap1 = time.time() - start
        
        self.currBatchImgNum = curr_bat_dat_dic['data'].shape[1]
        # cropped = self.cropped_data[self.batches_generated % 2]
        
        start = time.time()
#         cropped = np.zeros((self.get_data_dims(), curr_bat_dat_dic['data'].shape[1]*self.data_mult), dtype=np.single)
#         self.__trim_borders(curr_bat_dat_dic['data'], cropped)
        cropped = self.__trim_borders(curr_bat_dat_dic['data'])
        # cropped -= self.data_mean
        elap2 = time.time() - start
        
        # display time cost for data reading and image cropping
#         print "batch data reading %f secs random image cropping %f secs" %(elap1,elap2)
        self.batches_generated += 1
        
        return epoch, batch_idx, batchnum, epochBatchPerm, [cropped, np.tile(curr_bat_dat_dic['labels'], (1, self.data_mult))], [elap1, elap2]       
            
    def get_data_dims(self, idx=0):
        return self.inner_size ** 2 * 3 if idx == 0 else 1

    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        data_mean_3d = self.data_mean.reshape((self.num_colors, self.img_size, self.img_size))
        data_mean_center = data_mean_3d[:, self.border_size:self.border_size + self.inner_size, self.border_size:self.border_size + self.inner_size]
        data_mean_center = data_mean_center.reshape((3 * self.inner_size * self.inner_size, 1))
        return np.require((data + data_mean_center).T.reshape(data.shape[1], 3, self.inner_size, self.inner_size).swapaxes(1, 3).swapaxes(1, 2) / 255.0, dtype=np.single)

                   
    def __trim_borders(self, x):
        target = np.zeros((self.get_data_dims(), self.currBatchImgNum * self.data_mult), dtype=np.single)
        y = x.reshape(self.num_colors, self.img_size, self.img_size, x.shape[1])

        if self.test:  # don't need to loop over cases
            if self.multiview:
                start_positions = [(0, 0), (0, self.border_size * 2),
                                   (self.border_size, self.border_size),
                                  (self.border_size * 2, 0), (self.border_size * 2, self.border_size * 2)]
                end_positions = [(sy + self.inner_size, sx + self.inner_size) for (sy, sx) in start_positions]
                for i in xrange(self.num_views / 2):
                    pic = y[:, start_positions[i][0]:end_positions[i][0], start_positions[i][1]:end_positions[i][1], :]
                    target[:, i * x.shape[1]:(i + 1) * x.shape[1]] = pic.reshape((self.get_data_dims(), x.shape[1]))
                    target[:, (self.num_views / 2 + i) * x.shape[1]:(self.num_views / 2 + i + 1) * x.shape[1]] = pic[:, :, ::-1, :].reshape((self.get_data_dims(), x.shape[1]))
            else:
                pic = y[:, self.border_size:self.border_size + self.inner_size, self.border_size:self.border_size + self.inner_size, :]  # just take the center for now
                target[:, :] = pic.reshape((self.get_data_dims(), x.shape[1]))
        else:
            imgN = x.shape[1]            
#             randomness = 1
#             patchSize=np.ones((2),dtype=np.single)*self.inner_size

            startPos = np.single(nr.randint(0, self.border_size * 2 + 1, (imgN, 2)))
            horiFlip = np.single(nr.randint(0, 2, (imgN)))
            
#             startT=time.time()            
#             self.libModel.cropImage([imgData,target,startPos,horiFlip],self.img_size,self.img_size,3,imgN, self.inner_size,self.inner_size)
            self.libModel.cropImage([y, target, startPos, horiFlip], self.img_size, self.img_size, 3, imgN, self.inner_size, self.inner_size)
#             cropTime=time.time()-startT
#             target=target.reshape(self.num_colors, self.currBatchImgNum,self.inner_size,self.inner_size)
#             target=target.swapaxes(1,3).swapaxes(1,2).reshape(self.get_data_dims(),self.currBatchImgNum*self.data_mult)
            
#             startT=time.time()
#             target2 = np.zeros((self.get_data_dims(), self.currBatchImgNum*self.data_mult), dtype=np.single)
#             endPos=startPos+patchSize
#              
#             for c in xrange(x.shape[1]): # loop over cases
# #                 if randomness:
# #                     startY, startX = nr.randint(0,self.border_size*2 + 1), nr.randint(0,self.border_size*2 + 1)
# #                 else:
# #                     startY, startX = self.border_size + 1, self.border_size + 1
# #                 endY, endX = startY + self.inner_size, startX + self.inner_size
#                 pic = y[:,startPos[c,0]:endPos[c,0],startPos[c,1]:endPos[c,1], c]
# #                 pic = y[:,startY:endY,startX:endX, c]
#                 if randomness:
# #                     if nr.randint(2) == 0: # also flip the image with 50% probability
#                     if horiFlip[c]==0:
#                         pic = pic[:,:,::-1]
#                 target2[:,c] = pic.reshape((self.get_data_dims(),))
#             pythonCropTime=time.time()-startT
#             print "cropTime:%f pythonCropTime:%f" % (cropTime,pythonCropTime)
#              
#             diff=np.sum(np.sum(np.abs(target-target2),1))
#             if diff==0:
#                 print "\n :) no difference"
#             else:
#                 print "\n :( difference!"
#                 sys.exit(1)
        return target
    def print_batch_timing(self, timing):
        print '(loading:%.1f cropping:%.1f)' % (timing[0], timing[1])

class MITfivekDataProvider_4(LabeledMemoryBatchDataProvider):    
    def __init__(self, libModel, data_dir, batch_range, init_epoch=1,
                 init_batch_idx=None, epochBatchPerm=None, dp_params={}, test=False):
        LabeledMemoryBatchDataProvider.__init__(self, libModel, data_dir, batch_range, init_epoch,
                 init_batch_idx, epochBatchPerm, dp_params, test)

        self.regress_L_channel_only = dp_params['regress_L_channel_only']
        self.use_local_context_ftr = dp_params['use_local_context_ftr']
        ''' flag if we compute 
            1) mean colors for each of 25 contextual pooling regions
            2) color histogram (32 bins each) in a,b channels for a local window centered at the segment
        '''          
        self.use_local_context_color_ftr = dp_params['use_local_context_color_ftr']
        
        self.IMG_GLOBAL_FTR_ID = 0
        self.CTR_PIX_FTR_ID = 1
        self.COLOR_BASIS_ID = 2
        self.ENH_COLOR_ID = 3
        if self.use_local_context_ftr:
            self.PIX_CONTEXT_SEM_FTR_ID = 4
            self.PIX_COLOR_HIST_FTR = 5
            self.CONTEXTCOLOR_FTR = 6         
            
        self.global_ftr_mean = self.batch_meta['global_ftr_mean']
        
        self.local_context_paras = self.batch_meta['local_context_paras']
        if self.use_local_context_ftr:
            self.pixContextSemMean = self.batch_meta['pixContextSemMean']
            if self.use_local_context_color_ftr:
                self.pixContextColorMean = self.batch_meta['pixContextColorMean']
                self.pixColorHistMean = self.batch_meta['pixColorHistMean']
                self.local_context_color_paras = self.batch_meta['local_context_color_paras']

        self.img_global_ftr = {}
        for img in self.batch_meta['imgs']:
            globalFtrPath=os.path.join(self.batch_meta['in_img_global_ftr_dir'],img)
            globalFtr=unpickle(globalFtrPath)['pix_global_ftr']
            self.img_global_ftr[img]=globalFtr
           
        self.img_size = self.batch_meta['img_size']  # 1 when no patch exists
        self.num_colors = self.batch_meta['num_colors']
                
        print '--------complete MITfivekDataProvider_4 init-------'
        
    def get_img_size_num_colors(self):
        return self.img_size, self.num_colors        
        
        
    def get_batches_meta(self):
        return self.batch_meta
        
    '''in testing stage, prepare batch data to feed into NN'''
    def prepare_batch_data(self, patches, aux_data):
        new_setting = 1
        if not hasattr(self, 'patch_img_size'):
            ''' shape: pix_ftr_dim * n '''
            in_pix_ftr = patches
            assert len(patches.shape) == 2
            num = patches.shape[1]
        else:
            num, side, side, ch = patches.shape[0], patches.shape[1], patches.shape[2], patches.shape[3]
            patches_view = np.swapaxes(patches, 0, 3)


        if self.use_local_context_ftr:
            if self.use_local_context_color_ftr:
                in_img_nms, gt_color, pix_cont_sem_ftr, pix_color_hist_ftr, \
                pix_cont_color_ftr = aux_data[0], aux_data[1], aux_data[2], aux_data[3], aux_data[4]
            else:
                in_img_nms, gt_color, pix_cont_sem_ftr = \
                aux_data[0], aux_data[1], aux_data[2]                    
        else:
            in_img_nms, gt_color = aux_data[0], aux_data[1]
        assert num == len(in_img_nms)
        
        st_time = time.time()

        img_global_ftr_dim = self.get_data_dims(self.IMG_GLOBAL_FTR_ID)
        
        if not in_img_nms[0] in self.img_global_ftr.keys():
            '''new images appear in testing image list'''
            globalFtrPath = os.path.join(self.batch_meta['in_img_global_ftr_dir'],in_img_nms[0])
            globalFtr = unpickle(globalFtrPath)['pix_global_ftr']
            globalFtr -= self.global_ftr_mean
            img_global_ftr = np.tile(globalFtr.reshape((img_global_ftr_dim, 1)), (1, num))                 
        else:
            globalFtr = self.img_global_ftr[in_img_nms[0]] - self.global_ftr_mean
            img_global_ftr = np.tile(globalFtr.reshape((img_global_ftr_dim, 1)), (1, num))
        
        elapsed_1 = time.time() - st_time
        pix_ftr = in_pix_ftr - self.data_mean[:, np.newaxis]
 
        dummy_color_basis = np.zeros((self.get_data_dims(self.COLOR_BASIS_ID), num), dtype=np.single)
        
        if self.regress_L_channel_only:
            gt_color = gt_color[0, :].reshape((1, num))
        
        if self.use_local_context_ftr:
            pix_cont_sem_ftr = pix_cont_sem_ftr - self.pixContextSemMean[:, np.newaxis]
            if self.use_local_context_color_ftr:
                pix_cont_color_ftr = pix_cont_color_ftr - self.pixContextColorMean[:, np.newaxis]
                pix_color_hist_ftr = pix_color_hist_ftr - self.pixColorHistMean[:, np.newaxis]

        if new_setting == 1:
            img_global_ftr *= 100        
            pix_ftr *= 1
            if self.use_local_context_ftr:
                pix_cont_sem_ftr *= 50
                if self.use_local_context_color_ftr:
                    pix_color_hist_ftr *= 2e2
                    pix_cont_color_ftr *= 2
                            
        st_time = time.time()
        img_global_ftr = np.require(img_global_ftr, dtype=np.single, requirements='C')
        pix_ftr = np.require(pix_ftr, dtype=np.single, requirements='C')
        dummy_color_basis = np.require(dummy_color_basis, dtype=np.single, requirements='C')        
        gt_color = np.require(gt_color, dtype=np.single, requirements='C')
        if self.use_local_context_ftr:
            pix_cont_sem_ftr = np.require(pix_cont_sem_ftr, dtype=np.single, requirements='C')                      
            if self.use_local_context_color_ftr:
                pix_cont_color_ftr = np.require(pix_cont_color_ftr, dtype=np.single, requirements='C')
                pix_color_hist_ftr = np.require(pix_color_hist_ftr, dtype=np.single, requirements='C')

        
        elapsed_2 = time.time() - st_time
        if self.use_local_context_ftr:
            if self.use_local_context_color_ftr:
                return [img_global_ftr, pix_ftr, dummy_color_basis, gt_color, pix_cont_sem_ftr, \
                        pix_color_hist_ftr, pix_cont_color_ftr], \
                    [elapsed_1, elapsed_2]                        
            else:
                return [img_global_ftr, pix_ftr, dummy_color_basis, gt_color, pix_cont_sem_ftr], \
                    [elapsed_1, elapsed_2]
        else:
            return [img_global_ftr, pix_ftr, dummy_color_basis, gt_color], [elapsed_1, elapsed_2]


    def load_batch_data(self, zero_mean_data=0):
        LabeledMemoryBatchDataProvider.load_batch_data(self, zero_mean_data)

    def get_next_batch(self):
        new_setting = 1
        epoch, batch_idx, batch_num, epoc_batch_perm, curr_batchdat_dic = \
        LabeledMemoryBatchDataProvider.get_next_batch(self)

        st_time = time.time()
        num_imgs = curr_batchdat_dic['data'].shape[1]
            
        img_global_ftr_dim = self.get_data_dims(self.IMG_GLOBAL_FTR_ID)
        img_global_ftr = np.zeros((num_imgs,img_global_ftr_dim),dtype=np.single)
        assert num_imgs == len(curr_batchdat_dic['patch_to_image_name'])
        for i in range(num_imgs):
            img_global_ftr[i,:]=self.img_global_ftr[curr_batchdat_dic['patch_to_image_name'][i]]
            
        elapsed3 = time.time() - st_time
        
        st_time = time.time()
        if not hasattr(self, 'patch_img_size'):
            pix_ftr = curr_batchdat_dic['data'] - self.data_mean[:, np.newaxis]               
        else:
            pix_ftr = curr_batchdat_dic['patch_data'] - self.batch_meta['patch_data_mean'][:, np.newaxis]     
            
        ''' shape: num_imgs*segment_random_sample_num*3 '''
        in_pix_colors = curr_batchdat_dic['in_pix_data']
        segment_random_sample_num = in_pix_colors.shape[1]
        assert num_imgs == in_pix_colors.shape[0]
        in_pix_colors = in_pix_colors.reshape((num_imgs * segment_random_sample_num, 3))
        ''' shape: (num_imgs*segment_random_sample_num,10) '''
        in_pix_color_basis = quad_poly_color_basis(in_pix_colors)
        basis_dim = in_pix_color_basis.shape[1]
        assert basis_dim == 10
        in_pix_color_basis = in_pix_color_basis.reshape((num_imgs, segment_random_sample_num, basis_dim))
        # new shape: (segment_random_sample_num,10,num_imgs)
        in_pix_color_basis = in_pix_color_basis.swapaxes(0, 2).swapaxes(0, 1)
        in_pix_color_basis = in_pix_color_basis.reshape((segment_random_sample_num * basis_dim, num_imgs))
        
        enh_pix_colors = curr_batchdat_dic['labels']

        assert enh_pix_colors.shape[0] / 3 == segment_random_sample_num
        assert enh_pix_colors.shape[1] == num_imgs
        enh_pix_colors = enh_pix_colors.reshape((segment_random_sample_num, 3, num_imgs))
        if self.regress_L_channel_only:
            enh_pix_colors = enh_pix_colors[:, 0, :]
        else:
            enh_pix_colors = enh_pix_colors.reshape((segment_random_sample_num * 3, num_imgs))
        if self.use_local_context_ftr:
            ''' shape: dim * num_imgs '''
            pixContextSemFtr = curr_batchdat_dic['pixContextSemFtr'] - self.pixContextSemMean[:, np.newaxis]
            if self.use_local_context_color_ftr:
                pix_cont_color_ftr = curr_batchdat_dic['pixContextColor'] - self.pixContextColorMean[:, np.newaxis]
                pix_color_hist_ftr = curr_batchdat_dic['pixColorHist'] - self.pixColorHistMean[:, np.newaxis]
  
        elapsed4 = time.time() - st_time
        
        if new_setting == 1:
            img_global_ftr *= 100        
            pix_ftr *= 1
            if self.use_local_context_ftr:
                pixContextSemFtr *= 50
                if self.use_local_context_color_ftr:
                    pix_color_hist_ftr *= 2e2
                    pix_cont_color_ftr *= 2
                    
            
            
        st_time = time.time()   
        img_global_ftr = np.require(img_global_ftr.transpose(), dtype=np.single, requirements='C')
        pix_ftr = np.require(pix_ftr, dtype=np.single, requirements='C')              
        in_pix_color_basis = np.require(in_pix_color_basis, dtype=np.single, requirements='C')
        enh_pix_colors = np.require(enh_pix_colors, dtype=np.single, requirements='C')
        if self.use_local_context_ftr:
            pixContextSemFtr = np.require(pixContextSemFtr, dtype=np.single, requirements='C')                      
            if self.use_local_context_color_ftr:
                pix_cont_color_ftr = np.require(pix_cont_color_ftr, dtype=np.single, requirements='C')
                pix_color_hist_ftr = np.require(pix_color_hist_ftr, dtype=np.single, requirements='C')
            
        elapsed5 = time.time() - st_time
#         print 'MITfivekDataProvider_4 get_next_batch elapsed3,elapsed4,elapsed5:%f %f %f'\
#         % (elapsed3, elapsed4, elapsed5)
#         print 'shapes: img_global_ftr', img_global_ftr.shape, 'pix_ftr', pix_ftr.shape, \
#         'in_pix_color_basis', in_pix_color_basis.shape, 'enh_pix_colors', enh_pix_colors.shape        
#         if self.use_local_context_ftr:
#             print 'shapes: pixContextSemFtr',pixContextSemFtr.shape
        
        if self.use_local_context_ftr:
            if self.use_local_context_color_ftr:
                return epoch, batch_idx, batch_num, epoc_batch_perm, \
                [img_global_ftr, pix_ftr, in_pix_color_basis, enh_pix_colors, pixContextSemFtr, \
                 pix_color_hist_ftr, pix_cont_color_ftr]                        
            else:
                return epoch, batch_idx, batch_num, epoc_batch_perm, \
                [img_global_ftr, pix_ftr, in_pix_color_basis, enh_pix_colors, pixContextSemFtr]
                        
        else:
            return epoch, batch_idx, batch_num, epoc_batch_perm, \
                    [img_global_ftr, pix_ftr, in_pix_color_basis, enh_pix_colors]                   
                             
    def get_data_dims(self, idx=0):
        lc_paras = self.local_context_paras
        if self.use_local_context_color_ftr:
            lc_color_paras = self.local_context_color_paras
            
        if idx == self.IMG_GLOBAL_FTR_ID:
            for key in self.img_global_ftr.keys():
                return self.img_global_ftr[key].shape[0]
        elif idx == self.CTR_PIX_FTR_ID:
            if not hasattr(self, 'patch_img_size'):
                if CENTRAL_PX_FEATURE_DIM < 4:
                    return CENTRAL_PX_FEATURE_DIM
                else:
                    return next_4_multiple(CENTRAL_PX_FEATURE_DIM)
            else:
                return self.batch_meta['patch_num_vis']                    
        elif idx == self.COLOR_BASIS_ID:
            return 10
        elif idx == self.ENH_COLOR_ID:
            return 1 if self.regress_L_channel_only else 3
        elif self.use_local_context_ftr:
            if idx == self.PIX_CONTEXT_SEM_FTR_ID:
                return lc_paras['label_num'] * lc_paras['pool_region_num'] 
            else:
                if idx == self.CONTEXTCOLOR_FTR:
                    return lc_paras['pool_region_num'] * 3
                elif idx == self.PIX_COLOR_HIST_FTR:
                    return lc_color_paras['hist_bin_num'] * 2
                else:
                    raise DataProviderException("data index can not exceed %d" % (self.CONTEXTCOLOR_FTR + 1))                            
        else:
            raise DataProviderException("data index can not exceed %d" % (self.PIX_CONTEXT_SEM_FTR_ID + 1))
                
                
class MITfivekColorRegressionDataProvider(LabeledMemoryBatchDataProvider):    
    def __init__(self, libModel, data_dir, batch_range, init_epoch=1,
                 init_batch_idx=None, epochBatchPerm=None, dp_params={}, test=False):
        LabeledMemoryBatchDataProvider.__init__(self, libModel, data_dir, batch_range, init_epoch,
                 init_batch_idx, epochBatchPerm, dp_params, test)
        self.regress_L_channel_only = dp_params['regress_L_channel_only']
        self.use_local_context_ftr = dp_params['use_local_context_ftr']
        ''' flag if we compute 
            1) mean colors for each of 25 contextual pooling regions
            2) color histogram in a,b channels for a local window centered at the segment
        '''          
        self.use_local_context_color_ftr = dp_params['use_local_context_color_ftr']

        self.IMG_GLOBAL_FTR_ID = 0
        self.CTR_PIX_FTR_ID = 1
        self.ENH_COLOR_ID = 2
        if self.use_local_context_ftr:
            self.PIX_CONTEXT_SEM_FTR_ID = 3
            self.CONTEXTCOLOR_FTR = 4
            self.PIX_COLOR_HIST_FTR = 5         
        
        self.local_context_paras = self.batch_meta['local_context_paras']
        if self.use_local_context_ftr:
            self.pixContextSemMean = self.batch_meta['pixContextSemMean']
            if self.use_local_context_color_ftr:
                self.pixContextColorMean = self.batch_meta['pixContextColorMean']
                self.pixColorHistMean = self.batch_meta['pixColorHistMean']
        
        self.img_global_ftr = {}
        for img in self.batch_meta['imgs']:
            globalFtrPath=os.path.join(self.batch_meta['in_img_global_ftr_dir'],img)
            globalFtr=unpickle(globalFtrPath)['pix_global_ftr']
            self.img_global_ftr[img]=globalFtr
            
        self.img_size = self.batch_meta['img_size']  
        self.num_colors = self.batch_meta['num_colors']        
        print '--------complete MITfivekColorRegressionDataProvider init-------'
        
    def get_img_size_num_colors(self):
        return self.img_size, self.num_colors        
        
        
    def get_batches_meta(self):
        return self.batch_meta
        
    '''in testing stage, prepare batch data to feed into NN'''
    def prepare_batch_data(self, patches, aux_data):
        new_setting = 1
        '''shape: pix_ftr_dim * n'''
        in_pix_ftr = patches
        assert len(patches.shape) == 2
        num = patches.shape[1]

        if self.use_local_context_ftr:
            if self.use_local_context_color_ftr:
                in_img_nms, gt_color, pixContextSemFtr, pixContextColorFtr, pixContextPixNumFtr = \
                aux_data[0], aux_data[1], aux_data[2], aux_data[3], aux_data[4]
            else:
                in_img_nms, gt_color, pixContextSemFtr = \
                aux_data[0], aux_data[1], aux_data[2]                    
        else:
            in_img_nms, gt_color = aux_data[0], aux_data[1]
        assert num == len(in_img_nms)
        
        st_time = time.time()

        img_global_ftr_dim = self.get_data_dims(self.IMG_GLOBAL_FTR_ID)
        
        img_global_ftr = np.tile(self.img_global_ftr[in_img_nms[0]].reshape((img_global_ftr_dim, 1)), (1, num))        

        elapsed_1 = time.time() - st_time
        pix_ftr = in_pix_ftr - self.data_mean[:, np.newaxis]
        
        if self.regress_L_channel_only:
            gt_color = gt_color[0, :].reshape((1, num))
        
        if self.use_local_context_ftr:
            pixContextSemFtr = pixContextSemFtr - self.pixContextSemMean[:, np.newaxis]
            if self.use_local_context_color_ftr:
                pixContextColorFtr = pixContextColorFtr - self.pixContextColorMean[:, np.newaxis]
                pixContextPixNumFtr = pixContextPixNumFtr - self.pixColorHistMean[:, np.newaxis]
        
        if new_setting == 1:
            img_global_ftr *= 100        
            if self.use_local_context_ftr:
                pixContextSemFtr *= 1e2
                if self.use_local_context_color_ftr:
                    pixContextColorFtr *= 4e-4
                    pixContextPixNumFtr *= 4e-2
            pix_ftr *= 1
                            
        st_time = time.time()
        img_global_ftr = np.require(img_global_ftr, dtype=np.single, requirements='C')
        pix_ftr = np.require(pix_ftr, dtype=np.single, requirements='C')
        gt_color = np.require(gt_color, dtype=np.single, requirements='C')
        if self.use_local_context_ftr:
            pixContextSemFtr = np.require(pixContextSemFtr, dtype=np.single, requirements='C')                      
            if self.use_local_context_color_ftr:
                pixContextColorFtr = np.require(pixContextColorFtr, dtype=np.single, requirements='C')
                pixContextPixNumFtr = np.require(pixContextPixNumFtr, dtype=np.single, requirements='C')

        
        elapsed_2 = time.time() - st_time
        if self.use_local_context_ftr:
            if self.use_local_context_color_ftr:
                return [img_global_ftr, pix_ftr, gt_color, pixContextSemFtr, \
                        pixContextColorFtr, pixContextPixNumFtr], \
                    [elapsed_1, elapsed_2]                        
            else:
                return [img_global_ftr, pix_ftr, gt_color, pixContextSemFtr], \
                    [elapsed_1, elapsed_2]
        else:
            return [img_global_ftr, pix_ftr, gt_color], [elapsed_1, elapsed_2]
        
    def load_batch_data(self, zero_mean_data=0):
        LabeledMemoryBatchDataProvider.load_batch_data(self, zero_mean_data)

    def get_next_batch(self):
        new_setting = 1
        epoch, batch_idx, batch_num, epoc_batch_perm, curr_batchdat_dic = \
        LabeledMemoryBatchDataProvider.get_next_batch(self)

        st_time = time.time()
        num_sams = curr_batchdat_dic['data'].shape[1]        
        ''' in_pix_colors shape: (num_sams,segment_random_sample_num,3) '''
        in_pix_colors = curr_batchdat_dic['in_pix_data']
        segment_random_sample_num = in_pix_colors.shape[1]
        assert num_sams == in_pix_colors.shape[0]        
        
        img_global_ftr_dim = self.get_data_dims(self.IMG_GLOBAL_FTR_ID)
        img_global_ftr = np.zeros((num_sams,img_global_ftr_dim),dtype=np.single)
        assert num_sams == len(curr_batchdat_dic['patch_to_image_name'])
        for i in range(num_sams):
            img_global_ftr[i,:]=self.img_global_ftr[curr_batchdat_dic['patch_to_image_name'][i]]
                    
        elapsed3 = time.time() - st_time
        
        st_time = time.time()
        
        
        if self.use_local_context_ftr:
            ''' shape: (dim, num_sams) '''
            pixContextSemFtr = curr_batchdat_dic['pixContextSemFtr'] - self.pixContextSemMean[:, np.newaxis]
            pixContextSemFtr = np.repeat(pixContextSemFtr, segment_random_sample_num, axis=1)
            if self.use_local_context_color_ftr:
                pixContextColorFtr = curr_batchdat_dic['pixContextColor'] - self.pixContextColorMean[:, np.newaxis]
                pixContextColorFtr = np.repeat(pixContextColorFtr, segment_random_sample_num, axis=1)
                pixContextPixNumFtr = curr_batchdat_dic['pixContextPixNum'] - self.pixColorHistMean[:, np.newaxis]
                pixContextPixNumFtr = np.repeat(pixContextPixNumFtr,segment_random_sample_num, axis=1)
                
        '''pix_ftr shape:(3,num_sams,segment_random_sample_num)'''
        pix_ftr = in_pix_colors.swapaxes(0,2).swapaxes(1,2)
        pix_ftr = pix_ftr.reshape((3,num_sams*segment_random_sample_num)) - self.data_mean[:,np.newaxis]
        
        ''' enh_pix_colors shape: (segment_random_sample_num * 3,num_sams) '''
        enh_pix_colors = curr_batchdat_dic['labels']
        assert enh_pix_colors.shape[0] / 3 == segment_random_sample_num
        assert enh_pix_colors.shape[1] == num_sams
        enh_pix_colors = enh_pix_colors.reshape((segment_random_sample_num, 3, num_sams))
        if self.regress_L_channel_only:
            enh_pix_colors = enh_pix_colors[:, 0, :].reshape((segment_random_sample_num, 1, num_sams))
        enh_ch = 1 if self.regress_L_channel_only else 3
        '''enh_pix_colors shape:(enh_ch,num_sams,segment_random_sample_num) '''
        enh_pix_colors = enh_pix_colors.swapaxes(0,1).swapaxes(1,2)
        enh_pix_colors = enh_pix_colors.reshape((enh_ch,num_sams*segment_random_sample_num))
        
        elapsed4 = time.time() - st_time
        
        if new_setting == 1:
            img_global_ftr *= 100        
            if self.use_local_context_ftr:
                pixContextSemFtr *= 1e2
                if self.use_local_context_color_ftr:
                    pixContextColorFtr *= 4e-4
                    pixContextPixNumFtr *= 4e-2
            pix_ftr *= 1
        
        rand_perm = range(num_sams*segment_random_sample_num)
        random.shuffle(rand_perm)
     
        st_time = time.time()  
        img_global_ftr = img_global_ftr[rand_perm,:]
        img_global_ftr = np.require(img_global_ftr.transpose(), dtype=np.single, requirements='C')
        if self.use_local_context_ftr:
            pixContextSemFtr = pixContextSemFtr[:,rand_perm]
            pixContextSemFtr = np.require(pixContextSemFtr, dtype=np.single, requirements='C')                      
            if self.use_local_context_color_ftr:
                pixContextColorFtr = pixContextColorFtr[:, rand_perm]
                pixContextColorFtr = np.require(pixContextColorFtr, dtype=np.single, requirements='C')
                pixContextPixNumFtr = pixContextPixNumFtr[:,rand_perm]
                pixContextPixNumFtr = np.require(pixContextPixNumFtr, dtype=np.single, requirements='C')    
        pix_ftr=pix_ftr[:,rand_perm]    
        pix_ftr = np.require(pix_ftr, dtype=np.single, requirements='C')              
        enh_pix_colors=enh_pix_colors[:,rand_perm]
        enh_pix_colors = np.require(enh_pix_colors, dtype=np.single, requirements='C')

            
        elapsed5 = time.time() - st_time
#         print 'MITfivekColorRegressionDataProvider get_next_batch elapsed3,elapsed4,elapsed5:%f %f %f'\
#         % (elapsed3, elapsed4, elapsed5)
#         print 'shapes: img_global_ftr', img_global_ftr.shape, 'pix_ftr', pix_ftr.shape, \
#         'enh_pix_colors', enh_pix_colors.shape        
#         if self.use_local_context_ftr:
#             print 'shapes: pixContextSemFtr',pixContextSemFtr.shape

        if self.use_local_context_ftr:
            if self.use_local_context_color_ftr:
                return epoch, batch_idx, batch_num, epoc_batch_perm, \
                [img_global_ftr, pix_ftr, enh_pix_colors, pixContextSemFtr, \
                 pixContextColorFtr, pixContextPixNumFtr]                        
            else:
                return epoch, batch_idx, batch_num, epoc_batch_perm, \
                [img_global_ftr, pix_ftr, enh_pix_colors, pixContextSemFtr]
                        
        else:
            return epoch, batch_idx, batch_num, epoc_batch_perm, \
                [img_global_ftr, pix_ftr, enh_pix_colors]                   
                             
    def get_data_dims(self, idx=0):
        lc_paras = self.local_context_paras
        if idx == self.IMG_GLOBAL_FTR_ID:
            for key in self.img_global_ftr.keys():
                return self.img_global_ftr[key].shape[0]            
        elif idx == self.CTR_PIX_FTR_ID:
            if CENTRAL_PX_FEATURE_DIM < 4:
                return CENTRAL_PX_FEATURE_DIM
            else:
                return next_4_multiple(CENTRAL_PX_FEATURE_DIM)                   
        elif idx == self.ENH_COLOR_ID:
            return 1 if self.regress_L_channel_only else 3
        elif self.use_local_context_ftr:
            if idx == self.PIX_CONTEXT_SEM_FTR_ID:
                return lc_paras['label_num'] * lc_paras['pool_region_num'] 
            else:
                if idx == self.CONTEXTCOLOR_FTR:
                    return lc_paras['pool_region_num'] * 3
                elif idx == self.PIX_COLOR_HIST_FTR:
                    return lc_paras['pool_region_num']
                else:
                    raise DataProviderException("data index can not exceed %d" % (self.CONTEXTCOLOR_FTR + 1))                            
        else:
            raise DataProviderException("data index can not exceed %d" % (self.PIX_CONTEXT_SEM_FTR_ID + 1))
                            
'''             
class MITfivekDataProvider_gradmag(MITfivekDataProvider_4):
    def __init__(self, libModel, data_dir, batch_range, init_epoch=1,
                 init_batch_idx=None, epochBatchPerm=None, dp_params={}, test=False):
        MITfivekDataProvider_4.__init__\
        (self, libModel, data_dir, batch_range, init_epoch, init_batch_idx, epochBatchPerm, dp_params, test)
    
        self.IMG_GLOBAL_FTR_ID = 0
        self.CTR_PIX_FTR_ID = 1
        self.PIX_CONTEXT_SEM_FTR_ID = 2
        self.PIX_L_ID = 3
        self.PIX_IN_GRADMAG_ID = 4
        self.PIX_ENH_GRADMAG_ID = 5        
    
    def prepare_batch_data(self, pix_L, aux_data):
        # pix_L shape: (1,num)
        # in_pix_gradmag shape: (1,num)
        # pix_local_context shape: (ftr_dim,num)
        in_img_ids, in_pix_gradmag, pix_local_context = aux_data[0], aux_data[1], aux_data[2]
        num = pix_L.shape[1]
                
        img_global_ftr_dim = self.get_data_dims(self.IMG_GLOBAL_FTR_ID)
        img_global_ftr = np.tile(self.img_global_ftr[in_img_ids[0], :].reshape\
                                 ((img_global_ftr_dim, 1)), (1, num))
        img_global_ftr *= 100
        
        pix_ftr = pix_L - self.data_mean[0, np.newaxis]
        pix_local_context -= self.pixContextSemMean[:, np.newaxis]
        pix_local_context *= 100

        dummy_enh_pix_gradmag = np.zeros((1, num), dtype=np.single)
        
        idx = np.nonzero(pix_L == 0)
        pix_L[idx[0], idx[1]] = 1
                
        img_global_ftr = np.require(img_global_ftr, dtype=np.single, requirements='C')
        pix_ftr = np.require(pix_ftr, dtype=np.single, requirements='C')
        pix_local_context = np.require(pix_local_context, dtype=np.single, requirements='C')
        pix_L = np.require(pix_L, dtype=np.single, requirements="C")
        in_pix_gradmag = np.require(in_pix_gradmag, dtype=np.single, requirements='C')
        dummy_enh_pix_gradmag = np.require(dummy_enh_pix_gradmag, dtype=np.single, requirements='C')
        
        return [img_global_ftr, pix_ftr, pix_local_context, \
                pix_L, in_pix_gradmag, dummy_enh_pix_gradmag]
    
    def get_next_batch(self):
        print 'MITfivekDataProvider_gradmag get_next_batch'
        epoch, batch_idx, batchnum, epochBatchPerm, curr_batchdat_dic = \
        LabeledMemoryBatchDataProvider.get_next_batch(self)        
        
        num_pixs = curr_batchdat_dic['data'].shape[1]
        print 'MITfivekDataProvider_edge get_next_batch num_pixs:%d' % num_pixs
        
        img_global_ftr = self.img_global_ftr[curr_batchdat_dic['pix_to_imageID'], :]
        img_global_ftr *= 100
        # only keep L channel
        pix_ftr = (curr_batchdat_dic['data'] - self.data_mean[:, np.newaxis])[0, :].reshape((1, num_pixs))
        pix_local_context = curr_batchdat_dic['pix_local_context']\
        - self.pixContextSemMean[:, np.newaxis]
        pix_local_context *= 100
         
         
        pix_L = curr_batchdat_dic['data'][0, :].reshape((1, num_pixs)) 
        idx = np.nonzero(pix_L == 0)
        pix_L[idx[0], idx[1]] = 1
           
        in_pix_gradmag = curr_batchdat_dic['in_pix_gradmag']
        enh_pix_gradmag = curr_batchdat_dic['labels']

        
        img_global_ftr = np.require(img_global_ftr.transpose(), dtype=np.single, requirements='C')
        pix_ftr = np.require(pix_ftr, dtype=np.single, requirements='C')
        pix_local_context = np.require(pix_local_context, dtype=np.single, requirements='C')
        pix_L = np.require(pix_L, dtype=np.single, requirements='C')        
        in_pix_gradmag = np.require(in_pix_gradmag, dtype=np.single, requirements='C')
        enh_pix_gradmag = np.require(enh_pix_gradmag, dtype=np.single, requirements='C')
#         print 'img_global_ftr,pix_ftr,in_pix_gradmag,enh_pix_gradmag,pix_local_context,shape',\
#         img_global_ftr.shape,pix_ftr.shape,\
#         in_pix_gradmag.shape,enh_pix_gradmag.shape,pix_local_context.shape
        
        return epoch, batch_idx, batchnum, epochBatchPerm, \
            [img_global_ftr, pix_ftr, pix_local_context, pix_L, in_pix_gradmag, enh_pix_gradmag]
    
    def get_data_dims(self, idx=0):
        if idx == self.IMG_GLOBAL_FTR_ID:
            return self.img_global_ftr.shape[1]
        elif idx == self.CTR_PIX_FTR_ID:
            return 1                
        elif idx == self.PIX_CONTEXT_SEM_FTR_ID:
            lc_paras = self.local_context_paras
            return lc_paras['ftr_dim']
        elif idx == self.PIX_L_ID:
            return 1          
        elif idx == self.PIX_IN_GRADMAG_ID:
            return 1
        elif idx == self.PIX_ENH_GRADMAG_ID:
            return 1
        else:
            raise DataProviderException("data index can not exceed %d" % (self.PIX_ENH_GRADMAG_ID + 1))
'''
    
class TestDataProvider(LabeledMemoryBatchDataProvider):
    def __init__(self, libModel, data_dir, batch_range, init_epoch=1, init_batch_idx=None, epochBatchPerm=None, dp_params={}, test=False):
        LabeledMemoryBatchDataProvider.__init__(self, libModel, data_dir, batch_range, init_epoch,
                                     init_batch_idx, epochBatchPerm, dp_params, test)
        self.ftr_dim = self.batch_meta['ftr_dim']
        self.label_dim = self.batch_meta['label_dim']
        
    def load_batch_data(self):
        epoch, batchnum = self.curr_epoch, self.curr_batchnum
        self.curr_bat_dat_dic = unpickle(self.get_data_file_name(batchnum))
        self.curr_bat_dat_dic['data'] = np.require(self.curr_bat_dat_dic['data'], \
                                                  dtype=np.single, requirements='C')
        self.curr_bat_dat_dic['labels'] = np.require(self.curr_bat_dat_dic['labels'], \
                                                    dtype=np.single, requirements='C')
        print 'label_dim', self.label_dim


    def get_next_batch(self):
        epoch, batch_idx, batchnum, epochBatchPerm, curr_batchdat_dic = \
        LabeledMemoryBatchDataProvider.get_next_batch(self)
        return epoch, batch_idx, batchnum, epochBatchPerm, \
            [curr_batchdat_dic['data'], curr_batchdat_dic['labels']]
            
    def get_data_dims(self, idx=0):
        # three inputs: patch pxiel, multi-dim label and pixel local features
        if idx == 0:
            return self.ftr_dim
        else:
            return self.label_dim


class SaliencyDataProvider(LabeledDataProvider):
    GENERIC_FTR_DIM = 492
    LSK_FTR_DIM = 30
        
    def __init__(self, libModel, data_dir, batch_range, init_epoch=1,
                 init_batch_idx=None, epochBatchPerm=None, dp_params={}, test=False):
        LabeledDataProvider.__init__(self, libModel, data_dir, batch_range, init_epoch,
                 init_batch_idx, epochBatchPerm, dp_params, test)
        self.curr_bat_dat_dic = []
        ''' shape: (d,n) '''
        self.img_cnn_ftr = self.batch_meta['imgCnnFtr'].transpose()
        
    def load_batch_data(self):
        st = time.time()
        self.curr_bat_dat_dic = unpickle(self.get_data_file_name(self.curr_batchnum))        
        ep0 = time.time() - st
        print 'unpickle batch data time:%4.2f' % ep0
        
        nSmp = self.curr_bat_dat_dic['imgID'].shape[0]
        print 'load_batch_data: %d pixel samples' % nSmp
        
        if not self.test == DataProvider.DP_PREDICT:
            if len(self.curr_bat_dat_dic['labels'].shape) == 1:
                self.curr_bat_dat_dic['labels'] = self.curr_bat_dat_dic['labels'].reshape((1, nSmp))
            self.curr_bat_dat_dic['labels'] = np.require(self.curr_bat_dat_dic['labels'], dtype=np.single, \
                                                       requirements='C')
        else:
            dummy_labels = np.zeros((1, nSmp), dtype=np.single)
            self.curr_bat_dat_dic['labels'] = np.require(dummy_labels, dtype=np.single, requirements='C')
        
        st = time.time()
        self.curr_img_cnn_ftr = self.img_cnn_ftr[:, self.curr_bat_dat_dic['imgID']]
        self.curr_img_cnn_ftr *= 1e+3
        ep1 = time.time() - st
        
        st = time.time()
        self.curr_gen_ftr = self.curr_bat_dat_dic['genericFtr'] - self.batch_meta['genFtrMean'][:, np.newaxis]
        self.curr_gen_ftr *= 1e-1
        ep2 = time.time() - st
        
        st = time.time()
        self.curr_lsk_ftr = self.curr_bat_dat_dic['lskFtr'] - self.batch_meta['lskFtrMean'][:, np.newaxis]
        self.curr_lsk_ftr *= 1
        ep3 = time.time() - st
        
        st = time.time()
        self.curr_img_cnn_ftr = np.require(self.curr_img_cnn_ftr, dtype=np.single, requirements='C')
        self.curr_gen_ftr = np.require(self.curr_gen_ftr, dtype=np.single, requirements='C')
        self.curr_lsk_ftr = np.require(self.curr_lsk_ftr, dtype=np.single, requirements='C')         
        ep4 = time.time()
        
        
    def get_next_batch(self):
        self.load_batch_data()
        
        epoch, batch_idx, batchnum, epochBatchPerm = self.curr_epoch, self.batch_idx, self.curr_batchnum, self.epochBatchPerm
        
        self.advance_batch()        

        return epoch, batch_idx, batchnum, epochBatchPerm, \
            [self.curr_img_cnn_ftr, self.curr_gen_ftr, self.curr_lsk_ftr, \
             self.curr_bat_dat_dic['labels']]

    def get_data_dims(self, idx=0):
        if idx == 0:
            return self.img_cnn_ftr.shape[0]
        elif idx == 1:
            return self.GENERIC_FTR_DIM
        elif idx == 2:
            return self.LSK_FTR_DIM
        else:
            ''' dimensionality of label'''
            return 1
        
        
