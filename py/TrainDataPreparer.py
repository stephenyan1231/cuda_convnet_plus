# Author: Zhicheng

import os
import numpy as np
import random as rd
from scipy import misc
import exceptions as ecp
import scipy.io
import fnmatch
import sys
import time
from PCA import *
from util import *
from options import *

class TrainDataPreparer:
    def __init__(self, op):
        for o in op.get_options_list():
            setattr(self, o.name, o.value)
        
        
    @staticmethod
    def get_options_parser():
        op = OptionsParser()
        op.add_option("train-dir", "train_dir", StringOptionParser,
                      "Directory of train images", default="")
        op.add_option("train-pattern", "train_pattern", StringOptionParser,
                      "training image name pattern", default="*")
        op.add_option("output-dir", "output_dir", StringOptionParser,
                      "directory of output data batch files", default="")
        op.add_option("batch-start-index", "batch_start_index", IntegerOptionParser,
                      "batch starting index", default=1)
        op.add_option("batch-num", "batch_num", IntegerOptionParser,
                      "number of batch files", default=6)
        op.add_option("batch-fname", "batch_fname", StringOptionParser,
                      "file name template of batch files", default="data_batch")
        op.add_option("input-matlab-meta-data-file", "input_matlab_meta_data_file", StringOptionParser,
                      "input matlab meta data file name", default="")        
#         op.add_option("input-matlab-pixelPCA-file", "input_matlab_pixelPCA_file", StringOptionParser,
#                       "input matlab data file with pixel-wise PCA", default="")               
        op.add_option("output-meta-data-file", "output_meta_data_file", StringOptionParser,
                      "output meta data file name", default="meta")
        op.add_option("img-width", "img_width", IntegerOptionParser,
                      "image width", default=256)
        op.add_option("img-height", "img_height", IntegerOptionParser,
                      "image height", default=256)
        op.add_option("img-channel", "img_channel", IntegerOptionParser,
                      "image channel", default=3)
        op.add_option("PCA-img-num", "PCA_img_num", IntegerOptionParser,
                      "number of subsampled images for pixel-wise PCA", default=60000)
                
        return op      
      
    @staticmethod
    def parse_option(op):
        try:
            options = op.parse()
            op.eval_expr_defaults()
            return op
        except OptionMissingException, e:
            print e
            op.print_usage()
        except OptionException, e:
            print e
        sys.exit() 
        
                 
    def save_data(self):
        if os.path.exists(self.output_dir) == False:
            os.mkdir(self.output_dir)
        
        # generate data files and meta data file
        batch_num = self.batch_num
        batch_fname = self.batch_fname
        img_height = self.img_height
        img_width = self.img_width
        img_channel = self.img_channel
        pixelNum = img_height * img_width
        imgSize = pixelNum * img_channel       
        
        matlab_meta = scipy.io.loadmat(self.input_matlab_meta_data_file)
        synsets = matlab_meta['synsets']
#         costMat = matlab_meta['cost_matrix']  
        synsetsN = len(synsets)      
        print "synsets length:%d" % len(synsets)
        
        
        wnID_2_ILSVRSID = {};
        label_2_labelname = [None] * len(synsets)
        for i in range(len(synsets)):
            wnID_2_ILSVRSID[str(synsets[i][0][1][0])] = synsets[i][0][0][0][0]
            label_2_labelname[i] = str(synsets[i][0][2][0])
        # print "wnID_2_ILSVRSID"
        # print wnID_2_ILSVRSID
        
        wnID = [f for f in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, f))]
        wnNum = len(wnID)
        
        classSz = np.zeros((wnNum), dtype=np.uint32)  # classes at leaf nodes of semantic tree
        wnID2label = {}
        for i in range(wnNum):
            label = -1;
            for j in range(synsetsN):
                if str(synsets[j][0][1][0]) == wnID[i]:
                    label = j  # label starts from 0
                    break
            if label < 0:
                print "wnID %s is not found in synsets " % wnID[i]
                raise ecp.NameError('error')
            wnID2label[wnID[i]] = label
            classSz[i] = synsets[label][0][7][0][0]      
        
        totalImgN = sum(classSz)
        print 'totally %d images' % totalImgN
        
        # need to sample a subset of images because main memory can not hold all pixels in training images
        PCA_img_num = self.PCA_img_num
        if PCA_img_num > totalImgN:
            PCA_img_num = totalImgN
        print "use %d images to compute pixelwise PCA" % PCA_img_num
        
        imgID2Name = [None] * totalImgN
        imgID2Label = np.zeros((totalImgN), dtype=np.float32)
        classMember = [None] * wnNum
               
        num_image_correct = 1
        imgIdx = 0
        maxLabel = -1
        for i in range(wnNum):
            label = wnID2label[wnID[i]];        
            if label > maxLabel:
                maxLabel = label    
            subDir = wnID[i]
            subDirPath = os.path.join(self.train_dir, subDir)
            imgs = [img for img in os.listdir(subDirPath) if os.path.isfile(os.path.join(subDirPath, img)) and fnmatch.fnmatch(img, self.train_pattern)]
            if classSz[i] != len(imgs):
                print "wnID:%s %d images in folder and %d images from synset" % (wnID[i], len(imgs), classSz[i])
                num_image_correct = 0
            lClsMember = [];
            for j in range(len(imgs)):
                imgID2Name[imgIdx] = os.path.join(subDirPath, imgs[j])
                imgID2Label[imgIdx] = label
                lClsMember += [imgIdx]
                imgIdx = imgIdx + 1
            classMember[i] = lClsMember
        print 'max label : %d' % maxLabel
        if num_image_correct == 0:
            raise ecp.NameError('incorrect training images number')
        
        batch_size = np.zeros((wnNum, batch_num), dtype=np.uint32)
        for i in range(wnNum):
            batch_size[i][:] = get_batch_size(classSz[i], batch_num)
        
        # randomly distribute images to batches in a class-wise manner
        for i in range(len(classMember)):
            rd.shuffle(classMember[i])
                   
        # dataMean has shape (img_channel,img_height,img_width)
        dataMean = np.zeros((img_channel, img_height, img_width), dtype=np.float64)
        # dataMeanView has shape (img_height, img_width, img_channel)
        dataMeanView = dataMean.swapaxes(0, 2).swapaxes(0, 1)
      
        for i in range(batch_num):
            batchFileFn = batch_fname + '_' + str(i + self.batch_start_index)
            batchFileFn = os.path.join(self.output_dir, batchFileFn)
            print "%d th batch file name:%s" % (i, batchFileFn)              
            batchImgN = np.sum(batch_size[:, i])
            print "%d images in %d th batch out of %d batches" % (batchImgN, i, batch_num)       
      
        for i in range(batch_num):
            batchFileFn = batch_fname + '_' + str(i + self.batch_start_index)
            batchFileFn = os.path.join(self.output_dir, batchFileFn)
            print "%d th batch file name:%s" % (i, batchFileFn)
            
            bstart = time.clock()                
            batchImgN = np.sum(batch_size[:, i])
            print "%d images in %d th batch out of %d batches" % (batchImgN, i, batch_num)
            # On exit, batchData has shape (imgSize, batchImgN)
            batchData = np.zeros((img_channel, img_height, img_width, batchImgN), dtype=np.uint8)
            # batchDataView shape (batchImgN, img_height, img_width, img_channel)
            batchDataView = np.swapaxes(batchData, 0, 3)
            batchLabel = np.zeros(batchImgN, dtype=np.float32)
            batchImgId = []
#             batchImgIdx = 0
            
            # collect image ids in the batch
            for j in range(wnNum):
                batchClsN = batch_size[j, i]
                startIdx = np.sum(batch_size[j, 0:i])
                batchImgId += classMember[j][startIdx:startIdx + batchClsN]
            assert len(batchImgId) == batchImgN
            # important! shuffle images in the batch so that mini-batch has random class distribution
            rd.shuffle(batchImgId)
            
            for j in range(batchImgN):
                lImgID = batchImgId[j]
                try:
                    im = misc.imread(imgID2Name[lImgID])
                except Exception, e:
                    print e
                    sys.exit()                     
                if im.shape[1] != img_width or im.shape[0] != img_height:
                    raise ecp.NameError('incorrect image size')  
                if im.shape[2] == 3:
                    if img_channel != 3:
                        raise ecp.NameError('incorrect image color channels')
                    batchDataView[j][:][:][:] = im
                    dataMeanView += im
                elif im.mode == "L":
                    print "a grey image"
                    for p in range(img_channel):
                        batchDataView[j][:][:][p] = im
                        dataMeanView[:][:][p] += im                  
                else:
                    raise ecp.NameError('non-RGB non-gray images are not handled')
                batchLabel[j] = imgID2Label[lImgID]           
                            
#             for j in range(wnNum):
#                 if j % 200 == 0:
#                     print "progress : %5.4f " % (float(100 * j) / float(wnNum))
# #                 if i < batch_num - 1:
# #                     batchClsN = batchSize[j]
# #                 else:
# #                     batchClsN = lastBatchSize[j]
#                 batchClsN = batch_size[j, i]
#                     
#                 # print "batchClsN %d" % batchClsN
#                 startIdx = np.sum(batch_size[j, 0:i])
#                 for k in range(batchClsN):
# #                     lImgID = classMember[j][i * batchSize[j] + k]
#                     lImgID = classMember[j][startIdx + k]
#                     try:
#                         im = misc.imread(imgID2Name[lImgID])
#                     except Exception, e:
#                         print e
#                         sys.exit()                    
#                     # typically, im.shape=(imgH,imgW,imgChannel)
#                     if im.shape[1] != img_width or im.shape[0] != img_height:
#                         raise ecp.NameError('incorrect image size')  
#                     if im.shape[2] == 3:
#                         if img_channel != 3:
#                             raise ecp.NameError('incorrect image color channels')
#                         batchDataView[batchImgIdx][:][:][:] = im
#                         dataMeanView += im
#                     elif im.mode == "L":
#                         print "a grey image"
#                         for p in range(img_channel):
#                             batchDataView[batchImgIdx][:][:][p] = im
#                             dataMeanView[:][:][p] += im                  
#                     else:
#                         raise ecp.NameError('non-RGB non-gray images are not handled')   
#  
#                     batchLabel[batchImgIdx] = imgID2Label[lImgID]
#                     batchImgIdx += 1        
            batchData = np.reshape(batchData, (imgSize, batchImgN))
            
            batch_dic = {}
            batch_dic['data'] = batchData
            batch_dic['labels'] = batchLabel
 
            try:
                pickle(batchFileFn, batch_dic,True);
            except Exception as inst:
                print "exception occurs"
                print type(inst)
                print inst
            bend = time.clock()
            print 'batch elapsed time:%f' % (bend - bstart)
                
            
        dataMean /= totalImgN
        dataMean = dataMean.reshape((imgSize, 1))
        
        
        meta_dic = {}
        meta_dic['data_mean'] = dataMean
        meta_dic['num_vis'] = img_width * img_height * img_channel
        meta_dic['label_names'] = label_2_labelname[0:wnNum]
#         meta_dic['PCA_evecs'] = PCA_evecs
#         meta_dic['PCA_evals'] = PCA_evals
#         meta_dic['PCA_scaled_evecs'] = PCA_scaled_evecs_3
        pickle(self.output_meta_data_file, meta_dic,True)
        
if __name__ == "__main__":
    op = TrainDataPreparer.get_options_parser()
    op = TrainDataPreparer.parse_option(op)
    trDataPreparer = TrainDataPreparer(op)
    trDataPreparer.save_data()
        
            
