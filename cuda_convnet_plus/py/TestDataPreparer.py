# Author: Zhicheng

import os
import numpy as np
from scipy import misc
import exceptions as ecp
import random as rd
import fnmatch
import sys
import time

sys.path.append('D:\yzc\Dropbox\private\proj\cuda-convnet-read-only');
from options import *
from util import *

class TestDataPreparer:
    def __init__(self, op):        
        for o in op.get_options_list():
            setattr(self, o.name, o.value)
        
        print "test-dir:%s" % self.test_dir
            
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
        
        imgH = self.img_height
        imgW = self.img_width
        imgChannel = self.img_channel
        batch_num = self.batch_num 
        
        pixelNum=imgH * imgW
        imgSize=pixelNum*imgChannel        

        print "test data pattern: %s" % self.test_pattern
        if self.test_img_list:
            print 'test image list file: %s' % self.test_img_list
            list_file = open(self.test_img_list,'r')
            img_files = list_file.readlines()
            list_file.close()
            
            img_files =[f[:-1] if f[-1]=='\n' else f for f in img_files]
            img_files = [os.path.join(self.test_dir,f) for f in img_files]
        else:
            img_files=[]
            for dirname,subdirnames,filenames in os.walk(self.test_dir):
                img_files += [os.path.join(dirname,f) for f in filenames if fnmatch.fnmatch(f, self.test_pattern) ]
            
#             img_files = [f for f in os.listdir(self.test_dir) if os.path.isfile(os.path.join(self.test_dir, f)) and fnmatch.fnmatch(f, self.test_pattern)]
            list_file_path = os.path.join(self.test_dir,'img_list.txt')
            list_file=open(list_file_path,'w')
            # write relative path
            for f in img_files:
                list_file.write(f[len(self.test_dir)+1:]+'\n')
#             list_file.writelines(img_files)
            list_file.close()
            print 'find %d images' % len(img_files)
            
        print img_files[:10]
        # img_files=[f for f in os.listdir(self.test_dir) if os.path.isfile(os.path.join(self.test_dir,f))]
        test_meta={}
        test_meta['img_files']=img_files
        if self.test_label_file:
            pickle(os.path.join(self.output_dir, 'val_batches.meta'), test_meta, True)
        else:
            pickle(os.path.join(self.output_dir,'pred_batches.meta'),test_meta,True)
        imgN = len(img_files)
        print "find %d images" % imgN
        
        if self.test_label_file:
            labelFile = open(self.test_label_file, "r")
        imgIdx = 0

        
        for i in range(batch_num):
            bstart = time.clock()
            if i < (batch_num - 1):
                batchSize = imgN / batch_num
            else:
                batchSize = imgN - (batch_num - 1) * (imgN / batch_num)
            print "%d th batch out of %d batches has %d images" % (i, batch_num, batchSize)
            
            # On exit, batchData has shape (imgSize, batchSize)
            batchData = np.zeros((imgChannel, imgH, imgW, batchSize), dtype=np.uint8)
            batchDataView= np.swapaxes(batchData,0,3)
            if self.test_label_file:
                batchLabel = np.zeros((batchSize), dtype=np.float32)
            for j in range(batchSize):
                if j % 1000 == 0:
                    print "progress %3.2f" % (float(100 * j) / float(batchSize))
                imgFile = img_files[imgIdx]
                imgFile = os.path.join(self.test_dir, imgFile)
                try:
                    im = misc.imread(imgFile)
                except Exception, e:
                    print "fail to read image: %s" % imgFile
                    print e
                    sys.exit()
                           
                # typically, im.shape=(imgH,imgW,imgChannel)
                if im.shape[0] != imgH or im.shape[1] != imgW:
                    print 'im shape'
                    print im.shape
                    print 'expected image shape: (height,width)=(%d,%d)' % (imgH,imgW)
                    raise ecp.NameError('incorrect image shape')
                if im.shape[2]==3:
                    if imgChannel != 3:
                        raise ecp.NameError('incorrect image color channels')
                    batchDataView[j][:][:][:]=im
                elif im.shape[2] == 1:
                    print "a grey image"
                    for k in range(imgChannel):
                        batchDataView[j][:][:][k]=im                           
                else:
                    raise ecp.NameError('non-RGB non-L images are not handled')    
                
                if self.test_label_file:
                    batchLabel[j] = float(labelFile.readline()) - 1 # label ranges from 0 to num_classes-1        
                imgIdx += 1
            
            batchData=np.reshape(batchData, (imgSize,batchSize))
            batch_dic = {}
            batch_dic['data'] = batchData
            if self.test_label_file:
                batch_dic['labels'] = batchLabel
            print "pickle batch data"
            batchFilePath = os.path.join(self.output_dir, self.batch_fname + '_' + 
                                       str(self.batch_start_index + i))
            print "batchFilePath: %s" % batchFilePath
            pickle(batchFilePath, batch_dic,True)    
            bend = time.clock()
            print "batch elapsed time %f" % (bend - bstart)
        if self.test_label_file:
            labelFile.close()
        
        
#         testData = np.zeros((imgChannel, imgH, imgW, imgN), dtype=np.uint8)
#         testLabel = np.zeros((imgN), dtype=np.float32)
#            
#         labelFile = open(self.test_label_file, "r")
#         
#         for i in range(imgN):
#             if i % 200 == 0:
#                 print "progress %5.4f" % (float(i) / float(imgN))
#                 
#             imgFile = img_files[i]
#             imgFile = os.path.join(self.test_dir, imgFile)
#             im = Image.open(imgFile)
#             imgW2 = im.size[0]
#             imgH2 = im.size[1]
#             if imgW2 != imgW or imgH2 != imgH:
#                 raise ecp.NameError('incorrect image size')      
#             if im.mode == "RGB":
#                 if imgChannel != 3:
#                     raise ecp.NameError('incorrect image color channels')
#                 pixels = list(im.getdata())
#                 offset = 0
#                 for yi in range(imgH):
#                     for xi in range(imgW):
#                         pixel = pixels[offset]
#                         testData[0][yi][xi][i] = pixel[0]
#                         testData[1][yi][xi][i] = pixel[1]
#                         testData[2][yi][xi][i] = pixel[2]
#                         offset += 1
#             else:
#                 raise ecp.NameError('non-RGB images are not handled')  
#             
#             testLabel[i] = float(labelFile.readline())
#             
#         labelFile.close()
#         test_dic = {}
#         test_dic['data'] = testData
#         test_dic['label'] = testLabel
#         print "pickle data"
#         pickle(self.output_data_path, test_dic)
    @staticmethod
    def get_options_parser():
        op = OptionsParser()
        op.add_option("test-dir", "test_dir", StringOptionParser,
                      "Directory of test images", default="")
        op.add_option("test-pattern", "test_pattern", StringOptionParser,
                      "test image name pattern", default="*")
        op.add_option("test-img-list", "test_img_list", StringOptionParser,
                      "the list file of test images", default="")        
        op.add_option("output-dir", "output_dir", StringOptionParser,
                      "directory of output data batch files", default="")
        op.add_option("batch-start-index", "batch_start_index", IntegerOptionParser,
                      "batch starting index", default=1)
        op.add_option("batch-num", "batch_num", IntegerOptionParser,
                      "number of batch files", default=4)
        op.add_option("batch-fname", "batch_fname", StringOptionParser,
                      "file name template of batch files", default="batch_data")
        # if --test-label-file is not given, then the batch file does not include label.
        op.add_option("test-label-file", "test_label_file", StringOptionParser,
                      "Test images label file", default="")
        op.add_option("img-width", "img_width", IntegerOptionParser,
                      "image width", default=256)
        op.add_option("img-height", "img_height", IntegerOptionParser,
                      "image height", default=256)
        op.add_option("img-channel", "img_channel", IntegerOptionParser,
                      "image channel", default=3) 
        return op          
if __name__ == "__main__":
    op = TestDataPreparer.get_options_parser()
    op = TestDataPreparer.parse_option(op)
    tsDataPreparer = TestDataPreparer(op)
    tsDataPreparer.save_data()     
                 
