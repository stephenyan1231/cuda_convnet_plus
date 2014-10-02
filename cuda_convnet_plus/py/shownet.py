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
#   and/or other materials provided with the distribution.
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

import numpy as np
import sys
import getopt as opt
from util import *
from math import sqrt, ceil, floor
import os
from gpumodel import IGPUModel
import random as r
import numpy.random as nr
from convnet import ConvNet
from data import DataProvider
from options import *
from htmlWriter import *

''' matplotlib python library (Ubuntu/Fedora package name python-matplotlib) '''
import pylab as pl

class ShowNetError(Exception):
    pass

class ShowConvNet(ConvNet):
    def __init__(self, op, load_dic):
        ConvNet.__init__(self, op, load_dic)
        self.summary_dir = self.save_file + '_summary'
        if  not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)
        ''' here, (train,test,pred) corresponds to (train,validation,testing)'''
        self.write_features = [self.write_features_train, self.write_features_test, self.write_features_pred]
        
    def get_gpus(self):
        self.need_gpu = self.op.get_value('show_preds') or self.op.get_value('write_features_train') or \
        self.op.get_value('write_features_test') or self.op.get_value('write_features_pred') or self.op.get_value('confusion_mat')
        if self.need_gpu:
            ConvNet.get_gpus(self)
    
    def init_data_providers(self):
        class Dummy:
            def advance_batch(self):
                pass
        if self.need_gpu:
            ConvNet.init_data_providers(self)
            if self.op.get_value("write_features_pred") or self.op.get_value("show_preds") == 2:
                self.pred_data_provider = DataProvider.get_instance(self.libmodel, self.data_path, self.pred_batch_range,
                                                                    type=self.dp_type, dp_params=self.dp_params, test=DataProvider.DP_PREDICT)
            
        else:
            self.train_data_provider = self.test_data_provider = Dummy()
    
    def import_model(self):
        if self.need_gpu:
            ConvNet.import_model(self)
            
    def init_model_state(self):
        ConvNet.init_model_state(self)
        if self.op.get_value('show_preds') or self.op.get_value('confusion_mat'):
            if not self.op.get_value('logsoftmax'):
                raise ShowNetError('logsoftmax typed layer is not specified')
            self.softmax_idx = self.get_layer_idx(self.op.get_value('logsoftmax'), check_type='logsoftmax')
#             self.softmax_idx = self.get_layer_idx(self.op.get_value('show_preds'), check_type='softmax')
        
        if self.op.get_value('confusion_mat') and not self.op.get_value('test_meta'):
            raise ShowNetError("test-meta is not specified")
        
        self.ftr_layer_idx = np.zeros((3), dtype=np.uint32)
        if self.op.get_value('write_features_train'):
            self.ftr_layer_idx[0] = self.get_layer_idx(self.op.get_value('write_features_train'))
        if self.op.get_value('write_features_test'):
            self.ftr_layer_idx[1] = self.get_layer_idx(self.op.get_value('write_features_test'))        
        if self.op.get_value('write_features_pred'):
            self.ftr_layer_idx[2] = self.get_layer_idx(self.op.get_value('write_features_pred'))            
            
    def init_model_lib(self):
        if self.need_gpu:
            ConvNet.init_model_lib(self)

    def plot_cost(self):

        if self.show_cost not in self.train_outputs[0][0]:
            raise ShowNetError("Cost function with name '%s' not defined by given convnet." % self.show_cost)
        
        for cost_idx in self.cost_idx:
            train_errors = [o[0][self.show_cost][cost_idx] for o in self.train_outputs]
            test_errors = [o[0][self.show_cost][cost_idx] for o in self.test_outputs]
    
            numbatches = len(self.train_batch_range)
            test_errors = np.row_stack(test_errors)
            test_errors = np.tile(test_errors, (1, self.testing_freq))
            test_errors = list(test_errors.flatten())
            test_errors += [test_errors[-1]] * max(0, len(train_errors) - len(test_errors))
            test_errors = test_errors[:len(train_errors)]
    
            numepochs = len(train_errors) / float(numbatches)
            pl.figure(cost_idx)
            x = range(0, len(train_errors))
            pl.plot(x, train_errors, 'k-', label='Training set')
            pl.plot(x, test_errors, 'r-', label='Test set')
            pl.legend()
            # one tick is one epoch
            ticklocs = range(numbatches, len(train_errors) - len(train_errors) % numbatches + 1, numbatches)
            epoch_label_gran = int(ceil(numepochs / 20.))  # aim for about 20 labels
            epoch_label_gran = int(ceil(float(epoch_label_gran) / 10) * 10)  # but round to nearest 10
            ticklabels = map(lambda x: str((x[1] / numbatches)) if x[0] % epoch_label_gran == epoch_label_gran - 1 else '', enumerate(ticklocs))
    
            pl.xticks(ticklocs, ticklabels)
            pl.xlabel('Epoch')
    #        pl.ylabel(self.show_cost)
            pl.title(self.show_cost + ' cost_idx ' + str(cost_idx))
            
            pl.savefig(os.path.join(self.summary_dir, self.getCostPlotName(cost_idx)))
         
    # quantity is a list of values as many as # of testing
    def plot_testing_quantity(self, pl, quantity, linespec, label):
        numbatches = len(self.train_batch_range)
        trainBatch = len(self.train_outputs)
        quantity = np.row_stack(quantity)
        quantity = np.tile(quantity, (1, self.testing_freq))
        quantity = list(quantity.flatten())
        quantity += [quantity[-1]] * max(0, trainBatch - len(quantity))
        quantity = quantity[:trainBatch]
        
        x = range(0, trainBatch)
        pl.plot(x, quantity, linespec, label=label)
    
    def plot_training_quantity(self, quantity, linespec, label):
        trainBatch = len(self.train_outputs)
        x = range(0, trainBatch)
        pl.plot(x, quantity, linespec, label=label)
    
    def plot_training_quantity_tick(self, pl):
        numbatches = len(self.train_batch_range)
        trainBatch = len(self.train_outputs)
        numepochs = trainBatch / float(numbatches)
        ticklocs = range(numbatches, trainBatch - trainBatch % numbatches + 1, numbatches)
        epoch_label_gran = int(ceil(numepochs / 20.))  # aim for about 20 labels
        epoch_label_gran = int(ceil(float(epoch_label_gran) / 10) * 10)  # but round to nearest 10
        ticklabels = map(lambda x: str((x[1] / numbatches)) if x[0] % epoch_label_gran == epoch_label_gran - 1 else '', enumerate(ticklocs))

        pl.xticks(ticklocs, ticklabels)
        pl.xlabel('Epoch')        
        
    def plot_layer_weights_biases(self):
           numbatches = len(self.train_batch_range)
           
           figID = len(self.cost_idx) + 2
           for i, l in enumerate(self.layer_weights_biases[0].keys()):
               mean_abs_biases = [b[l]['biases'] for b in self.layer_weights_biases]
               mean_abs_biasesInc = [b[l]['biasesInc'] for b in self.layer_weights_biases]
               
               pl.figure(figID)
               figID += 1
               self.plot_testing_quantity(pl, mean_abs_biases, 'r-', label='mean abs biases')
               pl.legend()
               self.plot_training_quantity_tick(pl)
               pl.title(l + ' mean abs biases')
               pl.savefig(os.path.join(self.summary_dir, l + '-mean-abs-biases'))
               
               pl.figure(figID)
               figID += 1
               self.plot_testing_quantity(pl, mean_abs_biasesInc, 'k-', label='mean abs biasesInc')
               pl.legend()
               self.plot_training_quantity_tick(pl)
               pl.title(l + ' mean abs biasesInc')
               pl.savefig(os.path.join(self.summary_dir, l + '-mean-abs-biasesInc'))
               for i in xrange(len(self.layer_weights_biases[0][l]['weights'])):
                   mean_abs_weights = [b[l]['weights'][i] for b in self.layer_weights_biases]
                   mean_abs_weightsInc = [b[l]['weightsInc'][i] for b in self.layer_weights_biases]
                   pl.figure(figID)
                   figID += 1
                   self.plot_testing_quantity(pl, mean_abs_weights, 'r-', label='mean abs weights')
                   pl.legend()
                   self.plot_training_quantity_tick(pl)
                   pl.title(l + ' input' + str(i) + ' mean abs weights')
                   pl.savefig(os.path.join(self.summary_dir, l + '-input' + str(i) + '-mean-abs-weights'))
                   
                   pl.figure(figID)
                   figID += 1
                   self.plot_testing_quantity(pl, mean_abs_weightsInc, 'k-', label='mean abs weightsInc')
                   pl.legend()
                   self.plot_training_quantity_tick(pl)
                   pl.title(l + ' input' + str(i) + ' mean abs weightsInc')
                   pl.savefig(os.path.join(self.summary_dir, l + '-input' + str(i) + '-mean-abs-weightsInc'))    
        
    def getCostPlotName(self, cost_idx):
        return 'plot_cost_%d.png' % cost_idx
    
    def getFiltersPlotName(self):
        return 'filters.png'
    
    def getPredPlotName(self):
        return 'predictions.png'
        
    def make_filter_fig(self, filters, filter_start, fignum, _title, num_filters, combine_chans):
        FILTERS_PER_ROW = 16
        MAX_ROWS = 16
        MAX_FILTERS = FILTERS_PER_ROW * MAX_ROWS
        num_colors = filters.shape[0]
        f_per_row = int(ceil(FILTERS_PER_ROW / float(1 if combine_chans else num_colors)))
        filter_end = min(filter_start + MAX_FILTERS, num_filters)
        filter_rows = int(ceil(float(filter_end - filter_start) / f_per_row))
    
        filter_size = int(sqrt(filters.shape[1]))
        fig = pl.figure(fignum)
        fig.text(.5, .95, '%s %dx%d filters %d-%d' % (_title, filter_size, filter_size, filter_start, filter_end - 1), horizontalalignment='center') 
        num_filters = filter_end - filter_start
        if not combine_chans:
            bigpic = np.zeros((filter_size * filter_rows + filter_rows + 1, filter_size * num_colors * f_per_row + f_per_row + 1), dtype=np.single)
        else:
            bigpic = np.zeros((3, filter_size * filter_rows + filter_rows + 1, filter_size * f_per_row + f_per_row + 1), dtype=np.single)
    
        for m in xrange(filter_start, filter_end):
            filter = filters[:, :, m]
            y, x = (m - filter_start) / f_per_row, (m - filter_start) % f_per_row
            if not combine_chans:
                for c in xrange(num_colors):
                    filter_pic = filter[c, :].reshape((filter_size, filter_size))
                    bigpic[1 + (1 + filter_size) * y:1 + (1 + filter_size) * y + filter_size,
                           1 + (1 + filter_size * num_colors) * x + filter_size * c:1 + (1 + filter_size * num_colors) * x + filter_size * (c + 1)] = filter_pic
            else:
                filter_pic = filter.reshape((3, filter_size, filter_size))
                bigpic[:,
                       1 + (1 + filter_size) * y:1 + (1 + filter_size) * y + filter_size,
                       1 + (1 + filter_size) * x:1 + (1 + filter_size) * x + filter_size] = filter_pic
                
        pl.xticks([])
        pl.yticks([])
        if not combine_chans:
            pl.imshow(bigpic, cmap=pl.cm.gray, interpolation='nearest')
        else:
            bigpic = bigpic.swapaxes(0, 2).swapaxes(0, 1)
            pl.imshow(bigpic, interpolation='nearest')
        pl.savefig(os.path.join(self.summary_dir, self.getFiltersPlotName()))
        
    def plot_filters(self):
        filter_start = 0  # First filter to show
        layer_names = [l['name'] for l in self.layers]
        if self.show_filters not in layer_names:
            raise ShowNetError("Layer with name '%s' not defined by given convnet." % self.show_filters)
        layer = self.layers[layer_names.index(self.show_filters)]
        filters = layer['weights'][self.input_idx]
        if layer['type'] == 'fc':  # Fully-connected layer
            num_filters = layer['outputs']
            channels = self.channels
        elif layer['type'] in ('conv', 'local'):  # Conv layer
            num_filters = layer['filters']
            channels = layer['filterChannels'][self.input_idx]
            if layer['type'] == 'local':
                filters = filters.reshape((layer['modules'], layer['filterPixels'][self.input_idx] * channels, num_filters))
                filter_start = r.randint(0, layer['modules'] - 1) * num_filters  # pick out some random modules
                filters = filters.swapaxes(0, 1).reshape(channels * layer['filterPixels'][self.input_idx], num_filters * layer['modules'])
                num_filters *= layer['modules']

        filters = filters.reshape(channels, filters.shape[0] / channels, filters.shape[1])
        # Convert YUV filters to RGB
        if self.yuv_to_rgb and channels == 3:
            R = filters[0, :, :] + 1.28033 * filters[2, :, :]
            G = filters[0, :, :] + -0.21482 * filters[1, :, :] + -0.38059 * filters[2, :, :]
            B = filters[0, :, :] + 2.12798 * filters[1, :, :]
            filters[0, :, :], filters[1, :, :], filters[2, :, :] = R, G, B
        combine_chans = not self.no_rgb and channels == 3
        
        # Make sure you don't modify the backing array itself here -- so no -= or /=
        filters = filters - filters.min()
        filters = filters / filters.max()

        self.make_filter_fig(filters, filter_start, len(self.cost_idx), 'Layer %s' % self.show_filters, num_filters, combine_chans)
    
#     currently only single-view prediction is supported while multiview prediction not
    def plot_predictions(self):
        data = self.get_next_batch(train=self.show_preds)[4]  # get a test batch
        data_provider = self.test_data_provider if self.show_preds == 1 else self.pred_data_provider
        num_classes = data_provider.get_num_classes()
        NUM_ROWS = 2
        NUM_COLS = 4
        NUM_IMGS = NUM_ROWS * NUM_COLS
        NUM_TOP_CLASSES = min(num_classes, 5)  # show this many top labels
        
        label_names = data_provider.batch_meta['label_names']
        if self.only_errors:
            preds = np.zeros((data[0].shape[1], num_classes), dtype=np.single)
        else:
            preds = np.zeros((NUM_IMGS, num_classes), dtype=np.single)
            rand_idx = nr.randint(0, data[0].shape[1], NUM_IMGS)
            data[0] = np.require(data[0][:, rand_idx], requirements='C')
            data[1] = np.require(data[1][:, rand_idx], requirements='C')
        data += [preds]
        
        # Run the model
        self.libmodel.startFeatureWriter(data, self.softmax_idx)
        self.finish_batch()

#         data[2] stores log probabilities and so we take the their exponentials
        data[2] = np.exp(data[2])
                
        fig = pl.figure(len(self.cost_idx) + 1)
        fig.text(.4, .95, '%s test case predictions' % ('Mistaken' if self.only_errors else 'Random'))
        if self.only_errors:
            err_idx = nr.permutation(np.where(preds.argmax(axis=1) != data[1][0, :])[0])[:NUM_IMGS]  # what the net got wrong
            data[0], data[1], preds = data[0][:, err_idx], data[1][:, err_idx], preds[err_idx, :]
            
        data[0] = data_provider.get_plottable_data(data[0])
        for r in xrange(NUM_ROWS):
            for c in xrange(NUM_COLS):
                img_idx = r * NUM_COLS + c
                if data[0].shape[0] <= img_idx:
                    break
                pl.subplot(NUM_ROWS * 2, NUM_COLS, r * 2 * NUM_COLS + c + 1)
                pl.xticks([])
                pl.yticks([])
                img = data[0][img_idx, :, :, :]
                pl.imshow(img, interpolation='nearest')
                if self.show_preds == 1:
                    true_label = int(data[1][0, img_idx])

                img_labels = sorted(zip(data[2][img_idx, :], label_names), key=lambda x: x[0])[-NUM_TOP_CLASSES:]                
                pl.subplot(NUM_ROWS * 2, NUM_COLS, (r * 2 + 1) * NUM_COLS + c + 1, aspect='equal')

                ylocs = np.array(range(NUM_TOP_CLASSES)) + 0.5
                height = 0.5
                width = max(ylocs)
                if self.show_preds == 1:
                    pl.barh(ylocs, [l[0] * width for l in img_labels], height=height,
                            color=['r' if l[1] == label_names[true_label] else 'b' for l in img_labels])
                else:
                    pl.barh(ylocs, [l[0] * width for l in img_labels], height=height,
                            color=['b' for l in img_labels])
                if self.show_preds == 1:
                    pl.title(label_names[true_label])
                pl.yticks(ylocs + height / 2, [l[1] for l in img_labels])
                pl.xticks([width / 2.0, width], ['50%', ''])
                pl.ylim(0, ylocs[-1] + height * 2)
        pl.savefig(os.path.join(self.summary_dir, self.getPredPlotName()))   
        
    ''' write feature from given layer for training/test/predicting data '''
    def do_write_features(self):
        data_parts = [DataProvider.DP_TRAIN, DataProvider.DP_TEST, DataProvider.DP_PREDICT]     
        for i in range(len(data_parts)):
            part = data_parts[i]
            if part == DataProvider.DP_TRAIN and self.write_features_train:
                self.train_data_provider.restart_epoch()
            elif part == DataProvider.DP_TEST and self.write_features_test:
                self.test_data_provider.restart_epoch()
            elif part == DataProvider.DP_PREDICT and self.write_features_pred:
                self.pred_data_provider.restart_epoch()
            else:
                continue
            next_data = self.get_next_batch(train=part)
            num_ftrs = self.layers[self.ftr_layer_idx[i]]['outputs']
            num_batches = len(next_data[3])
            
            opt_dir = os.path.join(self.summary_dir, self.write_features[i])
            if not os.path.exists(opt_dir):
                os.mkdir(opt_dir)
            
            print '%d batches to process' % num_batches
            for j in range(num_batches):
                batch = next_data[2]
                data = next_data[4]
                print 'write feature for %d th of %d batches.  %d ' % (j + 1, num_batches, batch)
                
                ftrs = np.zeros((data[0].shape[1], num_ftrs), dtype=np.single)
                self.libmodel.startFeatureWriter(data + [ftrs], self.ftr_layer_idx[i])
                
                ''' load the next batch while the current one is computing '''
                next_data = self.get_next_batch(train=part)
                self.finish_batch()
                path_out = os.path.join(opt_dir, 'data_batch_%d' % batch)
                if not os.path.isfile(path_out):
                    pickle(path_out, {'data': ftrs, 'labels': data[1]})
                    print "write feature file %s" % path_out
                else:
                    print '%s already exists' % path_out            
            pickle(os.path.join(self.summary_dir, self.write_features[i], 'write_features_batches.meta'),\
                    {'source_model':self.load_file, 'num_vis':num_ftrs})
            
    def plotConfusionMatrix(self):
        num_classes = self.test_data_provider.get_num_classes()
        label_names = self.test_data_provider.batch_meta['label_names']
        
        test_meta = unpickle(os.path.join(self.data_path, self.test_meta))
#         print 'test_meta keys()'
#         print test_meta.keys()
        
        conf_mat = np.zeros((num_classes, num_classes), dtype=np.single)
        next_data = self.get_next_batch(train=DataProvider.DP_TEST)
        preds = np.zeros((0, num_classes), dtype=np.single)
        labels = np.zeros((1, 0), dtype=np.single)
        
        for i in range(len(self.test_data_provider.batch_range)):
            data = next_data[4]
            print 'process %d th batch %d' % (next_data[1], next_data[2])
            num_imgs = data[0].shape[1]
            loc_preds = np.zeros((num_imgs, num_classes), dtype=np.single)
            data += [loc_preds]
            self.libmodel.startFeatureWriter(data, self.softmax_idx)
            next_data = self.get_next_batch(train=DataProvider.DP_TEST)
            self.finish_batch()
            data[2] = np.exp(data[2])
            preds = np.vstack((preds, data[2]))
            labels = np.hstack((labels, data[1]))
            sort_idx = np.argsort(data[2], 1)
            pred_label = sort_idx[:, num_classes - 1]
#             print 'min max label:%f %f ' % (np.min(data[1]), np.max(data[1]))
#             print 'min max pred_label: %f %f' % (min(pred_label),max(pred_label))
            for j in xrange(num_imgs):
                conf_mat[pred_label[j], int(data[1][0, j])] += 1
#         normalize each column in confusion matrix
        col_sum = np.sum(conf_mat, 0).reshape((num_classes, 1))
        col_sum_sort = np.sort(np.sum(conf_mat, 0))
        col_sum0 = np.sum(conf_mat)
#         print 'sorted col_sum'
#         print col_sum_sort
#         print 'conf_mat sum:%f' % col_sum0
        conf_mat = conf_mat / col_sum
        pl.figure(0)
        pl.clf()
        pl.imshow(np.tile(conf_mat[:, :, np.newaxis], (1, 1, 3)))
        pl.savefig(os.path.join(self.summary_dir, 'confusion_mat.png'))
        confmat_dict = {}
        confmat_dict['conf_mat'] = conf_mat
        confmat_dict['label_names'] = label_names
        confmat_dict['preds'] = preds
        confmat_dict['img_files'] = test_meta['img_files']
        confmat_dict['img_labels'] = labels
        pickle(os.path.join(self.summary_dir, 'conf_mat'), confmat_dict)
        
        
    def start(self):
        '''
        write a summary (html) file in 'summary_dir'
        
        figure ID assignment:
         len(self.cost_idx) figures for each cost plot
         1 figure for filters visualization
         1 figure for predictions plot
         the rest of figures for tracing evolution of layer-wise weights, weightsInc, biases, biasesInc
        '''
        self.op.print_values()
        
        htmlFile = os.path.join(self.summary_dir, 'summary.html')
        
        fp = open(htmlFile, 'w')
        ckPtsFile = '%s:%s' % ('Checking point folder', self.load_file)    
        HTMLwriter.writeParagraph(fp, 'h3', ckPtsFile, ['b'])
        HTMLwriter.writeBlankLine(fp)
        option_values = self.op.print_values_to_string()
        for line in option_values:
            HTMLwriter.writeParagraph(fp, 'p', line)
        
        ''' make copies of layer definition files '''
        layer_def_fn = os.path.basename(self.op.options['layer_def'].value)
        layer_params_fn = os.path.basename(self.op.options['layer_params'].value)
        src_def, dest_def = self.op.options['layer_def'].value, os.path.join(self.summary_dir, layer_def_fn)
        src_params, dest_params = self.op.options['layer_params'].value, os.path.join(self.summary_dir, layer_params_fn)
        if not (os.path.exists(dest_def) and os.path.exists(dest_params)):
            if os.name == 'posix':
                # unix
                cmd_def = 'cp ' + src_def + ' ' + dest_def
                cmd_params = 'cp ' + src_params + ' ' + dest_params
            elif os.name == 'nt':
                # windows
                cmd_def = 'copy ' + src_def + ' ' + dest_def
                cmd_params = 'copy ' + src_params + ' ' + dest_params
            else:
                raise ShowNetError("unknown OS")
            os.system(cmd_def)
            os.system(cmd_params)            
        else:
            print '%s exists \n %s exists' % (dest_def, dest_params)

                
        HTMLwriter.writeBlankLine(fp)
        HTMLwriter.writeBlankLine(fp)
        if self.show_cost:
            self.plot_cost()
            
            for cost_idx in self.cost_idx:
                if cost_idx == 0:
                    HTMLwriter.writeParagraph(fp, 'h2', 'negative log-probability vs. epoch', ['b'])
                else:
                    HTMLwriter.writeParagraph(fp, 'h2', 'error rate vs. epoch', ['b'])
                HTMLwriter.insertImage(fp, self.getCostPlotName(cost_idx))
        if self.show_filters:
            self.plot_filters()
            HTMLwriter.writeParagraph(fp, 'h2', 'filters', ['b'])
            HTMLwriter.insertImage(fp, self.getFiltersPlotName())
        if self.show_preds:
            self.plot_predictions()
            HTMLwriter.writeParagraph(fp, 'h2', 'predictions', ['b'])
            HTMLwriter.insertImage(fp, self.getPredPlotName())         
        if self.write_features_train or self.write_features_test or self.write_features_pred:
            self.do_write_features()
        
        
        if self.show_layer_weights_biases:
            self.plot_layer_weights_biases()
        if self.confusion_mat:
            self.plotConfusionMatrix()
        
        
        fp.close()
            
    @classmethod
    def get_options_parser(cls):
        op = ConvNet.get_options_parser()
        for option in list(op.options):
            if option not in ('gpu', 'load_file', 'train_batch_range', 'test_batch_range', 'multiview_test'):
                op.delete_option(option)
                
        op.add_option("show-cost", "show_cost", StringOptionParser, "Show specified objective function", default="")
        op.add_option("show-filters", "show_filters", StringOptionParser, "Show learned filters in specified layer", default="")
        op.add_option("input-idx", "input_idx", IntegerOptionParser, "Input index for layer given to --show-filters", default=0)
        op.add_option("cost-idx", "cost_idx", ListOptionParser(IntegerOptionParser), "Cost function return value index for --show-cost", default=[])
#         op.add_option("cost-idx", "cost_idx", IntegerOptionParser, "Cost function return value index for --show-cost", default=0)
        op.add_option("no-rgb", "no_rgb", BooleanOptionParser, "Don't combine filter channels into RGB in layer given to --show-filters", default=False)
        op.add_option("yuv-to-rgb", "yuv_to_rgb", BooleanOptionParser, "Convert RGB filters to YUV in layer given to --show-filters", default=False)
        op.add_option("channels", "channels", IntegerOptionParser, "Number of channels in layer given to --show-filters (fully-connected layers only)", default=0)
        op.add_option("show-preds", "show_preds", IntegerOptionParser,
                      "Show predictions made by given softmax on test or predicting set. 1:test set 2:predicting set", default=0)
        op.add_option("pred-batch-range", "pred_batch_range", RangeOptionParser, "Data batch range: predicting set")
        op.add_option("logsoftmax", "logsoftmax", StringOptionParser, "name of logsoftmax typed layer", default="")
        op.add_option("only-errors", "only_errors", BooleanOptionParser, "Show only mistaken predictions (to be used with --show-preds)", default=False, requires=['show_preds'])
        
        op.add_option("write-features-train", "write_features_train", StringOptionParser, "Write train data features from given layer", default='', requires=[])
        op.add_option("write-features-test", "write_features_test", StringOptionParser, "Write test data features from given layer", default='', requires=[])
        op.add_option("write-features-pred", "write_features_pred", StringOptionParser, "Write prediction data features from given layer", default='', requires=[])        
#         op.add_option("feature-path", "feature_path", StringOptionParser, "Write test data features to this path (to be used with --write-features)", default="")
        
        op.add_option("show-layer-weights-biases", "show_layer_weights_biases", IntegerOptionParser, "Show evolution of layer-wise mean absolute of weights, weightsInc, bias, biasInc", default=0)
        
        op.add_option("confusion-mat", "confusion_mat", IntegerOptionParser, "plot confusion matrix", default=0)
        op.add_option("test-meta", "test_meta", StringOptionParser, "meta file for testing data used in plotting confusion matrix", default="")
 
        op.options['load_file'].default = None
        return op
    
if __name__ == "__main__":
    try:
        op = ShowConvNet.get_options_parser()
        op, load_dic = IGPUModel.parse_options(op)
        model = ShowConvNet(op, load_dic)
        model.start()
    except (UnpickleError, ShowNetError, opt.GetoptError), e:
        print "----------------"
        print "Error:"
        print e 
