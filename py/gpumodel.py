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

import numpy as n
import os
from time import time, asctime, localtime, strftime
from numpy.random import randn, rand
from numpy import s_, dot, tile, zeros, ones, zeros_like, array, ones_like
from util import *
from data import *
from options import *
from math import ceil, floor, sqrt
from data import DataProvider, dp_types
import sys
import shutil
import platform
import random
from os import linesep as NL
import socket 

class ModelStateException(Exception):
    pass

# GPU Model interface
class IGPUModel:
    def __init__(self, model_name, op, load_dic, filename_options=None, dp_params={}):
        # these are input parameters
        self.model_name = model_name
        self.op = op
        self.options = op.options
        self.load_dic = load_dic
        self.filename_options = filename_options
        self.dp_params = dp_params
        self.get_gpus()
        self.fill_excused_options()
        # assert self.op.all_values_given()

        for o in op.get_options_list():
            setattr(self, o.name, o.value)

        ''' these are things that the model must remember but they're not input parameters '''
        if load_dic:
            self.model_state = load_dic["model_state"]
            self.save_file = self.options["load_file"].value
            if not os.path.isdir(self.save_file):
                self.save_file = os.path.dirname(self.save_file)
        else:
            self.model_state = {}
            if filename_options is not None:

                self.save_file = model_name + "_" + '_'.join(['%s_%s' % (char, self.options[opt].get_str_value()) for opt, char in filename_options]) + '_' + strftime('%Y-%m-%d_%H.%M.%S')

                # added by LL
                computer_name = socket.gethostname().split('.')[0]
                self.save_file = self.save_file + computer_name
                # added end

            epochBatchPerm = range(len(self.train_batch_range))
            random.shuffle(epochBatchPerm)

            self.model_state["train_outputs"] = []
            self.model_state["test_outputs"] = []
            '''On evaluation on validation data, record weight (bias) & weight_inc (bias_inc) in each layer '''
            ''' model_state["layer_weights_biases"][batch_idx]. Also see func 'store_layer_weights_biases' '''
            self.model_state["layer_weights_biases"] = []
            self.model_state["epoch"] = 1
#             self.model_state["batchnum"] = self.train_batch_range[0]
            self.model_state["batch_idx"] = 0
            self.model_state["epochBatchPerm"] = epochBatchPerm
            self.model_state["weightsLayerEpsScale"] = 1.0

        self.import_model()
        self.init_data_providers()
        if load_dic:
            self.train_data_provider.advance_batch()

        '''model state often requires knowledge of data provider, so it's initialized after data provider'''
        try:
            self.init_model_state()
        except ModelStateException, e:
            print e
            sys.exit(1)

        for var, val in self.model_state.iteritems():
            setattr(self, var, val)

#         self.import_model()
        self.init_model_lib()

        ''' a quick hack. Todo '''
        if hasattr(self, 'libmodel'):
            self.libmodel.setWeightsLayerEpsScale(self.weightsLayerEpsScale)


    def import_model(self):
        print "========================="
        print "Importing %s C++ module" % ('_' + self.model_name)
        self.libmodel = __import__('_' + self.model_name)

    def fill_excused_options(self):
        pass

    def init_data_providers(self):
        self.dp_params['convnet'] = self
        self.dp_params['PCA_pixel_alter'] = self.PCA_pixel_alter        
        self.dp_params['regress_L_channel_only'] = self.regress_L_channel_only
        self.dp_params['use_local_context_ftr'] = self.use_local_context_ftr
        self.dp_params['use_local_context_color_ftr'] = self.use_local_context_color_ftr
        if hasattr(self,'use_position_ftr'):
            self.dp_params['use_position_ftr'] = self.use_position_ftr
        try:
            self.test_data_provider = DataProvider.get_instance(self.libmodel, self.data_path, self.test_batch_range,
                                                                type=self.dp_type, dp_params=self.dp_params, test=DataProvider.DP_TEST)
            self.train_data_provider = DataProvider.get_instance(self.libmodel, self.data_path, self.train_batch_range,
                                                                     self.model_state["epoch"], self.model_state["batch_idx"],
                                                                     self.model_state["epochBatchPerm"],
                                                                     type=self.dp_type, dp_params=self.dp_params, test=DataProvider.DP_TRAIN)
        except DataProviderException, e:
            print "Unable to create data provider: %s" % e
            self.print_data_providers()
            sys.exit()

    def init_model_state(self):
        pass
#         pred_layer_name = self.op.get_value('pred_layer_name')
#         if pred_layer_name:
#             self.pred_layer_idx = self.get_layer_idx(pred_layer_name)

    def init_model_lib(self):
        pass

    def setWeightsLayerEpsScale(self, num_batches_done):
        scale = 1.0 / (1.0 + float(num_batches_done) / float(self.weights_eps_scale_T))
        self.weightsLayerEpsScale = scale
        self.libmodel.setWeightsLayerEpsScale(scale)
        pass

    def start(self):
        if self.test_only:
            self.test_outputs += [self.get_test_error()]
            self.print_test_results()
            sys.exit(0)
        self.train()

    def train(self):
        print "========================="
        print "Training %s" % self.model_name
        self.op.print_values()
        print "========================="
        self.print_model_state()
        print "Running on CUDA device(s) %s" % ", ".join("%d" % d for d in self.device_ids)
        print "Current time: %s" % asctime(localtime())
        print "Saving checkpoints to %s" % os.path.join(self.save_path, self.save_file)
        print "========================="
        next_data = self.get_next_batch()
        
        while self.epoch <= self.num_epochs:
            data = next_data
            self.epoch, self.batch_idx, self.batchnum, self.epochBatchPerm = data[0], data[1], data[2], data[3]
            self.print_iteration()
            sys.stdout.flush()

            self.setWeightsLayerEpsScale(self.get_num_batches_done() - 1)

            compute_time_py = time()
            self.start_batch(data)

            ''' load the next batch while the current one is computing '''
            next_data = self.get_next_batch()
            batch_output = self.finish_batch()

            
            self.train_outputs += [batch_output]
            self.print_train_results()

            if self.get_num_batches_done() % self.testing_freq == 0:
                self.sync_with_host()
                # each element in list 'self.test_outputs' is [costs, numcases] where costs is a dictionary.
                # Key is cost name and value is a list of floating numbers
                self.test_outputs += [self.get_test_error()]
                if hasattr(self, 'layer_weights_biases'):
                    self.layer_weights_biases += [self.store_layer_weights_biases()]
                self.print_test_results()
                self.print_test_status()
                self.conditional_save()

            self.print_train_time(time() - compute_time_py)
            if len(next_data) > 5:
                self.train_data_provider.print_batch_timing(next_data[5])
            else:
                print '\n'

        self.cleanup()

    def cleanup(self):
        sys.exit(0)

    def sync_with_host(self):
        self.libmodel.syncWithHost()

    def print_model_state(self):
        pass

    def get_num_batches_done(self):
        return len(self.train_batch_range) * (self.epoch - 1) + self.batch_idx + 1
#         return len(self.train_batch_range) * (self.epoch - 1) + self.batchnum - self.train_batch_range[0] + 1

    def get_next_batch(self, train=DataProvider.DP_TRAIN):
        if train == DataProvider.DP_TRAIN:
            dp = self.train_data_provider
        elif train == DataProvider.DP_TEST:
            dp = self.test_data_provider
        else:
            dp = self.pred_data_provider
        return self.parse_batch_data(dp.get_next_batch(), train=train)

    def parse_batch_data(self, batch_data, train=True):
        # epoch, batchNum, epochBatchPerm, ['data':data,'labels':label]
        return batch_data[0], batch_data[1], batch_data[2], batch_data[3], batch_data[4]['data']

    def start_batch(self, batch_data, train=True):
        self.libmodel.startBatch(batch_data[4], not train)

    # return [dict, numcases]
    # dict['costname']=errorList
    # errorList=[error1,error2]
    def finish_batch(self):
        return self.libmodel.finishBatch()

    def print_iteration(self):
        print "\t%d.%d. [%d]..." % (self.epoch, (self.batch_idx + 1), self.batchnum),

    def print_train_time(self, compute_time_py):
        print "(%.3f sec)" % (compute_time_py),

    def print_train_results(self):
        batch_error = self.train_outputs[-1][0]
        if not (batch_error > 0 and batch_error < 2e20):
            print "Crazy train error: %.6f" % batch_error
            self.cleanup()

        print "Train error: %.6f " % (batch_error),

    def print_test_results(self):
        batch_error = self.test_outputs[-1][0]
        print "%s\t\tTest error: %.6f" % (NL, batch_error),

    def store_layer_weights_biases(self):
        pass

    def print_test_status(self):
        status = (len(self.test_outputs) == 1 or self.test_outputs[-1][0] < self.test_outputs[-2][0]) and "ok" or "WORSE"
        print status,

    def adjust_costlayer_coeff(self):
        pass

    def multiplyWeightsLayerEpsScale(self):
        pass

    def conditional_save(self):
        batch_error = self.test_outputs[-1][0]
        if batch_error > 0 and batch_error < self.max_test_err:
            self.save_state()
        else:
            print "\tTest error > %g, not saving." % self.max_test_err,

    def aggregate_test_outputs(self, test_outputs):
        test_error = tuple([sum(t[r] for t in test_outputs) / (1 if self.test_one else len(self.test_batch_range)) for r in range(len(test_outputs[-1]))])
        return test_error

#     def predict(self):
#         if not self.op.get_value('pred_layer_name'):
#             
#         
#         next_data = self.get_next_batch(train=False)
#         while True:
#             data = next_data
#             print 'predict on batch with (index,batchNum) : (%d,%d)' % (data[1], data[2])
            

    def get_test_error(self):
        next_data = self.get_next_batch(train=DataProvider.DP_TEST)
#         print "test data shape"
#         print next_data[5][0].shape
#         print next_data[5][1].shape
        test_outputs = []
        while True:
            data = next_data
            print 'test on batch with (index,batchNum) : (%d,%d)' % (data[1], data[2])
            self.start_batch(data, train=False)
            load_next = not self.test_one and data[1] < (len(self.test_batch_range) - 1)
            if load_next:  # load next batch
                next_data = self.get_next_batch(train=DataProvider.DP_TEST)
#             print "test data shape"
#             print next_data[5][0].shape
#             print next_data[5][1].shape
            test_outputs += [self.finish_batch()]
#             print "\n [get_test_error] batch_idx %d: batch_num %d :%s" % (data[1], data[2], str(test_outputs[-1]))
            if not load_next:
                break
            sys.stdout.flush()

        return self.aggregate_test_outputs(test_outputs)

    def set_var(self, var_name, var_val):
        setattr(self, var_name, var_val)
        self.model_state[var_name] = var_val
        return var_val

    def get_var(self, var_name):
        return self.model_state[var_name]

    def has_var(self, var_name):
        return var_name in self.model_state

    def save_state(self):
        for att in self.model_state:
            if hasattr(self, att):
                self.model_state[att] = getattr(self, att)

        dic = {"model_state": self.model_state,
               "op": self.op}

        checkpoint_dir = os.path.join(self.save_path, self.save_file)
        checkpoint_file = "%d.%d" % (self.epoch, self.batch_idx)
        checkpoint_file_full_path = os.path.join(checkpoint_dir, checkpoint_file)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        pickle(checkpoint_file_full_path, dic, compress=self.zip_save)

        for f in sorted(os.listdir(checkpoint_dir), key=alphanum_key):
            if sum(os.path.getsize(os.path.join(checkpoint_dir, f2)) for f2 in os.listdir(checkpoint_dir)) > self.max_filesize_mb * 1024 * 1024 and f != checkpoint_file:
                pass
#                 os.remove(os.path.join(checkpoint_dir, f))
            else:
                break

    @staticmethod
    def load_checkpoint(load_dir):
        if os.path.isdir(load_dir):
            return unpickle(os.path.join(load_dir, sorted(os.listdir(load_dir), key=alphanum_key)[-1]))
        return unpickle(load_dir)

    @staticmethod
    def get_options_parser():
        op = OptionsParser()
        op.add_option("f", "load_file", StringOptionParser, "Load file", default="", excuses=OptionsParser.EXCLUDE_ALL)
        op.add_option("train-range", "train_batch_range", RangeOptionParser, "Data batch range: training",default=range(0,1))
        op.add_option("test-range", "test_batch_range", RangeOptionParser, "Data batch range: testing",default=range(1,2))
        op.add_option("data-provider", "dp_type", StringOptionParser, "Data provider", default="default")
        op.add_option("test-freq", "testing_freq", IntegerOptionParser, "Testing frequency", default=25)
        op.add_option("epochs", "num_epochs", IntegerOptionParser, "Number of epochs", default=5000)
        op.add_option("data-path", "data_path", StringOptionParser, "Data path")
        op.add_option("save-path", "save_path", StringOptionParser, "Save path")
        op.add_option("max-filesize", "max_filesize_mb", IntegerOptionParser, "Maximum save file size (MB)", default=5000)
        op.add_option("max-test-err", "max_test_err", FloatOptionParser, "Maximum test error for saving")
        op.add_option("num-gpus", "num_gpus", IntegerOptionParser, "Number of GPUs", default=1)
        op.add_option("test-only", "test_only", BooleanOptionParser, "Test and quit?", default=0)
        op.add_option("zip-save", "zip_save", BooleanOptionParser, "Compress checkpoints?", default=0)
        op.add_option("test-one", "test_one", BooleanOptionParser, "Test on one batch at a time?", default=1)
        ''' to remove 'gpu' option in the future '''
        op.add_option("gpu", "gpu", ListOptionParser(IntegerOptionParser), "GPU override", default=OptionExpression("[0] * num_gpus"))
        op.add_option("default-gpu", "default_gpu", IntegerOptionParser, "default GPU ID", default=0)

        ''' the option below are added for image adjustment project '''
        op.add_option("use-local-context-ftr", "use_local_context_ftr", BooleanOptionParser, "use local context ftr?", default=0)
        op.add_option("use-local-context-color-ftr", "use_local_context_color_ftr", BooleanOptionParser,\
                      "use local context color feature (e.g. mean color of pooling regions) ", default=0,\
                      requires=['use_local_context_ftr'])              
        op.add_option("regress-L-channel-only", "regress_L_channel_only", BooleanOptionParser,\
                      "regress luminance(L) channel only?", default=0)
        op.add_option("in-img-dir", 'in_img_dir', StringOptionParser,
                      "overwrite input image folder in batches.meta", default='')           
        return op

    @staticmethod
    def print_data_providers():
        print "Available data providers:"
        for dp, desc in dp_types.iteritems():
            print "    %s: %s" % (dp, desc)

    def get_gpus(self):
        self.device_ids = [get_gpu_lock(g) for g in self.op.get_value('gpu')]
        if GPU_LOCK_NO_LOCK in self.device_ids:
            print "Not enough free GPUs!"
            sys.exit()

    @staticmethod
    def parse_options(op):
        try:
            load_dic = None
            options = op.parse()
            if options["load_file"].value_given:
                load_dic = IGPUModel.load_checkpoint(options["load_file"].value)
                old_op = load_dic["op"]
                old_op.merge_from(op)
                op = old_op
            op.eval_expr_defaults()
            return op, load_dic
        except OptionMissingException, e:
            print e
            op.print_usage()
        except OptionException, e:
            print e
        except UnpickleError, e:
            print "Error loading checkpoint:"
            print e
        sys.exit()

