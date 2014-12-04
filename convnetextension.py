from convnet import *
import cPickle
from MLTools import ModelEvaluation
from time import asctime, localtime, strftime

class ConvNetWithAutoStop(ConvNet):
    def __init__(self, op, load_dic, dp_params={}):
        ConvNet.__init__(self, op, load_dic, dp_params)
        if not load_dic:
            self.save_file = self.filename
            self.valid_outputs  = []
            self.model_state['valid_outputs'] = self.valid_outputs
    
    def init_data_providers(self):
        self.dp_params['convnet'] = self
        try:
            self.test_data_provider = DataProvider.get_instance(self.data_path, self.test_batch_range,
                                                                type=self.dp_type, dp_params=self.dp_params, test=True)
            self.valid_data_provider = DataProvider.get_instance(self.data_path, self.valid_batch_range,
                                                                 self.model_state["epoch"], self.model_state["batchnum"],
                                                                type=self.dp_type, dp_params=self.dp_params, test=False)
            self.train_data_provider = DataProvider.get_instance(self.data_path, self.train_batch_range,
                                                                     self.model_state["epoch"], self.model_state["batchnum"],
                                                                     type=self.dp_type, dp_params=self.dp_params, test=False)
        except DataProviderException, e:
            print "Unable to create data provider: %s" % e
            self.print_data_providers()
            sys.exit()
    
    def get_next_validation_batch(self):
        dp = self.valid_data_provider
        return self.parse_batch_data(dp.get_next_batch(), train=True)

    def print_valid_results(self):
        self.print_costs(self.valid_outputs[-1])

    def hasConverged(self):
        maxEpochsWithNoImprovement = self.iterations_to_wait
        
        last_valid_logprob = self.valid_outputs[-1][0]['logprob'][0]
            
        if last_valid_logprob < self.best_valid_logprob:
            self.best_valid_logprob = last_valid_logprob
            self.best_epoch = self.epoch
            return False
        else:
            if self.epoch - self.best_epoch < maxEpochsWithNoImprovement:
                return False
            else:
                return True

    def hasConverged_valid(self):
        last_logprob = self.valid_outputs[-1][0]['logprob'][0]
        if last_logprob <= self.target_logprob:
            return True
        return False

    def run_finetuning(self):
        print "training %d epochs with <Train> and <Valid> sets:" % self.num_epochs

        next_data = self.get_next_batch()
        for i in range(self.num_epochs):
            data = next_data
            self.epoch, self.batchnum = data[0], data[1]
            self.print_iteration()
            sys.stdout.flush()
            
            compute_time_py = time()
            self.start_batch(data)

            # load the next validation while the current one is computing
            validationdata = self.get_next_validation_batch()
            
            batch_output = self.finish_batch()
            self.train_outputs += [batch_output]

            self.start_batch(validationdata, train=True)  #Now backpropagating errors

            # load the next batch while the current one is computing

            next_data = self.get_next_batch()

            batch_output = self.finish_batch()
            self.valid_outputs += [batch_output]
            self.print_train_results()
            self.print_valid_results()
            self.print_train_time(time() - compute_time_py)

            if (i % 50 == 0):
                self.conditional_save()

        print "testing once in the test set:"
        
        self.sync_with_host()
        self.test_outputs += [self.get_test_error()]
        self.print_test_results()
        self.print_test_status()
        self.conditional_save()

    def print_iteration(self):
        print "%s %d.%d..." % (strftime("%H:%M:%S"), self.epoch, self.batchnum),
    
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

        self.best_valid_logprob = 1e100
        self.best_epoch = 0

        if (self.test_only):
            self.test_once()
            return
        if (self.finetunning):
            self.run_finetuning()
            return
        
        print "training at most %d epochs with <Train> set:" % self.num_epochs
        next_data = self.get_next_batch()
        for i in range(self.num_epochs):
            data = next_data
            self.epoch, self.batchnum = data[0], data[1]
            self.print_iteration()
            sys.stdout.flush()
            
            compute_time_py = time()
            self.start_batch(data)

            # load the next validation while the current one is computing
            validationdata = self.get_next_validation_batch()
            
            batch_output = self.finish_batch()
            self.train_outputs += [batch_output]

            self.start_batch(validationdata, train=False)

            # load the next batch while the current one is computing

            next_data = self.get_next_batch()

            batch_output = self.finish_batch()

            if (i % 50 == 0):
                self.conditional_save()

            self.valid_outputs += [batch_output]
            self.print_train_results()
            self.print_valid_results()
            self.print_train_time(time() - compute_time_py)
            if self.hasConverged():
                 self.target_logprob = self.train_outputs[-1][0]['logprob'][0]
                 print "Stopping at interation %d. Target logprob is %.4f\n" % (self.epoch, self.target_logprob)
                 break
        if not hasattr(self,'target_logprob'):
            self.target_logprob = self.train_outputs[-1][0]['logprob'][0]
        
        print "training at most %d epochs with <Train> and <Valid> sets:" % self.num_epochs
        next_data = self.get_next_validation_batch()
        for i in range(self.num_epochs):
            data = next_data
            self.epoch, self.batchnum = data[0], data[1]
            self.print_iteration()
            sys.stdout.flush()
            
            compute_time_py = time()
            self.start_batch(data)

            # load the next validation while the current one is computing
            validationdata = self.get_next_validation_batch()
            
            batch_output = self.finish_batch()
            self.train_outputs += [batch_output]

            self.start_batch(validationdata, train=True)  #Now backpropagating errors

            # load the next batch while the current one is computing

            next_data = self.get_next_batch()

            batch_output = self.finish_batch()
            self.valid_outputs += [batch_output]
            self.print_train_results()
            self.print_valid_results()
            self.print_train_time(time() - compute_time_py)

            if self.hasConverged_valid():
                 print "Stopping at interation %d. " % (self.epoch)
                 break
        
        print "testing once in the test set:"
        self.test_once()
        
    def test_once(self):
        self.sync_with_host()
        self.test_outputs += [self.get_test_error()]
        self.print_test_results()
        self.print_test_status()
        self.conditional_save()
            
    def get_test_patches_and_filenames(self):
        epoch, batchNum, data, filenames = self.test_data_provider.get_patches_and_filenames()
        return data, filenames
        
    def get_predictions(self, data):
        num_classes = self.test_data_provider.get_num_classes()
        
        preds = numpy.zeros((data[0].shape[1], num_classes), dtype=numpy.single)
        data += [preds]

        # Run the model
        softmax_idx = self.get_layer_idx('probs', check_type='softmax')
        self.libmodel.startFeatureWriter(data, softmax_idx)
        results = self.finish_batch()

        return preds, results
        
    def get_test_error(self):
        data, filenames = self.get_test_patches_and_filenames()
        if self.test_on_images:
            print 'testing on images'
            probabilities, test_results = self.get_predictions(data)
                   
            fileProbs, fileLabels, fileIDs = ModelEvaluation.GetUnormalizedJointLogProbability(probabilities, data[1].reshape(-1), filenames)
            filePredictions = ModelEvaluation.GetPredictions(fileProbs)
            fileAccuracy, misclassifiedFiles = ModelEvaluation.CalculateAccuracy(filePredictions, fileLabels)
            
            nExamples = test_results[1]
            results =  ({'logprob' : [test_results[0]['logprob'][0], (1-fileAccuracy) * test_results[1]]}, test_results[1])
            if self.test_only: # Print the individual batch results for safety
                print str(results)
            cPickle.dump({'fileProbs' : fileProbs, 'fileLabels' : fileLabels, 'fileIDs': fileIDs, 'fileAccuracy' : fileAccuracy}, open(self.test_output_file,'w'))
        else:
            print 'not testing on images'
            self.libmodel.startBatch(data, True)
            results = self.finish_batch()
        return results
    @classmethod
    def get_options_parser(cls):
        op = ConvNet.get_options_parser()
        op.add_option("finetunning", "finetunning", BooleanOptionParser, "Train for specified epochs and exit", default=False)
        op.add_option("valid-range", "valid_batch_range", RangeOptionParser, "Data batch range: validation")
        op.add_option("filename", "filename", StringOptionParser, "Filename")
        op.add_option("test-on-images", "test_on_images", BooleanOptionParser, "Test On Images", default=False)
        op.add_option("test-output-file", "test_output_file", StringOptionParser, "Test Output File")
        op.add_option("iterations-to-wait", "iterations_to_wait", IntegerOptionParser, "Max # of iterations without improvement before stopping", default=20)
        return op
        
if __name__ == "__main__":
    op = ConvNetWithAutoStop.get_options_parser()

    op, load_dic = IGPUModel.parse_options(op)
    model = ConvNetWithAutoStop(op, load_dic)
    model.start()
