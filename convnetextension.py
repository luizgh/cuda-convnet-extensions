from convnet import *

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
        maxEpochsWithNoImprovement = 20

        if self.epoch < maxEpochsWithNoImprovement:
            return False

        valid_logprob = [i[0]['logprob'][0] for i in self.valid_outputs]
        best_so_far = [valid_logprob[i] < min(valid_logprob[0:i]) for i in range(1,len(valid_logprob))]
        if not any(best_so_far[-maxEpochsWithNoImprovement:]):
            return True
        return False

    def hasConverged_valid(self):
        last_logprob = self.train_outputs[-1][0]['logprob'][0]
        if last_logprob <= self.target_logprob:
            return True
        return False


    def train(self):
        print "Saving checkpoints to %s" % os.path.join(self.save_path, self.save_file)
        
        print "training at most 1000 epochs with <Train> set:"
        next_data = self.get_next_batch()
        for i in range(1000):
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
            self.valid_outputs += [batch_output]
            self.print_train_results()
            self.print_valid_results()
            self.print_train_time(time() - compute_time_py)
            if self.hasConverged():
                 self.target_logprob = self.train_outputs[-1][0]['logprob'][0]
                 print "Stopping at interation %d. Target logprob is %.4f\n" % (self.epoch, self.target_logprob)
                 break
        
        
        print "training 100 epochs with <Validation> set:"
        next_data = self.get_next_validation_batch()
        for i in range(1000):
            data = next_data
            self.epoch, self.batchnum = data[0], data[1]
            self.print_iteration()
            sys.stdout.flush()
            
            compute_time_py = time()
            self.start_batch(data)

            # load the next batch while the current one is computing
            next_data = self.get_next_validation_batch()
            
            batch_output = self.finish_batch()
            self.train_outputs += [batch_output]
            self.print_train_results()
            self.print_train_time(time() - compute_time_py)

            if self.hasConverged_valid():
                 print "Stopping at interation %d. " % (self.epoch)
                 break
        
        print "testing once in the test set:"
        
        self.sync_with_host()
        self.test_outputs += [self.get_test_error()]
        self.print_test_results()
        self.print_test_status()
        self.conditional_save()
            
    @classmethod
    def get_options_parser(cls):
        op = ConvNet.get_options_parser()
        op.add_option("valid-range", "valid_batch_range", RangeOptionParser, "Data batch range: validation")
        op.add_option("filename", "filename", StringOptionParser, "Filename")
        return op
        
if __name__ == "__main__":
    op = ConvNetWithAutoStop.get_options_parser()

    op, load_dic = IGPUModel.parse_options(op)
    model = ConvNetWithAutoStop(op, load_dic)
    model.start()
