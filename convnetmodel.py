import layer as lay
from mock import Mock, PropertyMock, MagicMock
import cPickle

class ConvnetModel:
    def __init__(self, layer_def_file, layer_params_file, train_provider, verbose = False):
        self.layer_def = layer_def_file
        self.layer_params = layer_params_file
        self.minibatch_size = 128
        self.gpu_device_id = 0
        self.train_provider = train_provider
        self.verbose = verbose
        self.layers = self._BuildLayers()
        self._InitializeModel()
        self.train_results =[]
        
    def _BuildLayers(self):
        fakeModel = Mock()
        myOp = Mock()
        myOp.get_value.return_value = 0
        type(fakeModel).train_data_provider = self.train_provider
        type(fakeModel).op = PropertyMock(return_value = myOp)
        return lay.LayerParser.parse_layers(self.layer_def, self.layer_params, fakeModel)
    
    def _InitializeModel(self):
        self.libmodel = __import__('_ConvNet')
        self.libmodel.initModel(self.layers, self.minibatch_size, self.gpu_device_id)
    
    def Run(self, nIterations):
        next_data = self.train_provider.get_next_batch()
        for i in xrange(nIterations):
            if self.verbose:
                print("Running iteration %d of %d" % (i, nIterations))
            data = next_data
            self.libmodel.startBatch(data[2], False)
            next_data = self.train_provider.get_next_batch()
            thisBatchResult = self.libmodel.finishBatch()
            self.train_results.append(thisBatchResult)
