from textureconvnet import *
from data import *
import numpy 

def GetNewRepresentationFromRandomPatches(numInteractions, data_provider, model):
    #Get test data
    batchInformation = data_provider.get_next_batch_with_filenames()
    data = batchInformation[2]
    numExamplesPerBatch = data[0].shape[1]
    data_shape = model.layers[10]['outputs']
    newDatasetX = numpy.zeros((numExamplesPerBatch * numInteractions, data_shape), dtype=numpy.single)
    newDatasetY = numpy.zeros((numExamplesPerBatch * numInteractions), dtype=numpy.single)  
    allFilenames = []
    
    for i in xrange(numInteractions):
        print("Iteration %d of %d" % (i, numInteractions))
        batchInformation = data_provider.get_next_batch_with_filenames()
        data = batchInformation[2]
        filenames = batchInformation[3]
        
        result = RunModel(model, data)
        
        newDatasetX[numExamplesPerBatch * i: numExamplesPerBatch * (i+1), :] = result
        newDatasetY[numExamplesPerBatch * i: numExamplesPerBatch * (i+1)] = data[1].squeeze()
        allFilenames = allFilenames + filenames
    
    return newDatasetX, newDatasetY, allFilenames

def GetNewRepresentationFromAllPatches(data_provider, model):
    #Get test data
    batchInformation = data_provider.get_patches_and_filenames()
    data = batchInformation[2]
    filenames = batchInformation[3]
    result = RunModel(model, data)
    return result, data[1], filenames

def RunModel(model, data):
    #Run the model to obtain the output of the layer Local4 (last one before softmax)
    local4 = numpy.zeros((data[0].shape[1], model.layers[10]['outputs']), dtype=numpy.single)
    newData = [data[0], data[1], local4]
    layer_idx = model.get_layer_idx('local4')
    model.libmodel.startFeatureWriter(newData, layer_idx)
    results = model.finish_batch()
    return local4       #the C++ model updates this variable

def LoadModel(modelName):
    program_name = sys.argv[0]
    sys.argv = [program_name]
    sys.argv.append('-f')
    sys.argv.append(modelName)
    sys.argv.append('--test-only=1')
    sys.argv.append('--gpu=0')
    op = ConvNet.get_options_parser()
    op,load_dic= IGPUModel.parse_options(op)
    return ConvNet(op,load_dic)

if __name__ == '__main__':
    data_path = '/home/especial/vri/databases/preprocessados/micro_30pct'
    data_type = 'tree-cropped'
    
    model = LoadModel('executions/tree_micro_30pct_filter_5_patch_32/')
    numInteractions=10
    
    train_batch_range=[1]
    valid_batch_range=[2]
    test_batch_range=[3]
    
    dp_params = {'convnet': model, 'multiview_test': 0, 'patch_size': 32}
    
    train_provider = DataProvider.get_instance(data_path, train_batch_range, type=data_type, dp_params=dp_params, test=True)
    valid_provider = DataProvider.get_instance(data_path, valid_batch_range, type=data_type, dp_params=dp_params, test=True)
    test_provider = DataProvider.get_instance(data_path, test_batch_range, type=data_type, dp_params=dp_params, test=True)
    
    #train = GetNewRepresentationFromRandomPatches(numInteractions, train_provider, model)
    #valid = GetNewRepresentationFromRandomPatches(numInteractions, valid_provider, model)   
    train = GetNewRepresentationFromAllPatches(train_provider, model)
    valid = GetNewRepresentationFromAllPatches(valid_provider, model)
    test = GetNewRepresentationFromAllPatches(test_provider, model)
    
    numpy.save('micro_micro_30_5_32/train_grid_X.npy', train[0])
    numpy.save('micro_micro_30_5_32/train_grid_Y.npy', train[1])

    numpy.save('micro_micro_30_5_32/valid_grid_X.npy', valid[0])
    numpy.save('micro_micro_30_5_32/valid_grid_Y.npy', valid[1])

    #numpy.save('micro_micro_30_5_32/test_X.npy', test[0])
    #numpy.save('micro_micro_30_5_32/test_Y.npy', test[1])
    #numpy.save('micro_micro_30_5_32/test_filenames.npy', test[2])
