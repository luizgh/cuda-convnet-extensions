from generateNewDatasets import *
import sys

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: %s NumRandomPatches" % sys.argv[0])
        exit(1)
    numInteractions = int(sys.argv[1])
    print("Creating dataset with %d random patches" % numInteractions)

    data_path = '/home/ppginf/lghafemann/nobackup/data/brodatz_1batch'
    data_type = 'tree-cropped'
    
    model = LoadModel('executions/tree_macro_gray_10pct_30images_filter_12_patch_32')
    #numInteractions=10
    
    train_batch_range=[1]
    valid_batch_range=[2]
    test_batch_range=[3]
    
    dp_params = {'convnet': model, 'multiview_test': 0, 'patch_size': 32}
    
    train_provider = DataProvider.get_instance(data_path, train_batch_range, type=data_type, dp_params=dp_params, test=True)
    valid_provider = DataProvider.get_instance(data_path, valid_batch_range, type=data_type, dp_params=dp_params, test=True)
    test_provider = DataProvider.get_instance(data_path, test_batch_range, type=data_type, dp_params=dp_params, test=True)
    
    train = GetNewRepresentationFromRandomPatches(numInteractions, train_provider, model)
    valid = GetNewRepresentationFromRandomPatches(numInteractions, valid_provider, model)   
    test = GetNewRepresentationFromRandomPatches(numInteractions, test_provider, model)   
    
    numpy.save('brodatz_macro_gray_12_32__%d_patches/train_grid_X.npy' % numInteractions, train[0])
    numpy.save('brodatz_macro_gray_12_32__%d_patches/train_grid_Y.npy' % numInteractions, train[1])

    numpy.save('brodatz_macro_gray_12_32__%d_patches/valid_grid_X.npy' % numInteractions, valid[0])
    numpy.save('brodatz_macro_gray_12_32__%d_patches/valid_grid_Y.npy' % numInteractions, valid[1])

    numpy.save('brodatz_macro_gray_12_32__%d_patches/test_X.npy' % numInteractions, test[0])
    numpy.save('brodatz_macro_gray_12_32__%d_patches/test_Y.npy' % numInteractions, test[1])
    numpy.save('brodatz_macro_gray_12_32__%d_patches/test_filenames.npy' % numInteractions, test[2])
