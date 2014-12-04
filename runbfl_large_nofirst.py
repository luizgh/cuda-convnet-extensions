from convnetmodel import *
import sys
import cPickle

patch_sizes = [48, 64, 96]

def run_for_params(filtersize, patchsize, gpu):
    model = ConvnetModel()
    basename = 'author_bfl_large_filter_%d_patch_%d' % (filtersize, patchsize)
    result=model.Run(filename = basename, 
          test_output_file = 'executions/%s.pickle' % basename,
          data_path = '/home/especial/vri/databases/preprocessados/BFL_115_large',
          save_path = 'executions', 
          train_range= '1', valid_range= '2', test_range= '3',
          layer_def_file='./layersbfl/filter_%d.cfg' % filtersize, 
          layer_params_file= './layersbfl/params-conv.cfg',
          data_provider = 'tree-cropped', patch_size= '%d' % patchsize, logfile= 'logs/%s.log' % basename,
          layer_params_finetuning='./layersbfl/params-conv2.cfg', finetuning_epochs='200',
          maxEpochs = '43000', gpu= gpu,
          iterations_to_wait = "9000")
    return result


if len(sys.argv) < 2:
    print("Usage: %d <filter> [-gpu=X] - run all models for the model with the selected filter size")
    exit(1)

filter_size = int(sys.argv[1])

gpu = '0'
if len(sys.argv) >= 3:
    gpu = sys.argv[2]
    if sys.argv[2][0:5] == '-gpu=':
        gpu = sys.argv[2][5:]

result = {}
result['fileAccuracy'] = 0.5
for patch_size in patch_sizes:
    sys.stdout.write("Running on gpu %s for: patch: %d, filter: %d: " % (gpu, patch_size, filter_size))
    sys.stdout.flush()
    result = run_for_params(filter_size, patch_size, gpu)
    print "Accuracy: %s" % result['fileAccuracy']
