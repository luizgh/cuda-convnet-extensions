from convnetmodel import *
import sys
import cPickle


def run_for_params(pct, filtersize, patchsize, gpu):
    model = ConvnetModel()
    basename = 'tree_micro_%dpct_filter_%d_patch_%d_gpu%s' % (pct, filtersize, patchsize, gpu)
    result=model.Run(filename = basename, 
          test_output_file = 'executions/%s.pickle' % basename,
          data_path = '/home/especial/vri/databases/preprocessados/micro_%dpct' % pct,
          save_path = 'executions', 
          train_range= '1', valid_range= '2', test_range= '3',
          layer_def_file='./layersmicro/filter_%d.cfg' % filtersize, 
          layer_params_file= './layersmicro/treall-params-conv.cfg',
          data_provider = 'tree-cropped', patch_size= '%d' % patchsize, logfile= 'logs/%s.log' % basename,
          layer_params_finetuning='./layersmicro/treall-params-conv2.cfg', finetuning_epochs='20',
          maxEpochs = '7000', gpu= gpu,
          iterations_to_wait = "1000")
    return result


if len(sys.argv) < 4:
    print("Usage: %d <percent> <filter> <patch size> [gpu] - run all models for the model with <percent> size of the original image")
    exit(1)

pct = int(sys.argv[1])
filter_size = int(sys.argv[2])
patch_size = int(sys.argv[3])

gpu = '0'
if len(sys.argv) >= 5:
    gpu = sys.argv[4]
    if sys.argv[4][0:5] == '-gpu=':
        gpu = sys.argv[4][5:]
sys.stdout.write("Running on gpu %s for: Pct: %d, patch: %d, filter: %d: " % (gpu, pct, patch_size, filter_size))
sys.stdout.flush()
result = run_for_params(pct, filter_size, patch_size, gpu)
print "Accuracy: %s" % result['fileAccuracy']
