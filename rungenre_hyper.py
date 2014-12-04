from convnetmodel import *
import cPickle
import os
import shutil


fold = 1
ranges = ['2','3','1']

train_range= ranges[0]
valid_range= ranges[1]
test_range= ranges[2]

result = []

filter_sizes = [3, 5, 7, 10, 12]
patch_sizes = [32, 48]

def run_for_params(filtersize, patchsize):
    for i in range(30):
        print "Running model %d of 30" % (i+1)
        model = ConvnetModel()
        basename = 'genre_filter_%d_patch_%d_part_%d' % (filtersize, patchsize, i)
        model.Run(filename = basename, 
              test_output_file = 'executions/%s.pickle' % basename,
              data_path = '/home/ppginf/lghafemann/nobackup/data/genreparts_folds/%d' % i,
              save_path = 'executions/genre/', 
              train_range= train_range, valid_range= valid_range, test_range= test_range,
              layer_def_file='./layersgenre/filter_%d.cfg' % filtersize, layer_params_file= './layers/genre-params.cfg',
              layer_params_finetuning='./layers/genre-params-finetuning.cfg', finetuning_epochs='30',
              data_provider = 'tree-cropped', patch_size= "%d" % patchsize, logfile= 'logs/%s.log' % basename)

for filter_size in filter_sizes:
    for patch_size in patch_sizes:
        sys.stdout.write("Running for: patch: %d, filter: %d: " % (patch_size, filter_size))
        sys.stdout.flush()
        os.makedirs('executions/genre')
        run_for_params(filter_size, patch_size)
        shutil.rmtree('executions/genre')
