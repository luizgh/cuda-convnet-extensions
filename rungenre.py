from convnetmodel import *
import cPickle

model = ConvnetModel()

fold = 1
ranges = ['2','3','1']

train_range= ranges[0]
valid_range= ranges[1]
test_range= ranges[2]

result = []
cPickle.dump(result, open('executions/genre_fold%d/full_output.pickle' % fold,'w'))

for i in [15]:
    print "Running model %d of 30" % (i+1)
    result.append(model.Run(filename = 'genre_%d' % i, 
          test_output_file = 'executions/genre_fold%d/output_%d.pickle' % (fold,i),
          data_path = '/home/ppginf/lghafemann/nobackup/data/genreparts_folds/%d' % i,
          save_path = 'executions/genre_fold%d' % fold, 
          train_range= train_range, valid_range= valid_range, test_range= test_range,
          layer_def_file='./layers/genre-conv.cfg', layer_params_file= './layers/genre-params.cfg',
          layer_params_finetuning='./layers/genre-params-finetuning.cfg', finetuning_epochs='30',
          data_provider = 'tree-cropped', patch_size= '48', logfile= 'logs/genre_fold%d_split_%d' % (fold,i)))


cPickle.dump(result, open('executions/genre_fold%d/full_output.pickle' % fold,'w'))
