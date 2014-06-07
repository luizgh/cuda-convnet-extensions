from convnetmodel import *
import cPickle

model = ConvnetModel()

result = []
cPickle.dump(result, open('executions/genre/full_output.pickle','w'))

for i in range(30):
    print "Running model %d of 30" % (i+1)
    result.append(model.Run(filename = 'genre_%d' % i, 
          test_output_file = 'executions/genre/output_%d.pickle' % i,
          data_path = '/home/ppginf/lghafemann/nobackup/data/genreparts/%d' % i,
          save_path = 'executions/genre', 
          train_range= '1', valid_range= '2', test_range= '3',
          layer_def_file='./layers/genre-conv.cfg', layer_params_file= './layers/genre-params.cfg',
          data_provider = 'tree-cropped', patch_size= '48', logfile= 'logs/genre_split_%d' % i))


cPickle.dump(result, open('executions/genre/full_output.pickle','w'))
