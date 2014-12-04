from convnetmodel import *

model = ConvnetModel()

result=model.Run(filename = 'genre_2ndt', 
          test_output_file = 'executions/genre_2nd_testoutput.pickle',
          data_path = '/home/ppginf/lghafemann/nobackup/data/Genre_first_part',
          save_path = 'executions', 
          train_range= '1', valid_range= '2', test_range= '3',
          layer_def_file='./layers/genre-conv.cfg', layer_params_file= './layers/genre-params.cfg',
          layer_params_finetuning='./layers/genre-params-finetuning.cfg', finetuning_epochs='2',
          data_provider = 'tree-cropped', patch_size= '48', logfile= 'logs/genre_split')

print "Accuracy on files: %f" % result['fileAccuracy']

