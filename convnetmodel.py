import subprocess
import sys
import cPickle


class ConvnetModel:
    def Run(self, filename, test_output_file, data_path, save_path, 
             train_range, valid_range, test_range,
            layer_def_file, layer_params_file, data_provider,
            patch_size, logfile, iterations_to_wait = '20',  gpu='0', maxEpochs='5000', test_on_images='1'):

        test_freq='1'
        subprocess.check_call(["python", "convnetextension.py",
                        "--data-path=" + data_path, "--save-path="+save_path,
                        "--train-range="+train_range, "--valid-range="+valid_range,
                        "--test-range="+test_range, "--layer-def="+layer_def_file,
                        "--layer-params="+layer_params_file, "--data-provider=" + data_provider,
                        "--test-freq="+test_freq, "--epochs=" + maxEpochs, "--patch-size="+patch_size,
                        "--gpu="+gpu, "--filename=" + filename, "--test-output-file=" + test_output_file,
                        "--iterations-to-wait=" + iterations_to_wait,
                        "--test-on-images=" + test_on_images], stdout = open(logfile,'w'), stderr=sys.stderr)

        result = cPickle.load(open(test_output_file))
        return result
