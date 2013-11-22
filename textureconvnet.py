from convnet import ConvNet
from gpumodel import IGPUModel
from MLTools import ModelEvaluation
import numpy 

class TextureConvNet(ConvNet):
    def get_test_patches_and_filenames(self):
        epoch, batchNum, data, filenames = self.test_data_provider.get_patches_and_filenames()
        return data, filenames
        
    def get_predictions(self, data):
        num_classes = self.test_data_provider.get_num_classes()
        
        preds = numpy.zeros((data[0].shape[1], num_classes), dtype=numpy.single)
        data += [preds]

        # Run the model
        softmax_idx = self.get_layer_idx('probs', check_type='softmax')
        self.libmodel.startFeatureWriter(data, softmax_idx)
        results = self.finish_batch()

        return preds, results
        
    def get_test_error(self):
        data, filenames = self.get_test_patches_and_filenames()
        
        probabilities, test_results = self.get_predictions(data)
               
        fileProbs, fileLabels, fileIDs = ModelEvaluation.GetUnormalizedJointLogProbability(probabilities, data[1].reshape(-1), filenames)
        filePredictions = ModelEvaluation.GetPredictions(fileProbs)
        fileAccuracy, misclassifiedFiles = ModelEvaluation.CalculateAccuracy(filePredictions, fileLabels)
        
        nExamples = test_results[1]
        results =  ({'logprob' : [test_results[0]['logprob'][0], fileAccuracy * test_results[1]]}, test_results[1])
        if self.test_only: # Print the individual batch results for safety
            print str(results)
        return results
       

if __name__ == "__main__":
    op = ConvNet.get_options_parser()

    op, load_dic = IGPUModel.parse_options(op)
    model = TextureConvNet(op, load_dic)
    model.start()
