from convnet import *
import numpy 
sys.argv.append('-f')
sys.argv.append('bfl_115//')
sys.argv.append('--test-only=1')
sys.argv.append('--test-range=5')
op = ConvNet.get_options_parser()
op,load_dic= IGPUModel.parse_options(op)
model = ConvNet(op,load_dic)

#Get the next batch
a,b,data = LabeledMemoryDataProvider.get_next_batch(model.test_data_provider)
X = data['data']
y = data['labels']
filenames = data['filenames']

imgSize = int(numpy.sqrt(X.shape[0])) #divide by 3 if colored
patchSize=64 			
nPatches_1D = imgSize / patchSize

TotalNumberOfImages = X.shape[1]
TotalNumberOfPatches = TotalNumberOfImages * nPatches_1D* nPatches_1D

newX = numpy.zeros((patchSize*patchSize, TotalNumberOfPatches))
newY = numpy.zeros((1, TotalNumberOfPatches))
patchFilenames = []

currentPatch = 0
for i in range (TotalNumberOfImages):
    item = X[:,i].reshape(256,256)
    for row in range(nPatches_1D):
        for col in range(nPatches_1D):
            patch = item[row*patchSize:(row+1)*patchSize, col*patchSize: (col+1)*patchSize]
            newX[:,currentPatch] = patch.reshape(-1)
            newY[:,currentPatch] = y[:,i]
            patchFilenames.append(filenames[i])
            currentPatch +=1

newX = newX - model.train_data_provider.data_mean
newX = numpy.require(newX, dtype=numpy.float32, requirements='C')
newY = numpy.require(newY, dtype=numpy.float32, requirements='C')

data = [newX, newY]

num_classes = model.test_data_provider.get_num_classes()

preds = n.zeros((data[0].shape[1], num_classes), dtype=n.single)
data += [preds]

# Run the model
softmax_idx = model.get_layer_idx('probs', check_type='softmax')
model.libmodel.startFeatureWriter(data, softmax_idx)
result = model.finish_batch()

from MLTools import ModelEvaluation

fileProbs, fileLabels, fileIDs = ModelEvaluation.AggregateProbabilities (preds, newY.reshape(-1), patchFilenames, numpy.sum)
filePredictions = ModelEvaluation.GetPredictions(fileProbs)
fileAccuracy, misclassifiedFiles = ModelEvaluation.CalculateAccuracy(filePredictions, fileLabels)


sampleFileIDs =[id[0:3] for id in fileIDs]

sampleProbs, sampleLabels, sampleIDs = ModelEvaluation.AggregateProbabilities (fileProbs, fileLabels, sampleFileIDs, numpy.sum)
samplePredictions = ModelEvaluation.GetPredictions(sampleProbs)
sampleAccuracy, misclassifiedSamples = ModelEvaluation.CalculateAccuracy(samplePredictions, sampleLabels)


#directly from patches
sampleFileIDs =[id[0:3] for id in patchFilenames]

sampleProbs, sampleLabels, sampleIDs = ModelEvaluation.AggregateProbabilities (preds, newY.reshape(-1), sampleFileIDs, numpy.sum)
samplePredictions = ModelEvaluation.GetPredictions(sampleProbs)
sampleAccuracy, misclassifiedSamples = ModelEvaluation.CalculateAccuracy(samplePredictions, sampleLabels)


