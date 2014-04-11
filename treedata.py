from data import *
import numpy
import numpy.random as nr
import numpy as n
import random as r

class CroppedTREEDataProvider(LabeledMemoryDataProvider):
    def __init__(self, data_dir, batch_range=None, init_epoch=1, init_batchnum=None, dp_params=None, test=False):
        LabeledMemoryDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)

        #Get image size, and if colors are being used
        size = self.batch_meta['data_shape']
        if len(size) > 2:
            self.colors = size[2]
        else:
            self.colors=1

        self.size_r, self.size_c = size[0], size[1]
        
        #Get command-line parameters
        self.inner_size = dp_params['patch_size']
        self.border_size_r = (self.size_r - self.inner_size) / 2
        self.border_size_c = (self.size_c - self.inner_size) / 2
        self.multiview = dp_params['multiview_test'] and test
        self.num_views = 5*2
        self.data_mult = self.num_views if self.multiview else 1
        self.num_colors = self.colors
        
        for d in self.data_dic:
            d['data'] = n.require(d['data'], requirements='C')
            d['labels'] = n.require(n.tile(d['labels'].reshape((1, d['data'].shape[1])), (1, self.data_mult)), requirements='C')
        
        self.cropped_data = [n.zeros((self.get_data_dims(), self.data_dic[0]['data'].shape[1]*self.data_mult), dtype=n.single) for x in xrange(2)]

        self.batches_generated = 0
        self.data_mean = self.batch_meta['data_mean'].reshape((self.colors,self.size_r,self.size_c))[:,self.border_size_r:self.border_size_r+self.inner_size,self.border_size_c:self.border_size_c+self.inner_size].reshape((self.get_data_dims(), 1))

        #For test, extract the grid patches of the image
        if self.test:
            data = self.data_dic[0]
            X = data['data']
            y = data['labels']
            filenames = data['filenames']

            patchSize = self.inner_size
            nPatches_r = self.size_r / patchSize
            nPatches_c = self.size_c / patchSize

            TotalNumberOfImages = X.shape[1]
            TotalNumberOfPatches = TotalNumberOfImages * nPatches_r* nPatches_c

            newX = numpy.zeros((self.colors*patchSize*patchSize, TotalNumberOfPatches))
            newY = numpy.zeros((1, TotalNumberOfPatches))
            patchFilenames = []

            currentPatch = 0
            for i in range (TotalNumberOfImages):
                item = X[:,i].reshape(self.colors,self.size_r,self.size_c)
                for row in range(nPatches_r):
                    for col in range(nPatches_c):
                        patch = item[:,row*patchSize:(row+1)*patchSize, col*patchSize: (col+1)*patchSize]
                        newX[:,currentPatch] = patch.reshape(-1)
                        newY[:,currentPatch] = y[:,i]
                        patchFilenames.append(filenames[i])
                        currentPatch +=1

            newX = newX - self.data_mean
            newX = numpy.require(newX, dtype=numpy.float32, requirements='C')
            newY = numpy.require(newY, dtype=numpy.float32, requirements='C')

            self.testPatches = [newX, newY]
            self.testPatchFilenames = patchFilenames
    
    def get_patches_and_filenames(self):
            epoch, batchnum = self.curr_epoch, self.curr_batchnum
            self.advance_batch()
            return epoch, batchnum, self.testPatches, self.testPatchFilenames

    def get_next_batch(self):
        epoch, batchnum, datadic = LabeledMemoryDataProvider.get_next_batch(self)

        cropped = n.zeros((self.get_data_dims(), datadic['data'].shape[1]), dtype = n.single)

        self.trim_borders(datadic['data'], cropped)
        cropped -= self.data_mean
        self.batches_generated += 1
        return epoch, batchnum, [cropped, datadic['labels']]

    def get_filename(self):
        return self.data_dic[0]['filenames']
        
    def get_data_dims(self, idx=0):
        return self.inner_size**2 * self.colors if idx == 0 else 1

    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return n.require((data + self.data_mean).T.reshape(data.shape[1], self.colors, self.inner_size, self.inner_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)
    
    def trim_borders(self, x, target):
        y = x.reshape(self.colors, self.size_r, self.size_c, x.shape[1])

        if self.test: # don't need to loop over cases
            pic = y[:,self.border_size_r:self.border_size_r+self.inner_size,self.border_size_c:self.border_size_c+self.inner_size, :] # just take the center for now
            target[:,:] = pic.reshape((self.get_data_dims(), x.shape[1]))
        else:
            for c in xrange(x.shape[1]): # loop over cases
                startY, startX = nr.randint(0,self.border_size_r*2 + 1), nr.randint(0,self.border_size_c*2 + 1)
                endY, endX = startY + self.inner_size, startX + self.inner_size
                pic = y[:,startY:endY,startX:endX, c]
                if nr.randint(2) == 0: # also flip the image with 50% probability
                    pic = pic[:,:,::-1]
                target[:,c] = pic.reshape((self.get_data_dims(),))


class TREEDataProvider(LabeledMemoryDataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        LabeledMemoryDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.data_mean = self.batch_meta['data_mean']
        self.num_colors = 3
        self.img_size = 64
        # Subtract the mean from the data and make sure that both data and
        # labels are in single-precision floating point.
        for d in self.data_dic:
            # This converts the data matrix to single precision and makes sure that it is C-ordered
            d['data'] = n.require((d['data'] - self.data_mean), dtype=n.single, requirements='C')
            print d['labels'].shape
            print d['data'].shape[1]
            d['labels'] = n.require(d['labels'].reshape((1, d['data'].shape[1])), dtype=n.single, requirements='C')

    def get_next_batch(self):
        epoch, batchnum, datadic = LabeledMemoryDataProvider.get_next_batch(self)
        return epoch, batchnum, [datadic['data'], datadic['labels']]

    # Returns the dimensionality of the two data matrices returned by get_next_batch
    # idx is the index of the matrix. 
    def get_data_dims(self, idx=0):
        return self.img_size**2 * self.num_colors if idx == 0 else 1
    
    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return n.require((data + self.data_mean).T.reshape(data.shape[1], 3, self.img_size, self.img_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)
