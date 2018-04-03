# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.


'''
Toy example, generates images at random that can be used for training

Created on Jul 28, 2016

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals
import os
import skimage.data
import numpy as np
from Net.input_util import BaseDataProvider, normalize_d2f
from skimage.filters import sobel #, scharr, prewitt,roberts 


class GrayScaleDataProvider(BaseDataProvider):
    channels = 1
    n_class = 2
    
    def __init__(self, nx, ny, **kwargs):
        super(GrayScaleDataProvider, self).__init__()
        self.nx = nx
        self.ny = ny
        self.kwargs = kwargs
        rect = kwargs.get("rectangles", False)
        if rect:
            self.n_class=3
        
    def _next_data(self):
        return create_image_and_label(self.nx, self.ny, **self.kwargs)

class RgbDataProvider(BaseDataProvider):
    channels = 3
    n_class = 2
    
    def __init__(self, nx, ny, **kwargs):
        super(RgbDataProvider, self).__init__()
        self.nx = nx
        self.ny = ny
        self.kwargs = kwargs
        rect = kwargs.get("rectangles", False)
        if rect:
            self.n_class=3

        
    def _next_data(self):
        data, label = create_image_and_label(self.nx, self.ny, **self.kwargs)
        return to_rgb(data), label

class SFSDataProvider(object):
    channels = 3
    def __init__(self):
        # super(SFSDataProvider, self).__init__()
        self._init_data_counter()
        self.images, self.mask, self.normal =self._load_and_format_data()
        self.image_num = self.images.shape[0]
    def _load_and_format_data(self):
        color_dir = '/Users/CenIII/Onedrive/School/Umich/442 Computer Vision/HW/PJDATA/minitrain/color/'
        color,_ = self._load_data(color_dir)

        mask_dir = '/Users/CenIII/Onedrive/School/Umich/442 Computer Vision/HW/PJDATA/minitrain/mask/'
        mask,_ = self._load_data(mask_dir)

        normal_dir = '/Users/CenIII/Onedrive/School/Umich/442 Computer Vision/HW/PJDATA/minitrain/normal/'
        normal,_ = self._load_data(normal_dir)

        images = np.zeros((len(color),128,128,3),dtype='f')
        images[...,0] = normalize_d2f(color[...,2])
        images[...,1] = normalize_d2f(mask)
        for i in range(len(color)):
            images[i,:,:,2] = sobel(images[i,:,:,0])
        # mask_ = np.zeros((len(color),128,128,1),dtype='f')
        # mask_[...,0] = mask != 0
        normal = normalize_d2f(normal)
        return images, mask, normal

    def _load_data(self, data_dir):
        data_ = []
        file_order = []
        file_names = [os.path.join(data_dir, f)
            for f in os.listdir(data_dir)]
        file_order =  [ f
            for f in os.listdir(data_dir)]
        for f in file_names:
            data_.append(skimage.data.imread(f))
        data_ = np.array(data_,dtype='f')
        return data_, file_order

    def _init_data_counter(self):
        self.data_counter=0

    def _next_data(self):
        data = self.images[self.data_counter]
        label = self.normal[self.data_counter]
        self.data_counter = (self.data_counter+1)%self.image_num
        return data, label

    def __call__(self, n):
        train_data, labels = self._next_data()
        ix,iy,iz = train_data.shape
        ox,oy,oz = labels.shape
        X = np.zeros((n, ix, iy, iz))
        Y = np.zeros((n, ox, oy, oz))
        X[0] = train_data
        Y[0] = labels
        for i in range(1, n):
            train_data, labels = self._next_data()
            X[i] = train_data
            Y[i] = labels
        return X, Y

def create_image_and_label(nx,ny, cnt = 10, r_min = 5, r_max = 50, border = 92, sigma = 20, rectangles=False):
    
    
    image = np.ones((nx, ny, 1))
    label = np.zeros((nx, ny, 3), dtype=np.bool)
    mask = np.zeros((nx, ny), dtype=np.bool)
    for _ in range(cnt):
        a = np.random.randint(border, nx-border)
        b = np.random.randint(border, ny-border)
        r = np.random.randint(r_min, r_max)
        h = np.random.randint(1,255)

        y,x = np.ogrid[-a:nx-a, -b:ny-b]
        m = x*x + y*y <= r*r
        mask = np.logical_or(mask, m)

        image[m] = h

    label[mask, 1] = 1
    
    if rectangles:
        mask = np.zeros((nx, ny), dtype=np.bool)
        for _ in range(cnt//2):
            a = np.random.randint(nx)
            b = np.random.randint(ny)
            r =  np.random.randint(r_min, r_max)
            h = np.random.randint(1,255)
    
            m = np.zeros((nx, ny), dtype=np.bool)
            m[a:a+r, b:b+r] = True
            mask = np.logical_or(mask, m)
            image[m] = h
            
        label[mask, 2] = 1
        
        label[..., 0] = ~(np.logical_or(label[...,1], label[...,2]))
    
    image += np.random.normal(scale=sigma, size=image.shape)
    image -= np.amin(image)
    image /= np.amax(image)
    
    if rectangles:
        return image, label
    else:
        return image, label[..., 1]




def to_rgb(img):
    img = img.reshape(img.shape[0], img.shape[1])
    img[np.isnan(img)] = 0
    img -= np.amin(img)
    img /= np.amax(img)
    blue = np.clip(4*(0.75-img), 0, 1)
    red  = np.clip(4*(img-0.25), 0, 1)
    green= np.clip(44*np.fabs(img-0.5)-1., 0, 1)
    rgb = np.stack((red, green, blue), axis=2)
    return rgb

