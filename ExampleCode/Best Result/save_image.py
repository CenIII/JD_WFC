import tensorflow as tf
import skimage.data
import os
from PIL import Image
import numpy as np

#from test import load_data

from custom_layers_unet_new2 import unet 

def load_data(data_dir):
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
    
color_test_dir = './test/color'
color_test,order = load_data(color_test_dir)

images_test = np.zeros((len(color_test),128,128,2),dtype='f')
images_test[...,0] = color_test[...,2]

mask_test_dir = './test/mask/'
mask_test,_ = load_data(mask_test_dir)
images_test[...,1] = mask_test


normal_test_dir = './test/normal/'


test_samples = len(color_test)

x = tf.placeholder(tf.float32, shape=(None, 128,128,2)) 

_,output,_,_ = unet(x)

# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()


# Add ops to save and restore all the variables.
saver = tf.train.Saver()



# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  saver = tf.train.Saver()
  saver.restore(sess, "/tmp/tmp2/model_73000.ckpt")
  print("Model restored.")
  
  for i in range(test_samples):
      x_batch = np.zeros((1,128,128,2),dtype='f')
      x_batch[0,...,0] = images_test[i,...,0]
      x_batch[0,...,1] = mask_test[i]
      image_tensor = sess.run([output], feed_dict={x: x_batch})
      image_tensor  = np.asarray(image_tensor)
      # Normalize output surface normal
      image = image_tensor[0,0,...]
      image = (image/255.0-0.5)*2
      norm_factor = np.square(image).sum(axis=2)
      image= np.divide(image,np.expand_dims(np.sqrt(norm_factor),-1))
      image = (image/2+0.5)*255
      
      mask = mask_test[i]
      image = np.multiply(image,np.expand_dims(mask/255,-1))

      f = Image.fromarray(image.astype(np.uint8))
      f.save(os.path.join(normal_test_dir, order[i]))
  print("Complete!")
      
      
