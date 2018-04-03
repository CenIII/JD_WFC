# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 21:08:22 2017

@author: DENSO-ML4

@author: DJF group
"""
import os
import numpy as np
import skimage.data
import tensorflow as tf
from custom_layers_unet_new2 import unet
from skimage.filters import sobel #, scharr, prewitt,roberts 

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

def normalize_d2f(images):
    return (images/255.0-0.5)*2

color_dir = '/Users/CenIII/Onedrive/School/Umich/442 Computer Vision/HW/PJDATA/minitrain/color'
color,_ = load_data(color_dir)

mask_dir = '/Users/CenIII/Onedrive/School/Umich/442 Computer Vision/HW/PJDATA/minitrain/mask/'
mask,_ = load_data(mask_dir)

normal_dir = '/Users/CenIII/Onedrive/School/Umich/442 Computer Vision/HW/PJDATA/minitrain/normal/'
normal,_ = load_data(normal_dir)
normal = normalize_d2f(normal)

images_train = np.zeros((len(color),128,128,3),dtype='f')
images_train[...,0] = normalize_d2f(color[...,2])
images_train[...,1] = normalize_d2f(mask)
for i in range(len(color)):
    images_train[i,:,:,2] = sobel(images_train[i,:,:,0])


mask_ = np.zeros((len(color),128,128,1),dtype='f')
mask_[...,0] = mask != 0


# add edge to the original image
# images_train_new = np.zeros((len(color),128,128,2),dtype='f')


# for i in range(len(color)):
#     images_train_new [i,:,:,0] = images_train[i,:,:,0] 
#     images_train_new [...,1] = mask
#     images_train_new [...,2] = edge_scale*my_edge[i,:,:]
# images_train = images_train_new


# Define data size and batch size
n_samples = len(color)
batch_size = 20

# Define placeholders for input
x = tf.placeholder(tf.float32, shape=(None, 128,128,3)) 
y = tf.placeholder(tf.float32, shape=(None, 128,128,3))
z = tf.placeholder(tf.bool, shape=(None, 128,128,1))

output_pre,output,_,max_ = unet(x)
# Normalize output
output = (output/255.0-0.5)*2
output_mask = tf.abs(output) < 1e-5
output_no0 = tf.where(output_mask, 1e-5*tf.ones_like(output), output)
norm_factor = tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(output_no0),3)), -1)
norm_output = tf.divide(output_no0,norm_factor)


# Just XiaJB test
z_mask = z[...,0]
a11=tf.boolean_mask(tf.reduce_sum(tf.square(norm_output),3),z_mask)
a22=tf.boolean_mask(tf.reduce_sum(tf.square(y),3),z_mask)
a12=tf.boolean_mask(tf.reduce_sum(tf.multiply(y,norm_output),3),z_mask)
cos_dist = tf.clip_by_value(tf.where(tf.is_nan(a12 / tf.sqrt(tf.multiply(a11,a22))), -1*tf.ones_like(a12 / tf.sqrt(tf.multiply(a11,a22))), a12 / tf.sqrt(tf.multiply(a11,a22))), -1, 1)
loss1 = tf.reduce_mean(3.1415926/2-(cos_dist+tf.pow(cos_dist,3)/6+tf.pow(cos_dist,5)*3/40+tf.pow(cos_dist,7)*15/336+tf.pow(cos_dist,9)*105/3456))



output_pre = (output_pre/255.0-0.5)*2
output_mask_pre = tf.abs(output_pre) < 1e-5
output_no0_pre = tf.where(output_mask_pre, 1e-5*tf.ones_like(output_pre), output_pre)
norm_factor_pre = tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(output_no0_pre),3)), -1)
norm_output_pre = tf.divide(output_no0_pre,norm_factor_pre)


# Just XiaJB test
z_mask_pre = z[...,0]
a11_pre=tf.boolean_mask(tf.reduce_sum(tf.square(norm_output_pre),3),z_mask_pre)
a22_pre=tf.boolean_mask(tf.reduce_sum(tf.square(y),3),z_mask_pre)
a12_pre=tf.boolean_mask(tf.reduce_sum(tf.multiply(y,norm_output_pre),3),z_mask_pre)
cos_dist_pre = tf.clip_by_value(tf.where(tf.is_nan(a12_pre / tf.sqrt(tf.multiply(a11_pre,a22_pre))), -1*tf.ones_like(a12_pre / tf.sqrt(tf.multiply(a11_pre,a22_pre))), a12_pre / tf.sqrt(tf.multiply(a11_pre,a22_pre))), -1, 1)
loss2 = tf.reduce_mean(3.1415926/2-(cos_dist_pre+tf.pow(cos_dist_pre,3)/6+tf.pow(cos_dist_pre,5)*3/40+tf.pow(cos_dist_pre,7)*15/336+tf.pow(cos_dist_pre,9)*105/3456))

loss = loss1 + loss2

#correct_predictions = tf.equal(tf.multiply(y,z),tf.multiply(output,z))
#accuracy = tf.reduce_mean(tf.cast(correct_predictions,tf.float32))
opt_operation = tf.train.AdamOptimizer().minimize(loss)

# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

loss_val_min = 1e5

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

#restrict

os.environ["CUDA_VISIBLE_DEVICES"] = '0'   #指定第一块GPU可用
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 程序最多只能占用指定gpu50%的显存
config.gpu_options.allow_growth = True      #程序按需申请内存
sess = tf.Session(config = config)

with tf.Session() as sess:
  # Initialize Variables in graph
  # Restore variables from disk.
  saver = tf.train.Saver()
  #saver.save(sess, "./tmp/tmp2/model_58000.ckpt")
  #saver.restore(sess, "./tmp/tmp3/model_83000.ckpt")
  #print("Model restored.")

  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
  #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

  sess.run(tf.global_variables_initializer())
  # Gradient descent loop for 500 steps
  for i in range(1, 60000):
    # Select random minibatch
    indices = np.random.choice(n_samples, batch_size)
    x_batch, y_batch = images_train[indices], normal[indices]
    z_batch = mask_[indices]
    # Do gradient descent step
    _, loss_val = sess.run([opt_operation, loss], feed_dict={x: x_batch, y: y_batch, z:z_batch})
    init_op = tf.global_variables_initializer()
    print("step %d, training loss %g" % (i, loss_val))
    if i%50 == 0:
        print("step %d, training loss %g"%(i, loss_val))
    if i%1000 == 0:
        #accuracy_val = accuracy.eval({x: x_batch, y: y_batch, z:z_batch})
        if loss_val_min > loss_val:
            save_path = saver.save(sess, "./tmp/min.ckpt")
            loss_val_min = loss_val
            print("step %d, training loss %g, model stored" %(i, loss_val))
        #print(max_.eval())
    if i == 5000:
        save_path = saver.save(sess, "./tmp/tmp2/model_63000.ckpt")
    if i == 10000:
        save_path = saver.save(sess, "./tmp/tmp2/model_68000.ckpt")
    if i == 15000:
        save_path = saver.save(sess, "./tmp/tmp2/model_73000.ckpt")
    if i == 20000:
        save_path = saver.save(sess, "./tmp/tmp3/model_78000.ckpt")
    if i == 25000:
        save_path = saver.save(sess, "./tmp/tmp3/model_83000.ckpt")
    if i == 30000:
        save_path = saver.save(sess, "./tmp/tmp3/model_88000.ckpt")
    if i == 35000:
        save_path = saver.save(sess, "./tmp/tmp3/model_93000.ckpt")
    if i == 40000:
        save_path = saver.save(sess, "./tmp/tmp3/model_108000.ckpt")
    if i == 45000-1:
        save_path = saver.save(sess, "./tmp/model_83000.ckpt")

