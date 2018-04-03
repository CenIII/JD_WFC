from __future__ import division, print_function
#matplotlib inline
import numpy as np

from Net import input_gen
from Net.nets import Unet as Net
from Net.trainers import Trainer as Trainer
from Net import nets_util

nx = 572
ny = 572

generator = input_gen.GrayScaleDataProvider(nx, ny, cnt=20)

x_test, y_test = generator(1)

net = Net(channels=generator.channels, n_class=generator.n_class, layers=3, features_root=16)

trainer = Trainer(net, optimizer="momentum", opt_kwargs=dict(momentum=0.2))

path = trainer.train(generator, "./unet_trained", training_iters=20, epochs=10, display_step=2)

x_test, y_test = generator(1)

prediction = net.predict("./unet_trained/model.cpkt", x_test)

