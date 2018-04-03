from __future__ import division, print_function
#matplotlib inline
import numpy as np

from Net import input_gen
from Net.nets import Unet as Net
from Net.trainers import Trainer as Trainer
from Net import nets_util

generator = input_gen.SFSDataProvider()

net = Net(channels=generator.channels, n_class=3, layers=3, features_root=16)

trainer = Trainer(net, batch_size=2, optimizer="momentum", opt_kwargs=dict(momentum=0.2))

path = trainer.train(generator, "./unet_trained", training_iters=300, epochs=10, display_step=1)

x_test, y_test = generator(1)

prediction = net.predict(path, x_test)

# save 


# compute mae


