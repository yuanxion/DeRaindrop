import os
from skimage import io
import matplotlib.pyplot as plt

plt.ion()
fig = plt.figure()
for i in range(20):
    path = 'data/train/data/{}_rain.png'.format(i)
    # print('Iter ', i, path)
    os.system('ls -alh {}'.format(path))

    image = io.imread(path)
    plt.imshow(image)
    plt.show
    plt.pause(0.3)

plt.ioff()
