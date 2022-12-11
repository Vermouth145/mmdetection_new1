import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as T
from PIL import Image
import torch


def tensor2img(ts):
    ts = ts.cpu()
    array = np.array(ts)
    # np.savetxt('sa.txt', array)
    array /= np.max(array)
    if ts.ndim == 4:
        imgs = [array[i] for i in range(array.shape[0])]
    else:
        raise ValueError(" no batch dimension!")

    for item in imgs:
        img = np.transpose(item, (1, 2, 0)) * 256  # 长宽通道数
        img = img.astype(np.int)
        plt.imshow(img)
        plt.savefig('../demo/{}-sa.png'.format(1))
        # plt.show()

def main():
    a = torch.load('../demo/sa.pt')
    # print(a)
    tensor2img(a)

if __name__ == '__main__':
    main()