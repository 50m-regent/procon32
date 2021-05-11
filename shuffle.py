import sys
import cv2
import numpy as np
import random as rand

def shuffle(path, split_range=[2, 17], rot_pat=[0, 1, 2, 3]):
    img = cv2.imread(path)
    height, width, _ = img.shape

    h_split = w_split = 9999.0
    while height % w_split:
        while not w_split.is_integer() or width % h_split:
            h_split = rand.randrange(split_range[0], split_range[1])
            w_split = height / width * h_split

    splitted = np.array([np.split(h, w_split, 0) for h in np.split(img, h_split, 1)])

    for i in range(len(splitted)):
        np.random.shuffle(splitted[i])
    for i in range(len(splitted.transpose(1, 0, 2, 3, 4))):
        np.random.shuffle(splitted.transpose(1, 0, 2, 3, 4)[i])
    
    for i in range(len(splitted)):
        for j in range(len(splitted[i])):
            splitted[i][j] = np.rot90(splitted[i][j], rand.choice(rot_pat))

    return width // h_split, np.concatenate([np.concatenate(h) for h in splitted], 1)

if __name__ == '__main__':
    size, img = shuffle(sys.argv[1], split_range=[3, 17], rot_pat=[0])
    cv2.imwrite(sys.argv[1].split('.')[0] + f'_puzzle_{size}.ppm', img)