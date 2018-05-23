import os
import os.path
import cv2
import numpy as np

import torch
import torch.utils.data as data


def random_crop(tensor):
    """Taks input tensor and random cropped tensor (3, 16, 112, 112)"""

    margin_h = tensor.size()[2] - 112
    margin_w = tensor.size()[3] - 112

    return tensor[:,:,0:112,0:112] # specify crop size


def sampling_frame(tensor):
    """Takes 3D tensor and sampling 16 frame tensors"""

    # print(tensor.size()) # (3, d, 240, 320)
    num_sample = 16
    num_frm = tensor.size()[1]
    start = np.random.randint(0, (num_frm - num_sample + 1))
    out = tensor[:,start:start+num_sample,:,:]

    return random_crop(out)


def split_video(video_path):
    """Split videos and return image lists"""

    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    success = True
    img_list = []
    while success:
        success, image = vidcap.read()
        img_list.append(image)
        count += 1

    img_list = [x for x in img_list if x is not None] # remove 'None'

    return img_list


def expand_dim(array):
    """Expand dimension on the left side of the vector"""

    array = np.expand_dims(array, axis=0)
    array = np.expand_dims(array, axis=0)

    return array


def video_to_volume(img_list):
    """Takes video and """

    num_frm = len(img_list)
    h, w, c = img_list[0].shape[0], img_list[0].shape[1], img_list[0].shape[2]
    out = torch.Tensor(3, len(img_list), h, w)

    i = 0
    for img in img_list:

        c1 = expand_dim(img[:,:,0]) / 255
        c2 = expand_dim(img[:,:,1]) / 255
        c3 = expand_dim(img[:,:,2]) / 255

        out[0,i,:,:] = torch.from_numpy(c1[:,:,:,:])
        out[1,i,:,:] = torch.from_numpy(c2[:,:,:,:])
        out[2,i,:,:] = torch.from_numpy(c3[:,:,:,:])

        i += 1

    return out

#####################################

# data loader for video
VID_EXTENSIONS = ['.avi']

def is_video_file(filename):

    return any(filename.endswith(extension) for extension in VID_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    vids = []
    dir = os.path.expanduser(dir) # replace dir as a absolute path

    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_video_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    vids.append(item)

    return vids


def default_loader(path):
    """
    Args: path (string): Video path
    Returns: Video tensor
    """

    img_list = split_video(path)  # type: list
    vol = video_to_volume(img_list)
    sample_vol = sampling_frame(vol)
    # print(sample_vol.size())

    return sample_vol



class VideoFolder(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader):
        """
        Args:
            root (string): Root directory path.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
            loader (callable, optional): A function to load an image given its path.
        Attributes:
            classes (list): List of the class names.
            class_to_idx (dict): Dict with items (class_name, class_index).
            imgs (list): List of (image path, class_index) tuples
        """
        classes, class_to_idx = find_classes(root)
        vids = make_dataset(root, class_to_idx)

        print('vids:', vids)

        if len(vids) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                 "Supported video extensions are: " + ",".join(VID_EXTENSIONS)))

        self.root = root
        self.vids = vids
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args: index (int): Index
        Returns: tuple: (image, target) where target is class_index of the target class
        """
        
        path, target = self.vids[index]
        vid = self.loader(path)

        # if self.transform is not None:
        #     vid = self.transform(vid)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return vid, target

    def __len__(self):
        return len(self.vids)