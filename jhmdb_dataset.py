
# coding: utf-8

# # This file is used to make a dataset for jhmdb dataset
# Some of the code is from the caffe version, but it will be adapted to PyTorch dataloader version
# 
# Get idea from Tube-CNN code
# 
# Each chip last several frames, say self.depth = 8 frames. Use self._curr_idx to indicate current begin frame.
# 
# Also, each chip is belong to one vidoe (jhmdb has 928 video) with one action label. 
# 
# Therefore, the __getitem__() will get the entire video. But I will use sampler to divide this video to a list of chips, each chip is formed by 8 frames.
# 

# Define a pytorch dataset

# In[1]:


# This code is for Caffe data loader of J-HMDB database
# 
import torch
import torch.utils.data as data
import numpy as np
import os.path
import scipy.io as sio
import cPickle
import glob
import cv2
from IPython.core.debugger import set_trace
import matplotlib.pyplot as plt
# get_ipython().magic(u'matplotlib inline')
class JHMDB_Dataset(data.Dataset):
    """
    To access the input images and target which is annotation of each image
    """
    def __init__(self, data_path, name, clip_shape, split=1, depth=8, stride = 1):
        '''
        find the dataset folder, 
        
        
        '''
        self._data_path = data_path
        self._name = name
        self._height = clip_shape[0]
        self._width = clip_shape[1]
        self._split = split - 1
        self._vddb = []
        self.depth = depth
        self.stride = stride
        self._num_classes = 22
        self._classes = ('__background__',  # always index 0
                         'brush_hair', 'catch', 'clap', 'climb_stairs', 'golf',
                         'jump', 'kick_ball', 'pick', 'pour', 'pullup', 'push',
                         'run', 'shoot_ball', 'shoot_bow', 'shoot_gun', 'sit',
                         'stand', 'swing_baseball', 'throw', 'walk', 'wave')
        cache_file = os.path.join(self._data_path, 'cache',
                                  'jhmdb_%d_%d_db.pkl' % (self._height, self._width))
        self._class_to_ind = dict(zip(self._classes, xrange(self._num_classes)))
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                self._vddb = cPickle.load(fid)
            print ('{} gt vddb loaded from {}'.format(self._name, cache_file))
        else:
            self._vddb = self._read_video_list()

            [self._load_annotations(v) for v in self._vddb]
#             set_trace()
            with open(cache_file, 'wb') as fid:
                cPickle.dump(self._vddb, fid, cPickle.HIGHEST_PROTOCOL)
        self._curr_idx = 0

        mean_file = os.path.join(self._data_path, 'cache',
                                 'mean_frame_{}_{}.npy'.format(self._height,
                                                               self._width))
        if os.path.exists(mean_file):
            self._mean_frame = np.load(mean_file)
        else:
            self._mean_frame = self.compute_mean_frame()

        if name == 'train':
            self._vddb = self.keeps(1)
        else:
            if name == 'val':
                self._vddb = self.keeps(2)
        self.minimum_length = self.get_minimum_length()
    
    def get_minimum_length(self):
        res = 40
        for video in self._vddb:
            frames = video['video']
            bboxes = video['gt_bboxes']
            res = np.amin([res, len(frames), len(bboxes)])
        return res
        
        
        
        
    def __getitem__(self, index):
        '''
        index: the index in the whole video list, so it is the video id
        return items by index in the dataset
        need output 6 variables:
        batch_video   : a list of chips, each chip has 8 frames, the step between two frame is determined by input
        batch_label   : a list of chips's frames
        batch_bboxes  : the bboxes of each frame 
        batch_idx     : the batch index of this batch
        
        '''
        # get the entire video
        video = self._vddb[index]
        # and return the ground truth, the bounding boxes and the class label
        frames = video['video'][:self.minimum_length]               # a list of frames for this video
        bbox = video['gt_bboxes'][:self.minimum_length, :]     # a list of ground truth bboxes
        label = video['gt_label']             # the class label of this video
        split_id = video['split']              # which split is, used for train, vali, test split
        
        
        
        
        
        
        num_frame = len(frames)
        num_boxes = len(bbox)
        if num_frame != num_boxes:
            print("error data, video and gt_bbox are not match in the number")
            print("video has %d frames, but gt_bbox has %d frames"%(num_frame, num_boxes))
            num_frame = np.minimum(num_frame, num_boxes)
        frame_depth = 8
        '''
        clips = []
        curr_idx = 0
        end_idx = 0
        num_clips = 0
        while num_frame - curr_idx >= frame_depth :
            end_idx = curr_idx + self.depth
            one_clip_frame = frames[curr_idx : end_idx] # get 8 frame chip
            one_clip_bboxes = bbox[curr_idx : end_idx]
            one_clip_label = label
            one_clip = {"clip_frames":one_clip_frame, 
                        "clip_bboxes": one_clip_bboxes,
                        "clip_labels":one_clip_label,
                       "clip_idx":[curr_idx, end_idx]}
            clips.append(one_clip)
            curr_idx += self.stride
            num_clips += 1
        # transfer to numpy
        print("==>")
        print(clips[0]['clip_frames'].shape)
        print("==<")
        '''
        num_clips = num_frame - frame_depth + 1 # 33
        clips_frames = torch.zeros([num_clips, frame_depth, 240, 320, 3])
        clips_bboxes = torch.zeros([num_clips, frame_depth, 5])
        clips_labels = torch.zeros([num_clips, frame_depth])
        clips_indice = torch.zeros([num_clips, frame_depth, 2])
        
        for idx in range(num_clips):
            if idx + 8 > num_frame:
                break
            clips_frames[idx,:,:,:, :] = torch.from_numpy(frames[idx: idx + 8] )
            clips_bboxes[idx,:,:]      = torch.from_numpy(bbox[idx: idx + 8])
            clips_labels[idx,:]       = torch.from_numpy(np.array([label] * frame_depth))
            clips_indice[idx,:,:]     = torch.from_numpy(np.array([[idx, idx+frame_depth]] * frame_depth ))

        # return frames[0:end_idx], bbox[0:end_idx], label, chips
        return clips_frames, clips_bboxes, clips_indice, clips_labels
                       
 
    
    @property
    def vddb(self):
        return self._vddb

    @property
    def size(self):
        return len(self._vddb)
    
#     def keeps(self, num):
#         result = []
#         if num == 1: # for train
            
#             for i in xrange(len(self.vddb)):
#                 train_test = 'train'
#                 for split_id in self.vddb[i]['split']:
#                     if split_id == 2:
#                         train_test = 'test'
#                 if train_test == 'test':
#                     continue
#                 result.append(self.vddb[i])
#         if num == 2:
#             for i in xrange(len(self.vddb)):
#                 for split_id in self.vddb[i]['split']:
#                     if split_id == 2:
#                         train_test = 'test'
#                 if train_test == 'test':
#                     result.append(self.vddb[i])
            
# #         for i in xrange(len(self.vddb)):
# #             if self.vddb[i]['split'][self._split] == num:
# #                 result.append(self.vddb[i])
#         return result
    def keeps(self, num):
        result = []
        for i in xrange(len(self.vddb)):
            if self.vddb[i]['split'][self._split] == num:
                result.append(self.vddb[i])
        return result
    
    def _read_video_list(self):
        """Read JHMDB video list from a text file."""

        vddb = []
        tmp = []
        for i in xrange(1, self._num_classes):    # loop all the classes
            # read split 1 data
            file_name = os.path.join(self._data_path, 'splits',
                                     '{}_test_split1.txt'.format(self._classes[i]))
            if not os.path.isfile(file_name):
                raise NameError('The video list file does not exists: ' + file_name)
            with open(file_name) as f:
                lines = f.readlines()
                # Example, in brush_hair_test_split1.txt, the first line is:
                # Aussie_Brunette_Brushing_Long_Hair_brush_hair_u_nm_np1_ba_med_3.avi 1

            for line in lines:
                split = np.zeros(3, dtype=np.uint8)
                p1 = line.find(' ')
                video_name = self._classes[i] + '/' + line[: p1 - 4]
                split[0] = int((line[p1 + 1:].strip()))
                if split[0] == 1:
                    trainOrTest = 'train'
                
                vddb.append({'video_name': video_name,
                             'split': split})
                tmp.append(video_name)

            # read split 2 data
            file_name = os.path.join(self._data_path,'splits',
                                     '{}_test_split2.txt'.format(self._classes[i]))
            if not os.path.isfile(file_name):
                raise NameError('The video list file does not exists: ' + file_name)
            with open(file_name) as f:
                lines = f.readlines()

            for line in lines:
                p1 = line.find(' ')
                video_name = self._classes[i] + '/' + line[: p1 - 4]
                try:
                    index = tmp.index(video_name)
                    vddb[index]['split'][1] = int((line[p1 + 1:].strip()))
                except ValueError:
                    tmp.append(video_name)
                    split = np.zeros(3, dtype=np.uint8)
                    split[1] = int((line[p1 + 1:].strip()))
                    vddb.append({'video_name': video_name,
                                 'split': split})
            
            # read split 3 data
            file_name = os.path.join(self._data_path,'splits',
                                     '{}_test_split3.txt'.format(self._classes[i]))
            if not os.path.isfile(file_name):
                raise NameError('The video list file does not exists: ' + file_name)
            with open(file_name) as f:
                lines = f.readlines()
                
            for line in lines:
                p1 = line.find(' ')
                video_name = self._classes[i] + '/' + line[: p1 - 4]
                try:
                    index = tmp.index(video_name)
                    vddb[index]['split'][2] = int((line[p1 + 1:].strip()))
                except ValueError:
                    tmp.append(video_name)
                    split = np.zeros(3, dtype=np.uint8)
                    split[2] = int((line[p1 + 1:].strip()))
                    vddb.append({'video_name': video_name,
                                 'split': split})

        return vddb
    
    def _load_annotations(self, video):
        """Read video annotations from text files.
        """
        gt_file = os.path.join(self._data_path, 'puppet_mask',
                               video['video_name'], 'puppet_mask.mat')
        if not os.path.isfile(gt_file):
            raise Exception(gt_file + 'does not exist.')
        masks = sio.loadmat(gt_file)['part_mask']
        # for the example file, the shape is (240, 320, 40) because it has 40 frames
        print(gt_file)
        gt_label = self._class_to_ind[video['video_name'][: video['video_name'].find("/")]]
        # get label of that class
        # video['video_name'][: video['video_name'].find("/") is used to return the class name
        depth = masks.shape[2]
        # depth is the number of frames in this video, here it is 40 for example.

        ratio, pixels = self.clip_reader(video['video_name'])
        # here in the example, ratio is [1,1], pixels is a (240, 320, 3) numpy array

        gt_bboxes = np.zeros((depth, 5), dtype=np.float32)
        
        for j in xrange(depth):
            mask = masks[:, :, j]
            (a, b) = np.where(mask > 0)
            y1 = a.min()
            y2 = a.max()
            x1 = b.min()
            x2 = b.max()

            gt_bboxes[j] = np.array([ x1 * ratio[1], y1 * ratio[0],
                                     x2 * ratio[1], y2 * ratio[0], gt_label])
        video['video'] = pixels
        video['gt_bboxes'] = gt_bboxes
        video['gt_label'] = gt_label
        
    def clip_reader(self, video_prefix):
        """Load frames in the clip.
        Using openCV to load the clip frame by frame.
        If specify the cropped size (crop_size > 0), randomly crop the clip.
          Args:
            index: Index of a video in the dataset.
          Returns:
            clip: A matrix (channel x depth x height x width) saves the pixels.
          """
        clip = []
        r1 = 0
        framepath = os.path.join(self._data_path, 'Rename_Images', video_prefix)
        # get the original images for this video, for the example, it is 40 images
        num_frames = len(glob.glob(framepath + '/*.png'))
        for i in xrange(num_frames):
            filename = os.path.join(
                self._data_path, 'Rename_Images', video_prefix,
                '%05d.png' % (i + 1))

            im = cv2.imread(filename)
            if r1 == 0:
                r1 = self._height / im.shape[0]
                r2 = self._width / im.shape[1]
            im = cv2.resize(im, None, None, fx=r2, fy=r1,
                            interpolation=cv2.INTER_LINEAR)
            # if the image size is not the same as setting,then resizing it
            clip.append(im)
        return [r1, r2], np.asarray(clip, dtype=np.uint8)
    
    
    def compute_mean_frame(self):
        sum_frame = np.zeros((self._height, self._width, 3), dtype=np.float32)
        num_frames = 0
        for db in self._vddb:
            curr_frame = np.sum(db['video'], dtype=np.float32, axis=0)
            sum_frame += curr_frame
            num_frames += db['video'].shape[0]
        sum_frame = sum_frame / num_frames
        np.save(os.path.join(self._data_path, 'cache',
                             'mean_frame_{}_{}.npy'.format(self._height,
                                                           self._width)),
                sum_frame)
        return sum_frame
    
    def get_anchors(self):
        base_anchors = np.load(
            self._data_path + '/cache/anchors_8_12.npy').transpose()
        bottom_height = int(np.ceil(self._height / 16.0))
        bottom_width = int(np.ceil(self._width / 16.0))
        shift_x = np.arange(0, bottom_width)
        shift_y = np.arange(0, bottom_height)
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = 12
        K = shifts.shape[0]
        all_anchors = (base_anchors.reshape((1, A, 4)) +
                       shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
        #grid_anchors = all_anchors.reshape(bottom_height, bottom_width, A, 4) # heli add, to get all the anchors info

        all_anchors = all_anchors.reshape((K * A, 4))
        # only keep anchors inside the image
        inds_inside = np.where(
            (all_anchors[:, 0] >= 0) &
            (all_anchors[:, 1] >= 0) &
            (all_anchors[:, 2] < bottom_width) &  # width
            (all_anchors[:, 3] < bottom_height)  # height
        )[0]
        
        #inds_inside_grid = np.where(      # heli add
        #    (grid_anchors[:, :, :, 0] >= 0) &
        #    (grid_anchors[:, :, :, 1] >= 0) &
        #    (grid_anchors[:, :, :, 2] < bottom_width) &  # width
        #    (grid_anchors[:, :, :, 3] < bottom_height)  # height
        #)
        #res3 = np.vstack(inds_inside_grid).transpose((1, 0))
        res = all_anchors[inds_inside]
        #res = np.hstack((res4, res3))
        
        
        
        return res, all_anchors, inds_inside, (A, bottom_height, bottom_width)
    
    def cluster_bboxes(self, length=8, anchors=9):
        data = np.empty((0, 2))
        for db in self._vddb:       # loop through all the videos
            boxes = db['gt_bboxes']
            l = boxes.shape[0] - length + 1
            for i in xrange(l):
                if not (boxes[i, 0] + length == boxes[i + length - 1, 0] + 1):
                    print('Invalid boxes!')
                    continue
                curr = np.mean(boxes[i: i + length, 1: 5], axis=0)
                x = (curr[2] - curr[0]) / 16
                y = (curr[3] - curr[1]) / 16
                data = np.vstack((data, np.array([x, y])))
        import sklearn.cluster
        [centers, b, _] = sklearn.cluster.k_means(data, anchors)    # use k-means clustering to 
                                                                    # learn 12 anchor boxes 

        import matplotlib.pyplot as plt
        plt.figure(1)
        c = np.linspace(0, 1, anchors)
        for i in xrange(anchors):
            flag = b == i
            plt.plot(data[flag, 0], data[flag, 1], 'o', color=plt.cm.RdYlBu(c[i]))
            plt.xlabel('width')
            plt.ylabel('height')
        # plt.show()
        plt.savefig(os.path.join(self._data_path,
                                 'anchors_{}_{}.png'.format(length, anchors)))
        # this plt is showing 12 clusters
        cx1 = centers[:, 0] / 2
        cx2 = centers[:, 1] / 2
        r = np.vstack((-cx1, -cx2, cx1, cx2))  # two corners, four coordinate values
        np.save(os.path.join(self._data_path,
                             'cache',
                             'anchors_{}_{}.npy'.format(length, anchors)), r)
    
    def __len__(self):
        return len(self._vddb)


# In[2]:


# test, create a dataset instance
'''
data_path = '/nfs/stak/users/heli/heli/datasets/data/jhmdb'
clip_shape=[240, 320]
jhmdb_train = JHMDB_Dataset(data_path, 'train', clip_shape, split=1)
jhmdb_test = JHMDB_Dataset(data_path, 'test', clip_shape, split=1, stride=8)
frames, bbox, label, chips = jhmdb_train[0]
frames2, bbox2, label2, chips2 = jhmdb_test[0]
print(len(chips), len(chips2),len(frames), len(frames2))
'''


# In[5]:


def JHMDB_dataloader(data_path):
    shuffle = False    
    batch_size = 2
    num_workers = 1
    chip_shape = [240, 320]
    pin_memory = True
    train_dataset = JHMDB_Dataset(data_path, 'train', chip_shape, split=1)
    test_dataset = JHMDB_Dataset(data_path, 'test', chip_shape, split=1, stride=8)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle = shuffle,
        sampler=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    dataloaders = {
        "train":train_loader,
        'test': test_loader
    }
    dataset_sizes = {"train":len(train_dataset),
                     "test":len(test_dataset)}
    anchors = train_dataset.get_anchors()
    return dataloaders, dataset_sizes, anchors
    


# In[8]:


# test dataloader
# dat, size3datasets =JHMDB_dataloader ('/nfs/stak/users/heli/heli/datasets/data/jhmdb')
# train = dat['train']
# test = dat['test']
# print(len(train),  len(test), size3datasets)

