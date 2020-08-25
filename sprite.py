import os
import tqdm
import pickle
import numpy as np
import torch.utils.data
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import PIL
import functools
from torchvision import transforms
import pdb


def get_video_dataloader(data_root, args):
    base_dataset = VideoFolderDataset(data_root, cache=None, min_len=8)
    image_transforms = get_image_transforms(args)
    video_transforms = functools.partial(video_transform_fns, image_transforms=image_transforms)
    video_dataset = VideoDataset(base_dataset, 8, 1, video_transforms, 8, False)
    video_loader = DataLoader(video_dataset, batch_size=args.batch_size, drop_last=True,
                              num_workers=4, shuffle=True)
    return video_dataset, video_loader


def get_image_transforms(args):
    image_transforms = transforms.Compose([
        PIL.Image.fromarray,
        transforms.Resize(int(args.image_size)),
        transforms.ToTensor(),
        lambda x: x[:3, ::],
        transforms.Normalize((0.5, 0.5, .5), (0.5, 0.5, 0.5)),
    ])
    return image_transforms


def video_transform_fns(video, image_transforms):
    vid = []
    for im in video:
        vid.append(image_transforms(im))
    vid = torch.stack(vid).permute(1, 0, 2, 3)
    return vid


class VideoFolderDataset(torch.utils.data.Dataset):
    def __init__(self, folder, cache, min_len=32):
        dataset = ImageFolder(folder)
        self.total_frames = 0
        self.lengths = []
        self.images = []

        if cache is not None and os.path.exists(cache):
            with open(cache, 'rb') as f:
                self.images, self.lengths = pickle.load(f)
        else:
            for idx, (im, categ) in enumerate(
                    tqdm.tqdm(dataset, desc="Counting total number of frames")):
                img_path, _ = dataset.imgs[idx]
                shorter, longer = min(im.width, im.height), max(im.width, im.height)
                length = longer // shorter
                if length >= min_len:
                    self.images.append((img_path, categ))
                    self.lengths.append(length)

            if cache is not None:
                with open(cache, 'wb') as f:
                    pickle.dump((self.images, self.lengths), f)

        self.cumsum = np.cumsum([0] + self.lengths)
        print("Total number of frames {}".format(np.sum(self.lengths)))

    def __getitem__(self, item):
        path, label = self.images[item]
        im = PIL.Image.open(path)
        return im, label

    def __len__(self):
        return len(self.images)


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, video_length=16, every_nth=1, transform=None, trim_video_length=0, triplet=False):
        self.dataset = dataset
        self.video_length = video_length
        self.trim_video_length = trim_video_length
        self.every_nth = every_nth
        self.transforms = transform if transform is not None else lambda x: x
        self.triplet = triplet
        self.dataset_size = len(self)
    
    def _sample_video(self, index):
        video, target = self.dataset[index]
        video = np.array(video)
        horizontal = video.shape[1] > video.shape[0]
        shorter, longer = min(video.shape[0], video.shape[1]), max(video.shape[0], video.shape[1])
        _video_len = longer // shorter
        video_len = min(_video_len, self.trim_video_length) if self.trim_video_length > 0 else _video_len
        # videos can be of various length, we randomly sample sub-sequences
        if video_len >= self.video_length * self.every_nth:
            needed = self.every_nth * (self.video_length - 1)
            gap = video_len - needed
            start = 0 if gap == 0 else np.random.randint(0, gap, 1)[0]
            subsequence_idx = np.linspace(start, start + needed, self.video_length, endpoint=True, dtype=np.int32)
        elif video_len >= self.video_length:
            subsequence_idx = np.arange(0, self.video_length)
        else:
            raise Exception('Length is too short id - {}, len - {}'.format(self.dataset[index], video_len))
        frames = np.split(video, _video_len, axis=1 if horizontal else 0)
        selected = np.array([frames[s_id] for s_id in subsequence_idx])
        images = self.transforms(selected)
        # return data of shape [T, C, W, H]
        return images.transpose(0, 1), target

    def __getitem__(self, index):
        data_ancher, label_ancher = self._sample_video(index)
        if self.triplet:
            # Remark: in S3VAE, negative samples are sampled using action labels, which is NOT
            # strickly self-supervised. Here we just randomly sample another item as negative.
            index_neg = np.random.randint(self.dataset_size)
            data_neg, label_neg = self._sample_video(index_neg)
            permidx = np.random.permutation(data_ancher.shape[0])
            data_pos = data_ancher[permidx, ...]
            return {'images': data_ancher, 'labels': label_ancher,
                    'images_pos': data_pos, 'images_neg': data_neg, 'labels_neg': label_neg}
        return {'images': data_ancher, 'labels': label_ancher}

    def __len__(self):
        return len(self.dataset)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transforms = transform if transform is not None else lambda x: x

    def __getitem__(self, item):
        if item != 0:
            video_id = np.searchsorted(self.dataset.cumsum, item) - 1
            frame_num = item - self.dataset.cumsum[video_id] - 1
        else:
            video_id = 0
            frame_num = 0

        video, target = self.dataset[video_id]
        video = np.array(video)

        horizontal = video.shape[1] > video.shape[0]

        if horizontal:
            i_from, i_to = video.shape[0] * frame_num, video.shape[0] * (frame_num + 1)
            frame = video[:, i_from: i_to, ::]
        else:
            i_from, i_to = video.shape[1] * frame_num, video.shape[1] * (frame_num + 1)
            frame = video[i_from: i_to, :, ::]

        if frame.shape[0] == 0:
            print('video {}. From {} to {}. num {}'.format(video.shape, i_from, i_to, item))

        return {'images': self.transforms(frame), 'labels': target}

    def __len__(self):
        return self.dataset.cumsum[-1]
