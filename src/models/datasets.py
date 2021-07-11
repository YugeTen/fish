import copy
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class Poverty_Batched_Dataset(Dataset):
    """
    Batched dataset for Poverty. Allows for getting a batch of data given
    a specific domain index.
    """
    def __init__(self, dataset, split, batch_size, transform=None):
        self.split_array = dataset.split_array
        self.split_dict = dataset.split_dict
        split_mask = self.split_array == self.split_dict[split]
        self.split_idx = np.where(split_mask)[0]

        self.root = dataset.root
        self.no_nl = dataset.no_nl

        self.metadata_array = torch.stack([dataset.metadata_array[self.split_idx, i] for i in [0, 2]], -1)
        self.y_array = dataset.y_array[self.split_idx]

        self.eval = dataset.eval
        self.collate = dataset.collate
        self.metadata_fields = dataset.metadata_fields
        self.data_dir = dataset.data_dir

        self.transform = transform if transform is not None else lambda x: x

        domains = self.metadata_array[:, 1]
        self.domain_indices = [torch.nonzero(domains == loc).squeeze(-1)
                               for loc in domains.unique()]
        self.num_envs = len(domains.unique())

        self.domains = domains
        self.targets = self.y_array
        self.batch_size = batch_size

    def reset_batch(self):
        """Reset batch indices for each domain."""
        self.batch_indices, self.batches_left = {}, {}
        for loc, d_idx in enumerate(self.domain_indices):
            self.batch_indices[loc] = torch.split(d_idx[torch.randperm(len(d_idx))], self.batch_size)
            self.batches_left[loc] = len(self.batch_indices[loc])

    def get_batch(self, domain):
        """Return the next batch of the specified domain."""
        batch_index = self.batch_indices[domain][len(self.batch_indices[domain]) - self.batches_left[domain]]
        return torch.stack([self.transform(self.get_input(i)) for i in batch_index]), \
               self.targets[batch_index], self.domains[batch_index]

    def get_input(self, idx):
        """Returns x for a given idx."""
        img = np.load(self.root / 'images' / f'landsat_poverty_img_{self.split_idx[idx]}.npz')['x']
        if self.no_nl:
            img[-1] = 0
        img = torch.from_numpy(img).float()
        return img


    def __getitem__(self, idx):
        return self.transform(self.get_input(idx)), \
               self.targets[idx], self.domains[idx]

    def __len__(self):
        return len(self.targets)



class FMoW_Batched_Dataset(Dataset):
    """
    Batched dataset for FMoW. Allows for getting a batch of data given
    a specific domain index.
    """
    def __init__(self, dataset, split, batch_size, transform):
        self.split_array = dataset.split_array
        self.split_dict = dataset.split_dict
        split_mask = self.split_array == self.split_dict[split]
        split_idx = np.where(split_mask)[0]

        self.full_idxs = dataset.full_idxs[split_idx]
        self.chunk_size = dataset.chunk_size
        self.root = dataset.root

        self.metadata_array = dataset.metadata_array[split_idx]
        self.y_array = dataset.y_array[split_idx]

        self.eval = dataset.eval
        self.collate = dataset.collate
        self.metadata_fields = dataset.metadata_fields
        self.data_dir = dataset.data_dir
        self.transform = transform

        domains = dataset.metadata_array[split_idx, :2]
        self.domain_indices = [torch.nonzero((domains == loc).sum(-1) == 2).squeeze(-1)
                               for loc in domains.unique(dim=0)]
        self.domains = self.metadata_array[:, :2]
        self.num_envs = len(domains.unique(dim=0))

        self.targets = self.y_array
        self.batch_size = batch_size

    def reset_batch(self):
        """Reset batch indices for each domain."""
        self.batch_indices, self.batches_left = {}, {}
        for loc, d_idx in enumerate(self.domain_indices):
            self.batch_indices[loc] = torch.split(d_idx[torch.randperm(len(d_idx))], self.batch_size)
            self.batches_left[loc] = len(self.batch_indices[loc])

    def get_batch(self, domain):
        """Return the next batch of the specified domain."""
        batch_index = self.batch_indices[domain][len(self.batch_indices[domain]) - self.batches_left[domain]]
        return torch.stack([self.transform(self.get_input(i)) for i in batch_index]), \
               self.targets[batch_index], self.domains[batch_index]

    def get_input(self, idx):
        """Returns x for a given idx."""
        idx = self.full_idxs[idx]
        img = Image.open(self.root / 'images' / f'rgb_img_{idx}.png').convert('RGB')
        return img

    def __getitem__(self, idx):
        return self.transform(self.get_input(idx)), \
               self.targets[idx], self.domains[idx]

    def __len__(self):
        return len(self.targets)


class CivilComments_Batched_Dataset(Dataset):
    """
    Batched dataset for CivilComments. Allows for getting a batch of data given
    a specific domain index.
    """
    def __init__(self, train_data, batch_size=16):
        self.num_envs = 9 # civilcomments dataset has 8 attributes, plus 1 blank (no attribute)
        meta = torch.nonzero(train_data.metadata_array[:, :8] == 1)
        indices, domains = meta[:, 0],  meta[:, 1]
        blank_indices = torch.nonzero(train_data.metadata_array[:, :8].sum(-1) == 0).squeeze()
        self.domain_indices = [blank_indices] + [indices[domains == d] for d in domains.unique()]
        domain_indices_by_group = []
        for d_idx in self.domain_indices:
            domain_indices_by_group.append(d_idx[train_data.metadata_array[d_idx][:, -1]==0])
            domain_indices_by_group.append(d_idx[train_data.metadata_array[d_idx][:, -1]==1])
        self.domain_indices = domain_indices_by_group

        train_data._text_array = [train_data.dataset._text_array[i] for i in train_data.indices]
        self.metadata_array = train_data.metadata_array
        self.y_array = train_data.y_array
        self.data = train_data._text_array

        self.eval = train_data.eval
        self.collate = train_data.collate
        self.metadata_fields = train_data.metadata_fields
        self.data_dir = train_data.data_dir
        self.transform = train_data.transform

        self.data = train_data._text_array
        self.targets = self.y_array
        self.domains = self.metadata_array[:, :8]
        self.batch_size = batch_size

    def reset_batch(self):
        """Reset batch indices for each domain."""
        self.batch_indices, self.batches_left = {}, {}
        for loc, d_idx in enumerate(self.domain_indices):
            self.batch_indices[loc] = torch.split(d_idx[torch.randperm(len(d_idx))], self.batch_size)
            self.batches_left[loc] = len(self.batch_indices[loc])

    def get_batch(self, domain):
        """Return the next batch of the specified domain."""
        batch_index = self.batch_indices[domain][len(self.batch_indices[domain]) - self.batches_left[domain]]
        return torch.stack([self.transform(self.get_input(i)) for i in batch_index]), \
               self.targets[batch_index], self.domains[batch_index]

    def get_input(self, idx):
        """Returns x for a given idx."""
        return self.data[idx]

    def __getitem__(self, idx):
        return self.transform(self.get_input(idx)), \
               self.targets[idx], self.domains[idx]

    def __len__(self):
        return len(self.targets)


class GeneralWilds_Batched_Dataset(Dataset):
    """
    Batched dataset for Amazon, Camelyon and IwildCam. Allows for getting a batch of data given
    a specific domain index.
    """
    def __init__(self, train_data, batch_size=16, domain_idx=0):
        domains = train_data.metadata_array[:, domain_idx]
        self.domain_indices = [torch.nonzero(domains == loc).squeeze(-1) for loc in domains.unique()]
        train_data._input_array = [train_data.dataset._input_array[i] for i in train_data.indices]
        self.num_envs = len(domains.unique())

        self.metadata_array = train_data.metadata_array
        self.y_array = train_data.y_array
        self.data = train_data._input_array

        self.eval = train_data.eval
        self.collate = train_data.collate
        self.metadata_fields = train_data.metadata_fields
        self.data_dir = train_data.data_dir
        if 'iwildcam' in str(self.data_dir):
            self.data_dir = f'{self.data_dir}/train'
        self.transform = train_data.transform

        self.data = train_data._input_array
        self.targets = self.y_array
        self.domains = self.metadata_array[:, domain_idx]
        self.batch_size = batch_size

    def reset_batch(self):
        """Reset batch indices for each domain."""
        self.batch_indices, self.batches_left = {}, {}
        for loc, d_idx in enumerate(self.domain_indices):
            self.batch_indices[loc] = torch.split(d_idx[torch.randperm(len(d_idx))], self.batch_size)
            self.batches_left[loc] = len(self.batch_indices[loc])

    def get_batch(self, domain):
        """Return the next batch of the specified domain."""
        batch_index = self.batch_indices[domain][len(self.batch_indices[domain]) - self.batches_left[domain]]
        return torch.stack([self.transform(self.get_input(i)) for i in batch_index]), \
               self.targets[batch_index], self.domains[batch_index]

    def get_input(self, idx):
        """Returns x for a given idx."""
        if 'amazon' in self.data_dir:
            return self.data[idx]
        else:
            # All images are in the train folder
            img_path = f'{self.data_dir}/{self.data[idx]}'
            img = Image.open(img_path).convert('RGB')
            return img

    def __getitem__(self, idx):
        return self.transform(self.get_input(idx)), \
               self.targets[idx], self.domains[idx]

    def __len__(self):
        return len(self.targets)


class CDsprites_Dataset(Dataset):
    """
    Batched dataset for CDsprites. Allows for getting a batch of data given
    a specific domain index.
    """
    def __init__(self, dataset, data_dir, batch_size, num_domains, split):
        # filter out shape 3
        self.indices = np.where(dataset.latents[:, 0] != 3)[0]
        n_splits = [1 / (num_domains + 2)] * (num_domains + 2)

        domain_indices = self.compute_split(n_splits, data_dir)
        self.data_dir = data_dir
        self.latents = dataset.latents
        self.images = dataset.images

        if split=='val':
            self.latents = self.latents[domain_indices[-2]]
            self.images = self.images[domain_indices[-2]]
        elif split=='test':
            self.latents = self.latents[domain_indices[-1]]
            self.images = self.images[domain_indices[-1]]

        self.domain_indices = domain_indices[:num_domains]
        colors = copy.deepcopy(self.latents[:, 0]) - 1
        self.latents = np.concatenate([self.latents, np.expand_dims(colors, -1)], -1)
        self.batch_size = batch_size

        self.color_palattes = self.retrieve_colors(num_domains)
        self.split = split

    def reset_batch(self):
        """Reset batch indices for each domain."""
        self.batch_indices, self.batches_left = {}, {}
        for loc, env_idx in enumerate(self.domain_indices):
            self.batch_indices[loc] = torch.split(env_idx[torch.randperm(len(env_idx))], self.batch_size)
            self.batches_left[loc] = len(self.batch_indices[loc])

    def get_batch(self, domain):
        """Return the next batch of the specified domain."""
        batch_index = self.batch_indices[domain][len(self.batch_indices[domain]) - self.batches_left[domain]]
        return torch.stack([self.get_input(i, domain)[0] for i in batch_index]), \
               (torch.Tensor(self.latents[batch_index])[:, 0]-1).long(), None


    def compute_split(self, n_splits, data_dir):
        os.makedirs(os.path.join(data_dir, 'cdsprites'), exist_ok=True)
        path = os.path.join(data_dir, 'cdsprites', f'train_test_val_split.pt')
        if not os.path.exists(path):
            torch.save(torch.tensor(self.indices[torch.randperm(len(self.indices))]), path)
        rand_indices = torch.load(path)
        N = len(rand_indices)

        split_indices = []
        start_idx = 0
        for split in n_splits:
            end_idx = start_idx + int(N * split)
            split_indices.append(rand_indices[start_idx:end_idx])
            start_idx = end_idx
        return split_indices

    def retrieve_colors(self, total_envs=10):
        path = os.path.join(self.data_dir, 'cdsprites', f'colors_{total_envs}.pt')
        return torch.load(path)

    def get_input(self, idx, env):
        image = torch.Tensor(self.images[idx])
        latent = torch.Tensor(self.latents[idx])

        if len(image.shape)>3:
            canvas = torch.zeros_like(image).repeat(1,3,1,1)
            for c, (img, l) in enumerate(zip(image, latent)):
                canvas[c, ...] = self.get_domain_color_palatte(img, l, env)
        else:
            canvas = self.get_domain_color_palatte(image, latent, env)
        return (canvas, latent, None)


    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        image = torch.Tensor(self.images[idx])
        latent = torch.Tensor(self.latents[idx])

        if len(image.shape)>3:
            canvas = torch.zeros_like(image).repeat(1,3,1,1)
            for c, (img, l) in enumerate(zip(image, latent)):
                canvas[c, ...] = self.get_color_palatte(img, l)
        else:
            canvas = self.get_color_palatte(image, latent)
        return (canvas, latent[0].long()-1, latent[0:])

    def get_color_palatte(self, image, latent):
        chosen_color = torch.randint(high=len(self.color_palattes) - 1, size=(1,)).item()
        cc = int(latent[-1].long()) if self.split == 'train' else \
            torch.randint(high=2, size=(1,)).item()
        canvas = self.color_palattes[chosen_color][cc]
        return canvas*image

    def get_domain_color_palatte(self, image, latent, chosen_color):
        cc = int(latent[-1].long())
        canvas = self.color_palattes[chosen_color][cc]
        return canvas*image

    def eval(self, ypreds, ys, metas):
        total = ys.size(0)
        correct = (ypreds == ys).sum().item()
        test_val = [
            {'acc_avg': correct/total},
            f"Accuracy: {correct/total*100:6.2f}%"
        ]
        return test_val


DspritesDataSize = torch.Size([1, 64, 64])
class DspritesDataset(Dataset):
    """2D shapes dataset.
    More info here:
    https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_reloading_example.ipynb
    """
    def __init__(self, data_root, train=True, train_fract=0.8, split=True, clip=False):
        """
        Args:
            npz_file (string): Path to the npz file.
        """
        filename = 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
        self.npz_file = data_root + '/' + filename
        self.npz_train_file = data_root + '/train_' + filename
        self.npz_test_file = data_root + '/test_' + filename
        if not os.path.isfile(self.npz_file):
            self.download_dataset(self.npz_file)
        if split:
            if not (os.path.isfile(self.npz_train_file) and os.path.isfile(self.npz_test_file)):
                self.split_dataset(data_root, self.npz_file, self.npz_train_file,
                                   self.npz_test_file, train_fract, clip)
            dataset = np.load(self.npz_train_file if train else self.npz_test_file,
                              mmap_mode='r')
        else:
            rdataset = np.load(self.npz_file, encoding='latin1', mmap_mode='r')
            dataset = {'latents': rdataset['latents_values'][:, 1:],  # drop colour
                       'images': rdataset['imgs']}

        self.latents = dataset['latents']
        self.images = dataset['images']

    def download_dataset(self, npz_file):
        from urllib import request
        url = 'https://github.com/deepmind/dsprites-dataset/blob/master/' \
              'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true'
        print('Downloading ' + url)
        data = request.urlopen(url)
        with open(npz_file, 'wb') as f:
            f.write(data.read())

    def split_dataset(self, data_root, npz_file, npz_train_file, npz_test_file, train_fract, clip):
        print('Splitting dataset')
        dataset = np.load(npz_file, encoding='latin1', mmap_mode='r')
        latents = dataset['latents_values'][:, 1:]
        images = np.array(dataset['imgs'], dtype='float32')
        images = images.reshape(-1, *DspritesDataSize)
        if clip:
            images = np.clip(images, 1e-6, 1 - 1e-6)

        split_idx = np.int(train_fract * len(latents))
        shuffled_range = np.random.permutation(len(latents))
        train_idx = shuffled_range[range(0, split_idx)]
        test_idx = shuffled_range[range(split_idx, len(latents))]

        np.savez(npz_train_file, images=images[train_idx], latents=latents[train_idx])
        np.savez(npz_test_file, images=images[test_idx], latents=latents[test_idx])

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        image = torch.Tensor(self.images[idx]).unsqueeze(0)
        latent = torch.Tensor(self.latents[idx])
        return (image, latent)