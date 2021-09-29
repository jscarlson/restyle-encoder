import os
from argparse import Namespace
from tqdm import tqdm
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import sys

import faiss
from PIL import Image

sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDatasetWithPath
from options.test_options import TestOptions
from models.psp import pSp
from models.e4e import e4e
from utils.model_utils import ENCODER_TYPES
from utils.common import tensor2im
from utils.inference_utils import get_average_image


def main():

    # path setup
    test_opts = TestOptions().parse()
    out_path_results = os.path.join(test_opts.exp_dir, 'inference_results')
    out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled')
    out_path_latents = os.path.join(test_opts.faiss_dir, 'inference_latents')
    os.makedirs(out_path_results, exist_ok=True)
    os.makedirs(out_path_coupled, exist_ok=True)
    os.makedirs(out_path_latents, exist_ok=True)

    # update test options with options used during training
    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    opts = Namespace(**opts)

    # model setup
    net = pSp(opts)
    net.eval()
    net.cuda()

    # dataset setup
    print('Loading dataset for {}'.format(opts.dataset_type))
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    dataset = InferenceDatasetWithPath(
        root=opts.data_path,
        transform=transforms_dict['transform_inference'],
        opts=opts
    )
    dataloader = DataLoader(
        dataset,
        batch_size=opts.test_batch_size,
        shuffle=True,
        num_workers=int(opts.test_workers),
        drop_last=True
    )

    # n images to generate
    if opts.n_images is None:
        opts.n_images = len(dataset)

    # get the image corresponding to the latent average
    avg_image = get_average_image(net, opts)

    # inference setup
    global_i = 0
    global_time = []
    all_input_paths = []

    # read in latents
    if not opts.save_latents:
        index = faiss.read_index(os.path.join(opts.faiss_dir, 'index.bin'))
        lookup_arrays = np.load(os.path.join(opts.faiss_dir, 'lookup_array.npy'), mmap_mode='r')
        
    # inference
    for input_batch, input_paths in tqdm(dataloader):

        if global_i >= opts.n_images:
            break

        all_input_paths.extend(input_paths)

        with torch.no_grad():

            input_cuda = input_batch.cuda().float()
            tic = time.time()

            result_batch, result_latents = run_on_batch(input_cuda, net, opts, avg_image)

            if opts.save_latents:

                latent_array = result_latents.cpu().detach().numpy().astype('float32')
                latents_save_path = os.path.join(out_path_latents, f'{global_i}.npy')
                with open(latents_save_path, 'wb') as f:
                    np.save(f, latent_array)
                
            else:
                
                closest_latents = run_faiss(
                    result_latents, 
                    index, 
                    lookup_arrays, 
                    n_latents=opts.n_latents, 
                    n_neighbors=opts.n_neighbors,
                    verbose=opts.verbose
                )

                for bidx, clatent in enumerate(closest_latents):

                    closest_input_cuda = torch.from_numpy(clatent).cuda().float()
                    result_neighbors, _ = run_on_batch(closest_input_cuda, net, opts, avg_image, just_decode=True)

                    im_path = input_paths[bidx]
                    input_im = tensor2im(input_batch[bidx])
                    res = [np.array(input_im)]
                    res = res + [np.array(tensor2im(result_neighbors[i])) for i in range(opts.n_neighbors)]
                    res = np.concatenate(res, axis=1)
                    Image.fromarray(res).save(os.path.join(out_path_coupled, os.path.basename(im_path)))

            toc = time.time()
            global_time.append(toc - tic)

        global_i += opts.test_batch_size

    # faiss index creation
    if opts.save_latents:
        index, lookup_arrays = setup_faiss(opts, n_latents=opts.n_latents, n_imgs=global_i)
        faiss.write_index(index, os.path.join(opts.faiss_dir, 'index.bin'))
        with open(os.path.join(opts.faiss_dir, 'lookup_array.npy'), 'wb') as f:
            np.save(f, lookup_arrays)
        
    # create stats
    stats_path = os.path.join(opts.exp_dir, 'stats.txt')
    result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
    print(result_str)
    with open(stats_path, 'w') as f:
        f.write(result_str)


def setup_faiss(opts, n_latents, n_imgs, dim=512, wplus=10):

    # create index
    index = faiss.IndexFlatIP(dim)
    all_arrays = np.empty((n_imgs, wplus, dim), dtype=np.float32)

    # load index
    root_dir = os.path.join(opts.faiss_dir, 'inference_latents')
    idx = 0
    for filename in tqdm(os.listdir(root_dir)):
        saved_latents = np.load(os.path.join(root_dir, filename))
        all_arrays[idx:idx+opts.test_batch_size,:,:] = saved_latents
        reshaped_latents = reshape_latent(saved_latents, n_latents)
        faiss.normalize_L2(reshaped_latents)
        index.add(reshaped_latents)
        idx += opts.test_batch_size

    print(f'Total indices {index.ntotal}')

    return index, all_arrays


def run_faiss(query_latents, index, all_arrays, n_latents, n_neighbors=5, verbose=True):
    
    # search index
    reshaped_query_latents = reshape_latent(query_latents, n_latents)
    D, I = index.search(reshaped_query_latents, n_neighbors)
    if verbose:
        print(I)
        print(D)

    # return closest
    closest_indices = np.apply_along_axis(lambda x: x[:n_neighbors], axis=1, arr=I)
    closest_codes = [all_arrays[cidx,:,:] for cidx in closest_indices]

    return closest_codes


def reshape_latent(latents, n_latents):
    if torch.is_tensor(latents):
        latents = latents.cpu().detach().numpy()
    return np.ascontiguousarray(
        np.sum(latents[:,:n_latents,:], axis=1).reshape((latents.shape[0], -1))
    )


def run_on_batch(inputs, net, opts, avg_image, just_decode=False):

    if not just_decode:

        y_hat, latent = None, None

        for iter in range(opts.n_iters_per_batch):

            if iter == 0:
                avg_image_for_batch = avg_image.unsqueeze(0).repeat(inputs.shape[0], 1, 1, 1)
                x_input = torch.cat([inputs, avg_image_for_batch], dim=1)
            else:
                x_input = torch.cat([inputs, y_hat], dim=1)

            y_hat, latent = net.forward(x_input,
                                        latent=latent,
                                        randomize_noise=False,
                                        return_latents=True)

    else:

        y_hat, latent = net.forward(inputs,
                                    latent=None,
                                    input_code=True,
                                    randomize_noise=False,
                                    return_latents=True)


    return y_hat, latent


if __name__ == '__main__':
    main()
