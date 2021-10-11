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
from datasets.inference_dataset import InferenceDatasetWithPath, SeqSampler
from options.test_options import TestOptions
from models.psp import pSp
from models.e4e import e4e
from utils.model_utils import ENCODER_TYPES
from utils.common import tensor2im
from utils.inference_utils import get_average_image
from utils.lm_utils import *


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
    """
    dataloader = DataLoader(
        dataset,
        batch_size=opts.test_batch_size,
        shuffle=True,
        num_workers=int(opts.test_workers),
        drop_last=True
    )
    """
    dataloader = DataLoader(
        dataset,
        batch_sampler=SeqSampler(dataset.seq_ids),
        num_workers=int(opts.test_workers)
    )
    
    # n images to generate
    if opts.n_images is None:
        opts.n_images = len(dataset)

    # get the image corresponding to the latent average
    avg_image = get_average_image(net, opts)

    # inference setup
    global_i = 0
    global_time = []
    batch_input_paths = {}

    # read in latents
    if not opts.save_latents:
        index = faiss.read_index(os.path.join(opts.faiss_dir, 'index.bin'))
        lookup_arrays = np.load(os.path.join(opts.faiss_dir, 'lookup_array.npy'), mmap_mode='r')
        with open(os.path.join(opts.faiss_dir, 'im_names.txt')) as f:
            im_names = f.read().split()
        
    # read in lm
    if not opts.save_latents:
        bertjapanese = AutoModelForMaskedLM.from_pretrained('cl-tohoku/bert-base-japanese-char')
        bertjapanesetokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-char")

    # pca
    if opts.pca is not None:
        lat_comp = np.load(opts.pca)["lat_comp"]
        first_four_lat_comp = np.squeeze(lat_comp[:4,:,:], axis=1)
    else:
        first_four_lat_comp = None

    # setup eval
    if opts.eval_data:
        top1_acc = []
        top5_acc = []
        top10_acc = []
        
    # inference
    for input_batch, input_paths in tqdm(dataloader):

        if global_i >= opts.n_images:
            break

        with torch.no_grad():

            input_cuda = input_batch.cuda().float()
            tic = time.time()

            result_batch, result_latents = run_on_batch(input_cuda, net, opts, avg_image)

            if opts.save_latents:

                latent_array = result_latents.cpu().detach().numpy().astype('float32')
                latents_save_path = os.path.join(out_path_latents, f'{global_i}.npy')
                batch_input_paths[str(global_i)] = list(input_paths)
                with open(latents_save_path, 'wb') as f:
                    np.save(f, latent_array)
                
            else:
                
                closest_latents, closet_im_names = run_faiss(
                    result_latents, 
                    index, 
                    lookup_arrays,
                    im_names,
                    n_latents=opts.n_latents, 
                    n_neighbors=opts.n_neighbors,
                    verbose=opts.verbose,
                    pcomp=first_four_lat_comp
                )

                sequence_ocr_recog_chars = []

                for bidx, (clatent, bimgn) in enumerate(zip(closest_latents, closet_im_names)):

                    closest_input_cuda = torch.from_numpy(clatent).cuda().float()
                    result_neighbors, _ = run_on_batch(closest_input_cuda, net, opts, avg_image, just_decode=True)

                    im_path = input_paths[bidx]
                    input_im = tensor2im(input_batch[bidx])

                    viz_results(input_im, result_neighbors, out_path_coupled, im_path, bimgn, opts)

                    ocr_recog_chars = [extract_char_from_im_name(imgn) for imgn in bimgn]
                    sequence_ocr_recog_chars.append(ocr_recog_chars)

                    if opts.eval_data:
                        top1_acc.append(ocr_recog_chars[0] == extract_char_from_im_name(im_path))
                        top5_acc.append(extract_char_from_im_name(im_path) in ocr_recog_chars[:5])
                        top10_acc.append(extract_char_from_im_name(im_path) in ocr_recog_chars)

                beam_output = beam_search_from_marginal_mlm(
                    sequence_ocr_recog_chars, 
                    bertjapanese, bertjapanesetokenizer, 
                    beams=opts.n_beams
                )

                print("***\n\n\n***\n\n")
                print(input_paths)
                print(sequence_ocr_recog_chars)
                print(beam_output)

            toc = time.time()
            global_time.append(toc - tic)

        global_i += opts.test_batch_size

    # faiss index creation
    if opts.save_latents:
        index, lookup_arrays, ord_batch_paths = setup_faiss(opts, batch_input_paths, n_latents=opts.n_latents, n_imgs=global_i, pcomp=first_four_lat_comp)
        faiss.write_index(index, os.path.join(opts.faiss_dir, 'index.bin'))
        with open(os.path.join(opts.faiss_dir, 'lookup_array.npy'), 'wb') as f:
            np.save(f, lookup_arrays)
        with open(os.path.join(opts.faiss_dir, 'im_names.txt'), 'w') as f:
            f.write('\n'.join(ord_batch_paths))

    # create stats
    stats_path = os.path.join(opts.exp_dir, 'stats.txt')
    result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
    print(result_str)
    with open(stats_path, 'w') as f:
        f.write(result_str)
    if opts.eval_data:
        top1_acc = sum(top1_acc)/len(top1_acc)
        top5_acc = sum(top5_acc)/len(top5_acc)
        top10_acc = sum(top10_acc)/len(top10_acc)
        print(f"Top-1 accuracy is {top1_acc}")
        print(f"Top-5 accuracy is {top5_acc}")
        print(f"Top-10 accuracy is {top10_acc}")


def viz_results(input_im, result_neighbors, out_path_coupled, im_path, bimgn, opts):

    # viz results encoded
    res = [np.array(input_im)]
    res = res + [np.array(tensor2im(result_neighbors[i])) for i in range(opts.n_neighbors)]
    res = np.concatenate(res, axis=1)
    Image.fromarray(res).save(os.path.join(out_path_coupled, os.path.basename(im_path)))

    # viz results wrt src file
    res = [np.array(input_im)]
    res = res + [np.array(Image.open(i).convert('RGB')) for i in bimgn]
    res = np.concatenate(res, axis=1)
    Image.fromarray(res).save(os.path.join(out_path_coupled, f"src_im_{os.path.basename(im_path)}"))

    # ocr top1 save
    top_char = extract_char_from_im_name(bimgn[0])
    input_im.save(os.path.join(out_path_coupled, 
        f"top1_{top_char}.png"))


def setup_faiss(opts, batch_im_paths, n_latents, n_imgs, dim=512, wplus=10, pcomp=None):

    # create index
    index = faiss.IndexFlatIP(dim)
    all_arrays = np.empty((n_imgs, wplus, dim), dtype=np.float32)
    all_paths = []

    # load index
    root_dir = os.path.join(opts.faiss_dir, 'inference_latents')
    idx = 0

    for filename in tqdm(os.listdir(root_dir)):

        paths = batch_im_paths[os.path.splitext(filename)[0]]
        all_paths.extend(paths)

        saved_latents = np.load(os.path.join(root_dir, filename))
        all_arrays[idx:idx+opts.test_batch_size,:,:] = saved_latents

        reshaped_latents = embed_latent(saved_latents, n_latents, np.mean, pcomp)
        faiss.normalize_L2(reshaped_latents)
        index.add(reshaped_latents)

        idx += opts.test_batch_size

    print(f'Total indices {index.ntotal}')

    return index, all_arrays, all_paths


def run_faiss(query_latents, index, all_arrays, all_im_names, n_latents, n_neighbors=5, verbose=True, pcomp=None):
    
    # search index
    reshaped_query_latents = embed_latent(query_latents, n_latents, np.mean, pcomp)
    D, I = index.search(reshaped_query_latents, n_neighbors)
    if verbose:
        print(I)
        print(D)

    # return closest
    closest_indices = np.apply_along_axis(lambda x: x[:n_neighbors], axis=1, arr=I)
    closest_codes = [all_arrays[cidx,:,:] for cidx in closest_indices]
    closest_im_names = [[all_im_names[i]  for i in cidx] for cidx in closest_indices]

    return closest_codes, closest_im_names


def embed_latent(latents, n_latents, agg_func, pcomp=None):

    if torch.is_tensor(latents):
        latents = latents.cpu().detach().numpy() # eg (2, 10, 512)

    if pcomp is None:

        embedding = np.ascontiguousarray(
            agg_func(latents[:,:n_latents,:], axis=1).reshape((latents.shape[0], -1))
        )

    else:

        embedding = np.empty((latents.shape[0], pcomp.shape[0], latents.shape[-1]))

        for i in range(latents.shape[0]):

            latent = latents[i,:,:].T
            proj = pcomp @ latent
            embedding[i,:,:] = proj

        embedding = np.ascontiguousarray(
            embedding.reshape((latents.shape[0], -1))
        )

    return embedding


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


def extract_char_from_im_name(imn):
    return os.path.basename(imn)[0]


def nn_scoring(neighbors):
    neighbors_set = list(set(neighbors))
    scores = [sum([len(neighbors) - idx for idx, x in enumerate(neighbors) if x == s]) for s in neighbors_set]
    return max(neighbors_set, key=lambda x: scores[neighbors_set.index(x)])


if __name__ == '__main__':
    main()
