from argparse import ArgumentParser


class TestOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        # arguments for inference script
        self.parser.add_argument('--exp_dir', type=str,
                                 help='Path to experiment output directory')
        self.parser.add_argument('--checkpoint_path', default=None, type=str,
                                 help='Path to ReStyle model checkpoint')
        self.parser.add_argument('--data_path', type=str, default='gt_images',
                                 help='Path to directory of images to evaluate')
        self.parser.add_argument('--resize_outputs', action='store_true',
                                 help='Whether to resize outputs to 256x256 or keep at original output resolution')
        self.parser.add_argument('--test_batch_size', default=2, type=int,
                                 help='Batch size for testing and inference')
        self.parser.add_argument('--test_workers', default=2, type=int,
                                 help='Number of test/inference dataloader workers')
        self.parser.add_argument('--n_images', type=int, default=None,
                                 help='Number of images to output. If None, run on all data')

        # arguments for iterative inference
        self.parser.add_argument('--n_iters_per_batch', default=5, type=int,
                                 help='Number of forward passes per batch during training.')

        # arguments for encoder bootstrapping
        self.parser.add_argument('--model_1_checkpoint_path', default=None, type=str,
                                 help='Path to encoder used to initialize encoder bootstrapping inference.')
        self.parser.add_argument('--model_2_checkpoint_path', default=None, type=str,
                                 help='Path to encoder used to iteratively translate images following '
                                      'model 1\'s initialization.')

        # PSNP
        self.parser.add_argument('--faiss_dir', type=str, default=None, required=True, help='Path to NPY files for FAISS')
        self.parser.add_argument('--pca', type=str, default=None, help='Path to PCA npz')
        self.parser.add_argument('--save_latents', action='store_true', help='Save latents for nearest neighbors comparisons')
        self.parser.add_argument('--save_images', action='store_true', help='Save images')
        self.parser.add_argument('--n_latents', default=5, type=int, help='Number of W+ vectors to concat for FAISS')
        self.parser.add_argument('--n_neighbors', default=10, type=int, help='Number of nearest neighbors to visualize')
        self.parser.add_argument('--n_beams', default=10, type=int, help='Number of beams for beam search')
        self.parser.add_argument('--verbose', action='store_true', default=False, help='Be... verbose')
        self.parser.add_argument('--eval_data', action='store_true', default=False, help='Evaluate OCR accuracy using input data labels')

        # arguments for editing
        self.parser.add_argument('--edit_directions', type=str, default='age,smile,pose',
                                 help='comma-separated list of which edit directions top perform.')
        self.parser.add_argument('--factor_ranges', type=str, default='5,5,5',
                                 help='comma-separated list of max ranges for each corresponding edit.')


    def parse(self):
        opts = self.parser.parse_args()
        return opts
