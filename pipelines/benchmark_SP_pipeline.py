import csv
import os

from torch.utils.data import DataLoader
import torchvision.transforms as tvs_trans

from datasets.folder_dataset import FolderDataset
from evaluators.utils import get_evaluator
from networks.utils import get_network
from postprocessors import get_postprocessor
from preprocessors.test_preprocessor import TestStandardPreProcessor
from utils.logger import setup_logger


class Benchmark_SP_Pipeline:
    def __init__(self, config):
        self.config = config

        # build model
        self.net = get_network(self.config.network)
        self.nets = {'net': self.net}

        # preprocessor: force a square resize first to ensure batchable tensors
        base_preprocessor = TestStandardPreProcessor(self.config, 'test')
        image_size = self.config.dataset.image_size
        self.preprocessor = tvs_trans.Compose([
            tvs_trans.Resize((image_size, image_size)),
            base_preprocessor,
        ])

        # dataset/dataloader
        self.folder_path = self.config.dataset.folder
        self.dataset = FolderDataset(self.folder_path, transform=self.preprocessor)
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=self.config.dataset.batch_size,
                                     shuffle=False,
                                     num_workers=self.config.dataset.num_workers)

        # postprocessor
        self.postprocessor = get_postprocessor(self.config)

    def run(self):
        setup_logger(self.config)

        # force energy mode for unlabeled inference
        self.postprocessor.mode = 'energy'

        # inference
        pred_list, logit_list, label_list, score_list = self.postprocessor.inference(
            self.nets, self.dataloader, mode='energy', eval=False)

        # map to labels
        idx_to_label = {0: 'real', 1: 'fake'}

        # save CSV
        os.makedirs(self.config.output_dir, exist_ok=True)
        csv_path = os.path.join(self.config.output_dir, 'benchmark_predictions.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image_path', 'prediction'])
            for i, pred in enumerate(pred_list):
                image_path = self.dataset.image_paths[i]
                writer.writerow([image_path, idx_to_label.get(int(pred), str(int(pred)))])

        print('Saved predictions to {}'.format(csv_path), flush=True)



