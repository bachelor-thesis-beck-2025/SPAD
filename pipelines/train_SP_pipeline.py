import utils.comm as comm
from datasets import get_dataloader_SP
from evaluators.utils import get_evaluator
from networks.utils import get_network
from recorders.utils import get_recorder
from trainers.utils import get_trainer
from utils.logger import setup_logger

from postprocessors import get_postprocessor


class Train_SP_Pipeline:
    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        # generate output directory and save the full config file
        setup_logger(self.config)

        # get dataloader
        loader_dict = get_dataloader_SP(self.config)
        train_loader = loader_dict['train']

        # init network
        net = get_network(self.config.network)

        # init trainer and evaluator
        trainer = get_trainer(net, train_loader, self.config)

        evaluator = get_evaluator(self.config)

        postprocessor = get_postprocessor(self.config)

        if comm.is_main_process():
            # init recorder
            recorder = get_recorder(self.config)

            print('Start training...', flush=True)
        
        if not comm.is_main_process():
            # Ensure recorder is defined for non-main processes
            recorder = None


        batch_idx = 0
        for epoch_idx in range(1, self.config.optimizer.num_epochs + 1):
            nets, train_metrics, batch_idx = trainer.train_epoch(epoch_idx, batch_idx, evaluator, postprocessor, recorder)

            comm.synchronize()
            if comm.is_main_process():
                # Evaluate on validation set
                val_metrics = evaluator.eval_acc_SP(nets, loader_dict['test'], postprocessor, epoch_idx, batch_idx)
                
                # save model and report the result
                
                recorder.save_model(nets, val_metrics)
                recorder.report(train_metrics, val_metrics)
                
                # original pipeline
                # recorder.save_model(nets, train_metrics)
                # recorder.report(train_metrics)

        if comm.is_main_process():
            recorder.summary()
            print(u'\u2500' * 70, flush=True)

        if comm.is_main_process():
            print('Completed!', flush=True)
