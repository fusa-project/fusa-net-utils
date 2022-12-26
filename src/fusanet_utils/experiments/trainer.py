import logging
import time
import numpy as np
import torch

from dvclive import Live

from .metrics import accuracy, f1_score, error_rate

logger = logging.getLogger(__name__)


def criterion(label):
    if label.ndim == 3:  # SED
        return torch.nn.BCELoss()
    else:
        return torch.nn.CrossEntropyLoss()


class EarlyStopping():
    def __init__(self, metric='valid_loss', patience=5):

        self.metric = metric
        if self.metric == 'valid_loss':
            self.best = np.Inf
            self.metric_operator = np.less
        elif self.metric == 'f1_score':
            self.best = -np.Inf
            self.metric_operator = np.greater
        self.patience = patience
        self.counter = 0
        self.early_stop = False

    def __call__(self, epoch, epoch_value):
        if self.metric_operator(self.best, epoch_value):
            self.counter += 1
            logger.info(f"Early stopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info("Training early stopped")
        else:
            self.best = epoch_value
            self.counter = 0
            logger.info(f"New best {self.metric} in epoch {epoch} ({self.best})")


def train_model(loaders: tuple, params: dict, model_path: str) -> None:
    """
    Make more abstract to other models
    """
    train_loader, valid_loader = loaders

    n_train, n_valid = len(train_loader.dataset), len(valid_loader.dataset)
    model = torch.load(model_path)
    
    # TODO: Clean this function
    
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=params['train']['learning_rate'])
    cuda = params['train']['cuda']
    if cuda and torch.cuda.device_count() > 0:
        logger.info('GPU number: {}'.format(torch.cuda.device_count()))
        device = 'cuda'
        torch.backends.cudnn.benchmark = True
    else:
        device = 'cpu'
    logger.info(f'Using {device}')

    early_stopping = EarlyStopping(params["train"]["stopping_criteria"],
                                   params["train"]["patience"])
    clip = 10.0
    with Live() as live:
        for key in ['model', 'pretrained',
                    'augmentation', 'balanced']:
            live.log_param(key, params["train"][key])
        for epoch in range(params["train"]["nepochs"]):
            global_loss = 0.0
            global_accuracy = 0.0
            model.to(device)
            model.train()
            start_time = time.time()
            for batch in train_loader:
                marshalled_batch = {}
                for key in batch:
                    if key == 'filename':
                        continue
                    marshalled_batch[key] = batch[key].to(device, non_blocking=True)
                    #if key == 'waveform':
                    #    marshalled_batch[key] = marshalled_batch[key][:,:,:320002]
                optimizer.zero_grad()
                y = model.forward(marshalled_batch)
                loss = criterion(marshalled_batch['label'])(y, marshalled_batch['label'])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                global_loss += loss.item()
                global_accuracy += accuracy(y, marshalled_batch['label'])
            logger.info(f"{epoch}, train/loss {global_loss/n_train:0.4f}")
            logger.info(f"{epoch}, train/accuracy {global_accuracy/n_train:0.4f}")
            live.log_metric('train/loss', global_loss/n_train)
            live.log_metric('train/accuracy', global_accuracy/n_train)
            logger.info(f"train time: {time.time() - start_time:0.4f} [s]")

            global_loss = 0.0
            global_accuracy = 0.0
            global_f1_score = 0.0
            global_error_rate = 0.0
            model.eval()
            start_time = time.time()
            with torch.no_grad():
                for batch in valid_loader:
                    marshalled_batch = {}
                    for key in batch:
                        if key == 'filename':
                            continue
                        marshalled_batch[key] = batch[key].to(device, non_blocking=True)
                        #if key == 'waveform':
                        #    marshalled_batch[key] = marshalled_batch[key][:,:,:320002]
                    y = model.forward(marshalled_batch)
                    loss = criterion(marshalled_batch['label'])(y, marshalled_batch['label'])
                    global_loss += loss.item()
                    global_accuracy += accuracy(y, marshalled_batch['label'])
                    global_f1_score += f1_score(y, marshalled_batch['label'])
                    global_error_rate += error_rate(y, marshalled_batch['label'])
            logger.info(f"{epoch}, valid/loss {global_loss/n_valid:0.4f}")
            logger.info(f"{epoch}, valid/accuracy {global_accuracy/n_valid:0.4f}")
            logger.info(f"{epoch}, f1_score_macro {global_f1_score/len(valid_loader):0.4f}")
            logger.info(f"{epoch}, error_rate {global_error_rate/len(valid_loader):0.4f}")
            if params["train"]["stopping_criteria"] == 'valid_loss':
                early_stopping(epoch, global_loss/n_valid)
            elif params["train"]["stopping_criteria"] == 'f1_score':
                early_stopping(epoch, global_f1_score/len(valid_loader))
            live.log_metric('valid/loss', global_loss/n_valid)
            live.log_metric('valid/accuracy', global_accuracy/n_valid)
            live.log_metric('f1_score_macro', global_f1_score/len(valid_loader))
            live.log_metric('error_rate', global_error_rate/len(valid_loader))
            logger.info(f"valid time: {time.time() - start_time:0.4f} [s]")

            # Saving best model
            if early_stopping.early_stop:
                break
            else:
                live.log_metric('best_epoch', epoch)
                live.log_metric('valid/best_loss', global_loss/n_valid)
                live.log_metric('valid/best_accuracy', global_accuracy/n_valid)
                live.log_metric('best_f1_score_macro', global_f1_score/len(valid_loader))
                live.log_metric('best_error_rate', global_error_rate/len(valid_loader))

                if device == 'cuda':
                    model.cpu()
                torch.save(model, model_path)
                #model.create_trace()
            live.next_step()

