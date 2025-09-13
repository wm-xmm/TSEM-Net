"""
Training the model
Extended from original implementation of PANet by Wang et al.
"""
import os
import shutil
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import csv

from models.med_seg import FewShotSeg
from dataloaders.GenericSuper import SuperpixelDataset
from dataloaders.dataset_utils import DATASET_INFO
import dataloaders.augutils as myaug

from util.utils import set_seed, t2n, to01, compose_wt_simple
from util.metric import Metric
import glob
from config_ssl_upload import ex
import tqdm
# config pre-trained model caching path
os.environ['TORCH_HOME'] = "./pretrained_model"  
pretrained_path= ""

@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)  

        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

    os.makedirs(f'{_run.observers[0].dir}/snapshots/best_weights', exist_ok=True)
    ALL_SCALE = "MIDDLE"  
    CPT = "myexp"

    for EVAL_FOLD in range(5):
        for SUPERPIX_SCALE in ALL_SCALE:
            PREFIX = f"train_{_config['dataset']}_lbgroup{_config['label_sets']}_scale_{SUPERPIX_SCALE}_vfold{EVAL_FOLD}"
            print(PREFIX)
            LOGDIR = f".runs2/exps/{CPT}_{SUPERPIX_SCALE}_{_config['label_sets']}"

        set_seed(_config['seed'])
        cudnn.enabled = True
        cudnn.benchmark = True
        torch.cuda.set_device(device=_config['gpu_id'])
        torch.set_num_threads(1)

        _log.info('###### Create model ######')


        model = model.cuda()

        max_label = 13
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total number of parameters**********: {total_params}")

        model.train()
        _log.info('###### Load data ######')
        ### Training set
        data_name = _config['dataset']
        if data_name == 'SABS_Superpix':
            baseset_name = 'SABS1'
        elif data_name == '3Dircadb_Superpix':
            baseset_name = '3Dircadb'
        elif data_name == 'CHAOST2_Superpix':
            baseset_name = 'CHAOST2'
        else:
            raise ValueError(f'Dataset: {data_name} not found')

        ### Transforms for data augmentation
        tr_transforms = myaug.transform_with_label({'aug': myaug.augs[_config['which_aug']]})
        assert _config['scan_per_load'] < 0 # by default we load the entire dataset directly


        test_labels = DATASET_INFO[baseset_name]['LABEL_GROUP']['pa_all'] - DATASET_INFO[baseset_name]['LABEL_GROUP'][_config["label_sets"]]
        _log.info(f'###### Labels excluded in training : {[lb for lb in _config["exclude_cls_list"]]} ######')
        _log.info(f'###### Unseen labels evaluated in testing: {[lb for lb in test_labels]} ######')

        tr_parent = SuperpixelDataset( # base dataset
            which_dataset = baseset_name,
            base_dir=_config['path'][data_name]['data_dir'],
            idx_split=0,
            mode='train',
            min_fg=str(_config["min_fg_data"]), # dummy entry for superpixel dataset
            transforms=tr_transforms,
            nsup = _config['task']['n_shots'],
            scan_per_load = _config['scan_per_load'],
            exclude_list = _config["exclude_cls_list"],
            superpix_scale = _config["superpix_scale"],
            fix_length = _config["max_iters_per_load"] if (data_name == 'C0_Superpix') or (data_name == 'CHAOST2_Superpix') else None
        )

        ### dataloaders
        trainloader = DataLoader(
            tr_parent,
            batch_size=_config['batch_size'],
            shuffle=True,
            num_workers=_config['num_workers'],
            pin_memory=True,
            drop_last=True
        )
        _log.info('###### Set optimizer ######')

        if _config['optim_type'] == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), **_config['optim'])
        else:
            raise NotImplementedError

        scheduler = MultiStepLR(optimizer, milestones=_config['lr_milestones'],  gamma = _config['lr_step_gamma'])

        my_weight = compose_wt_simple(_config["use_wce"], data_name)
        criterion = nn.CrossEntropyLoss(ignore_index=_config['ignore_label'], weight = my_weight)  # ignore_label=255

        i_iter = 0 # total number of iteration
        n_sub_epoches = _config['n_steps'] // _config['max_iters_per_load'] # number of times for reloading

        log_loss = {'loss': 0, 'align_loss': 0, 'edge_histogram': 0, 'query_loss': 0,}

        _log.info('###### Training ######')
        best_loss = 1e10
        losses = []
        iter= []
        loss_end = []
        for sub_epoch in range(n_sub_epoches):
            _log.info(f'###### This is epoch {sub_epoch} of {n_sub_epoches} epoches ######')
            for _, sample_batched in enumerate(trainloader):
                # Prepare input
                i_iter += 1
                # add writers
                support_images = [[shot.cuda() for shot in way]
                                  for way in sample_batched['support_images']]
                support_fg_mask = [[shot[f'fg_mask'].float().cuda() for shot in way]
                                   for way in sample_batched['support_mask']]
                support_bg_mask = [[shot[f'bg_mask'].float().cuda() for shot in way]
                                   for way in sample_batched['support_mask']]

                query_images = [query_image.cuda()
                                for query_image in sample_batched['query_images']]
                query_labels = torch.cat(
                    [query_label.long().cuda() for query_label in sample_batched['query_labels']], dim=0)
                ori_image = torch.cat(
                    [query_image.long().cuda() for query_image in sample_batched['query_images']], dim=0)
                optimizer.zero_grad()
                # FIXME: in the model definition, filter out the failure case where pseudolabel falls outside of image or too small to calculate a prototype
                # try:
                query_pred, align_loss, debug_vis, assign_mats = model(support_images, support_fg_mask, support_bg_mask, query_images, isval = False, val_wsize = None)
                # except:
                #     print('Faulty batch detected, skip')
                #     continue
                w1 = [0.1, 0.3, 0.5, 0.7, 0.9]
                mar_val_metric_node = Metric(max_label=max_label, n_scans=5)  # n_scans=5
                query_loss = criterion(query_pred, query_labels)  # 交叉熵损失
                #perimeter_loss = mar_val_metric_node.Perimeter_loss(query_pred, query_labels)

                #Hullarea_loss = mar_val_metric_node.Hullarea_loss(query_pred, query_labels)  #凸包
                # Holedet = (mar_val_metric_node.Holedet(query_pred, query_labels))*0.1        #检测孔洞
                # Eucdis_loss = mar_val_metric_node.Eucdis_loss(query_pred, query_labels)      #欧式
                # print("######:", Holedet, "_____:", Eucdis_loss)
                # main_loss =  Eucdis_loss+ Holedet  #Hullarea_loss+
                #loss1 = w1[EVAL_FOLD]*(query_loss + align_loss)+(1-w1[EVAL_FOLD])*(main_loss)

                edge_histogram = mar_val_metric_node.edge_histogram(ori_image, query_pred, query_labels)
                loss = w1[EVAL_FOLD] * (query_loss + align_loss) + (1 - w1[EVAL_FOLD]) * edge_histogram

                optimizer.zero_grad()
                #loss1 = query_loss + align_loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                # Log loss
                loss = loss.detach().data.cpu().numpy()
                query_loss = query_loss.detach().data.cpu().numpy()
                #perimeter_loss = perimeter_loss.detach().data.cpu().numpy()
                edge_histogram = edge_histogram.detach().data.cpu().numpy()
                align_loss = align_loss.detach().data.cpu().numpy() if align_loss != 0 else 0

                _run.log_scalar('loss', loss)
                _run.log_scalar('query_loss', query_loss)
                #_run.log_scalar('perimeter_loss', perimeter_loss)
                _run.log_scalar('edge_histogram', edge_histogram)
                # _run.log_scalar('align_loss', align_loss)
                log_loss['loss'] += loss
                log_loss['query_loss'] += query_loss
                #log_loss['perimeter_loss'] += perimeter_loss
                log_loss['edge_histogram'] += edge_histogram
                # log_loss['align_loss'] += align_loss

                # print loss and take snapshots
                if (i_iter + 1) % _config['print_interval'] == 0:

                    loss = log_loss['loss'] / _config['print_interval']
                    edge_histogram = log_loss['edge_histogram'] / _config['print_interval']
                    query_loss = log_loss['query_loss'] / _config['print_interval']

                    log_loss['loss'] = 0
                    log_loss['edge_histogram'] = 0
                    log_loss['query_loss'] = 0

                    if loss < best_loss:
                        folder_path = f'{_run.observers[0].dir}/snapshots/best_weights'
                        if folder_path is not None:
                            for file in os.listdir(folder_path):
                                if file.endswith('.pth'):
                                    os.remove(os.path.join(folder_path, file))
                        torch.save(model.state_dict(), os.path.join(f'{_run.observers[0].dir}/snapshots/best_weights', f'{str(EVAL_FOLD)}_{i_iter + 1}.pth'))
                        best_loss = loss

                    print(f'step {i_iter+1}: loss: {loss}, edge_loss:{edge_histogram}, CE_loss:{query_loss}')

                    if (i_iter + 1) % 1000 == 0:
                        losses.append(loss)
                        iter.append(i_iter + 1)
                        plt.plot(iter, losses)
                        plt.title("Loss")
                        plt.xlabel("Iter step")
                        plt.ylabel("Loss")
                        plt.savefig(os.path.join(f'{_run.observers[0].dir}/snapshots', str(EVAL_FOLD) + '.png'))
                        plt.close()

                    if (i_iter + 1) % 5000 == 0:
                        loss_end.append(loss)


                if (i_iter + 1) % _config['save_snapshot_every'] == 0:
                    if (i_iter + 1)>=30000 and data_name == 'SABS_Superpix':
                        _log.info('###### Taking snapshot ######')
                        torch.save(model.state_dict(), os.path.join(f'{_run.observers[0].dir}/snapshots', f'{str(EVAL_FOLD)}_{i_iter + 1}.pth'))
                    if data_name == 'CHAOST2_Superpix':
                        _log.info('###### Taking snapshot ######')
                        torch.save(model.state_dict(), os.path.join(f'{_run.observers[0].dir}/snapshots',
                                                                    f'{str(EVAL_FOLD)}_{i_iter + 1}.pth'))


                if data_name == 'C0_Superpix' or data_name == 'CHAOST2_Superpix':
                    if (i_iter + 1) % _config['max_iters_per_load'] == 0:
                        _log.info('###### Reloading dataset ######')
                        trainloader.dataset.reload_buffer()
                        print(f'###### New dataset with {len(trainloader.dataset)} slices has been loaded ######')


        titles = ['一', '二', '三', '四', '五']
        os.makedirs(f'{_run.observers[0].dir}/result', exist_ok=True)
        output = f'{_run.observers[0].dir}/result'
        fold = 'result_loss.csv'
        file_path = os.path.join(output, fold)
        if EVAL_FOLD == 0:
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                row = [str(titles[EVAL_FOLD])] + [str(val) for val in loss_end]
                writer.writerow(row)
        else:
            with open(file_path, 'a', newline='') as file:
                writer = csv.writer(file)
                row = [str(titles[EVAL_FOLD])] + [str(val) for val in loss_end]
                writer.writerow(row)

        