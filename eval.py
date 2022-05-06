import numpy as np 
import os 

import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader 
import torchvision.models as models
from torchvision import transforms
from tensorboardX import SummaryWriter

from emotic import Emotic 
from emotic_dataset import Emotic_PreDataset
from loss import DiscreteLoss, ContinuousLoss_SL1, ContinuousLoss_L2
from prepare_models import prep_models
from skipnet import get_skipnet
from test import test_data, test_scikit_ap, test_vad
from tqdm import tqdm
import argparse
import os
from emotic import Emotic
from test import test_emotic
from inference import inference_emotic


def train_emotic(result_path, model_path, train_log_path, val_log_path, ind2cat, ind2vad, context_norm, body_norm, args):
    ''' Prepare dataset, dataloders, models. 
    :param result_path: Directory path to save the results (val_predidictions mat object, val_thresholds npy object).
    :param model_path: Directory path to load pretrained base models and save the models after training. 
    :param train_log_path: Directory path to save the training logs. 
    :param val_log_path: Directoty path to save the validation logs. 
    :param ind2cat: Dictionary converting integer index to categorical emotion. 
    :param ind2vad: Dictionary converting integer index to continuous emotion dimension (Valence, Arousal and Dominance).
    :param context_norm: List containing mean and std values for context images. 
    :param body_norm: List containing mean and std values for body images. 
    :param args: Runtime arguments. 
    '''
    # Load preprocessed data from npy files
    train_context = np.load(os.path.join(args.data_path, 'train_context_arr.npy'))
    train_body = np.load(os.path.join(args.data_path, 'train_body_arr.npy'))
    train_cat = np.load(os.path.join(args.data_path, 'train_cat_arr.npy'))
    train_cont = np.load(os.path.join(args.data_path, 'train_cont_arr.npy'))

    val_context = np.load(os.path.join(args.data_path, 'val_context_arr.npy'))
    val_body = np.load(os.path.join(args.data_path, 'val_body_arr.npy'))
    val_cat = np.load(os.path.join(args.data_path, 'val_cat_arr.npy'))
    val_cont = np.load(os.path.join(args.data_path, 'val_cont_arr.npy'))

    print ('train ', 'context ', train_context.shape, 'body', train_body.shape, 'cat ', train_cat.shape, 'cont', train_cont.shape)
    print ('val ', 'context ', val_context.shape, 'body', val_body.shape, 'cat ', val_cat.shape, 'cont', val_cont.shape)

    # Initialize Dataset and DataLoader 
    train_transform = transforms.Compose([transforms.ToPILImage(),transforms.RandomHorizontalFlip(), transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])

    train_dataset = Emotic_PreDataset(train_context, train_body, train_cat, train_cont, train_transform, context_norm, body_norm)
    val_dataset = Emotic_PreDataset(val_context, val_body, val_cat, val_cont, test_transform, context_norm, body_norm)

    # train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)

    print ( 'val loader ', len(val_loader))

    # Prepare models 
    model_context, model_body = prep_models(context_model=args.context_model, body_model=args.body_model, model_dir=model_path)

    model_context = torch.load(r'debug_exp/models/0.3542_13_model_context1.pth', map_location=lambda storage, loc: storage)
    use_model='skipnet'
    if use_model!='skipnet':
        model_body = nn.Sequential(*(list(model_body.children())[:-1]))
        emotic_model = Emotic(list(model_context.children())[-1].in_features, list(model_body.children())[-1].in_features)
    else:
        model_body = get_skipnet(num_classes=512, skip_w=8)

        model_body = torch.load(r'debug_exp/models/0.3542_13_model_body1.pth', map_location=lambda storage, loc: storage)

        #state_dict = {str.replace(k, 'module.', ''): v for k, v in model_weight['state_dict'].items()}
        # model_body.load_state_dict(state_dict)

        # emotic_model = Emotic(list(model_context.children())[-1].in_features, list(model_body.children())[-1].out_features)

        emotic_model = torch.load(r'debug_exp/models/0.3542_13_model_emotic1.pth', map_location=lambda storage, loc: storage)

        # state_dict = {str.replace(k, 'module.', ''): v for k, v in model_weight['state_dict'].items()}
        # emotic_model.load_state_dict(state_dict)

    # model_context = nn.Sequential(*(list(model_context.children())[:-1]))


    device = torch.device("cuda:%s" %(str(args.gpu)) if torch.cuda.is_available() else "cpu")
    #opt = optim.Adam((list(emotic_model.parameters()) + list(model_context.parameters()) + list(model_body.parameters())), lr=args.learning_rate, weight_decay=args.weight_decay)
    # scheduler = StepLR(opt, step_size=7, gamma=0.1)

    # train_writer = SummaryWriter(train_log_path)
    val_writer = SummaryWriter(val_log_path)

    # training
    # train_data(opt, scheduler, [model_context, model_body, emotic_model], device, train_loader, val_loader, disc_loss, cont_loss, train_writer, val_writer, model_path, args,ind2cat, ind2vad,len(val_dataset))
    # validation
    test_data([model_context, model_body, emotic_model], device, val_loader, ind2cat, ind2vad, len(val_dataset), result_dir=result_path, test_type='val')





def check_paths(args):
    ''' Check (create if they don't exist) experiment directories.
    :param args: Runtime arguments as passed by the user.
    :return: List containing result_dir_path, model_dir_path, train_log_dir_path, val_log_dir_path.
    '''
    folders = [args.result_dir_name, args.model_dir_name]
    paths = list()
    for folder in folders:
        folder_path = os.path.join(args.experiment_path, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        paths.append(folder_path)

    log_folders = ['train', 'val']
    for folder in log_folders:
        folder_path = os.path.join(args.experiment_path, args.log_dir_name, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        paths.append(folder_path)
    return paths

if __name__ == '__main__':
    experiment_path='debug_exp'
    inference_file='sample_inference_list.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'train_test', 'inference'])
    parser.add_argument('--data_path', type=str, default='./data/emotic_pre',
                        help='Path to preprocessed data npy files/ csv files')
    parser.add_argument('--experiment_path', default=experiment_path, type=str,
                        help='Path to save experiment files (results, models, logs)')
    # parser.add_argument('--model_dir_name', type=str, default='facemodels', help='Name of the directory to save models')
    parser.add_argument('--model_dir_name', type=str, default='models', help='Name of the directory to save models')
    parser.add_argument('--result_dir_name', type=str, default='results',
                        help='Name of the directory to save results(predictions, labels mat files)')
    parser.add_argument('--log_dir_name', type=str, default='logs',
                        help='Name of the directory to save logs (train, val)')
    # parser.add_argument('--log_dir_name', type=str, default='facelogs',
    #                                       help='Name of the directory to save logs (train, val)')
    parser.add_argument('--inference_file', default=inference_file, type=str,
                        help='Text file containing image context paths and bounding box')
    parser.add_argument('--context_model', type=str, default='resnet18', choices=['resnet18', 'resnet50'],
                        help='context model type')
    parser.add_argument('--body_model', type=str, default='resnet18', choices=['resnet18', 'resnet50'],
                        help='body model type')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--cat_loss_weight', type=float, default=0.5, help='weight for discrete loss')
    parser.add_argument('--cont_loss_weight', type=float, default=0.5, help='weight fot continuous loss')
    parser.add_argument('--continuous_loss_type', type=str, default='Smooth L1', choices=['L2', 'Smooth L1'],
                        help='type of continuous loss')
    parser.add_argument('--discrete_loss_weight_type', type=str, default='dynamic',
                        choices=['dynamic', 'mean', 'static'], help='weight policy for discrete loss')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=52)  # use batch size = double(categorical emotion classes)
    # Generate args
    args = parser.parse_args()
    print('mode ', args.mode)

    result_path, model_path, train_log_path, val_log_path = check_paths(args)

    cat = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 'Disconnection', \
           'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear',
           'Happiness', \
           'Pain', 'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning']
    cat2ind = {}
    ind2cat = {}
    for idx, emotion in enumerate(cat):
        cat2ind[emotion] = idx
        ind2cat[idx] = emotion

    vad = ['Valence', 'Arousal', 'Dominance']
    ind2vad = {}
    for idx, continuous in enumerate(vad):
        ind2vad[idx] = continuous

    context_mean = [0.4690646, 0.4407227, 0.40508908]
    context_std = [0.2514227, 0.24312855, 0.24266963]
    body_mean = [0.43832874, 0.3964344, 0.3706214]
    body_std = [0.24784276, 0.23621225, 0.2323653]
    context_norm = [context_mean, context_std]
    body_norm = [body_mean, body_std]

    if args.mode == 'train':
        if args.data_path is None:
            raise ValueError('Data path not provided. Please pass a valid data path for training')
        with open(os.path.join(args.experiment_path, 'config.txt'), 'w') as f:
            print(args, file=f)
        train_emotic(result_path, model_path, train_log_path, val_log_path, ind2cat, ind2vad, context_norm, body_norm,
                     args)
    elif args.mode == 'test':
        if args.data_path is None:
            raise ValueError('Data path not provided. Please pass a valid data path for testing')
        test_emotic(result_path, model_path, ind2cat, ind2vad, context_norm, body_norm, args)
    elif args.mode == 'train_test':
        if args.data_path is None:
            raise ValueError('Data path not provided. Please pass a valid data path for training and testing')
        with open(os.path.join(args.experiment_path, 'config.txt'), 'w') as f:
            print(args, file=f)
        train_emotic(result_path, model_path, train_log_path, val_log_path, ind2cat, ind2vad, context_norm, body_norm,
                     args)
        test_emotic(result_path, model_path, ind2cat, ind2vad, context_norm, body_norm, args)
    elif args.mode == 'inference':
        if args.inference_file is None:
            raise ValueError('Inference file not provided. Please pass a valid inference file for inference')
        inference_emotic(args.inference_file, model_path, result_path, context_norm, body_norm, ind2cat, ind2vad, args)
    else:
        raise ValueError('Unknown mode')


