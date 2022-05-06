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
from sam import SAM
from skipnet import get_skipnet, SkipNet
from test import test_data, test_scikit_ap, test_vad
from tqdm import tqdm

from utility.bypass_bn import enable_running_stats, disable_running_stats
from utility.log import Log


def train_data(opt, scheduler, models, device, train_loader, val_loader, disc_loss, cont_loss, train_writer, val_writer,
               model_path, args, ind2cat, ind2vad, a_len):
    '''
    Training emotic model on train data using train loader.
    :param opt: Optimizer object.
    :param scheduler: Learning rate scheduler object.
    :param models: List containing model_context, model_body and emotic_model (fusion model) in that order.
    :param device: Torch device. Used to send tensors to GPU if available.
    :param train_loader: Dataloader iterating over train dataset.
    :param val_loader: Dataloader iterating over validation dataset.
    :param disc_loss: Discrete loss criterion. Loss measure between discrete emotion categories predictions and the target emotion categories.
    :param cont_loss: Continuous loss criterion. Loss measure between continuous VAD emotion predictions and the target VAD values.
    :param train_writer: SummaryWriter object to save train logs.
    :param val_writer: SummaryWriter object to save validation logs.
    :param model_path: Directory path to save the models after training.
    :param args: Runtime arguments.
    '''

    model_context, model_body, emotic_model = models

    emotic_model.to(device)
    model_context.to(device)
    model_body.to(device)

    print('starting training')
    last_ap = 0.0
    for e in range(args.epochs):

        running_loss = 0.0
        running_cat_loss = 0.0
        running_cont_loss = 0.0

        emotic_model.train()
        model_context.train()
        model_body.train()
        t_index = 0
        # train models for one epoch
        for images_context, images_body, labels_cat, labels_cont in tqdm(iter(train_loader)):
            t_index += 1
            images_context = images_context.to(device)
            images_body = images_body.to(device)
            labels_cat = labels_cat.to(device)
            labels_cont = labels_cont.to(device)

            opt.zero_grad()

            pred_context = model_context(images_context)
            pred_body = model_body(images_body)

            pred_cat, pred_cont = emotic_model(pred_context, pred_body)
            cat_loss_batch = disc_loss(pred_cat, labels_cat)
            # 原cont_loss_batch = cont_loss(pred_cont * 10, labels_cont * 10)
            # 尝试
            cont_loss_batch = cont_loss(pred_cont, labels_cont)

            loss = (args.cat_loss_weight * cat_loss_batch) + (args.cont_loss_weight * cont_loss_batch)

            running_loss += loss.item()
            running_cat_loss += cat_loss_batch.item()
            running_cont_loss += cont_loss_batch.item()

            loss.backward()
            opt.step()

        if e % 1 == 0:
            print('epoch = %d loss = %.4f cat loss = %.4f cont_loss = %.4f' % (
            e, running_loss / t_index, running_cat_loss / t_index, running_cont_loss / t_index))

        train_writer.add_scalar('losses/total_loss', running_loss / t_index, e)
        train_writer.add_scalar('losses/categorical_loss', running_cat_loss / t_index, e)
        train_writer.add_scalar('losses/continuous_loss', running_cont_loss / t_index, e)

        running_loss = 0.0
        running_cat_loss = 0.0
        running_cont_loss = 0.0

        emotic_model.eval()
        model_context.eval()
        model_body.eval()
        num_images = a_len
        cat_preds = np.zeros((num_images, 26))
        cat_labels = np.zeros((num_images, 26))
        cont_preds = np.zeros((num_images, 3))
        cont_labels = np.zeros((num_images, 3))

        t_index = 0
        indx = 0
        with torch.no_grad():
            # validation for one epoch
            for images_context, images_body, labels_cat, labels_cont in iter(val_loader):
                t_index += 1
                images_context = images_context.to(device)
                images_body = images_body.to(device)
                labels_cat = labels_cat.to(device)
                labels_cont = labels_cont.to(device)

                pred_context = model_context(images_context)
                pred_body = model_body(images_body)
                pred_cat, pred_cont = emotic_model(pred_context, pred_body)

                cat_loss_batch = disc_loss(pred_cat, labels_cat)
                # 原cont_loss_batch = cont_loss(pred_cont * 10, labels_cont * 10)
                # 尝试
                cont_loss_batch = cont_loss(pred_cont, labels_cont)
                loss = (args.cat_loss_weight * cat_loss_batch) + (args.cont_loss_weight * cont_loss_batch)

                cat_preds[indx: (indx + pred_cat.shape[0]), :] = pred_cat.to("cpu").data.numpy()
                cat_labels[indx: (indx + labels_cat.shape[0]), :] = labels_cat.to("cpu").data.numpy()
                cont_preds[indx: (indx + pred_cont.shape[0]), :] = pred_cont.to("cpu").data.numpy() * 10
                cont_labels[indx: (indx + labels_cont.shape[0]), :] = labels_cont.to("cpu").data.numpy() * 10
                indx = indx + pred_cat.shape[0]

                running_loss += loss.item()
                running_cat_loss += cat_loss_batch.item()
                running_cont_loss += cont_loss_batch.item()

        cat_preds = cat_preds.transpose()
        cat_labels = cat_labels.transpose()
        cont_preds = cont_preds.transpose()
        cont_labels = cont_labels.transpose()

        ap = test_scikit_ap(cat_preds, cat_labels, ind2cat)
        mean_ap = ap.mean()
        if last_ap < mean_ap:
            last_ap = mean_ap
            torch.save(emotic_model, os.path.join(model_path, f'{mean_ap:.4f}_{e}_model_emotic1.pth'))
            torch.save(model_context, os.path.join(model_path, f'{mean_ap:.4f}_{e}_model_context1.pth'))
            torch.save(model_body, os.path.join(model_path, f'{mean_ap:.4f}_{e}_model_body1.pth'))

        test_vad(cont_preds, cont_labels, ind2vad)
        # thresholds = get_thresholds(cat_preds, cat_labels)
        print('completed testing')
        if e % 1 == 0:
            print('epoch = %d validation loss = %.4f cat loss = %.4f cont loss = %.4f ' % (
            e, running_loss / t_index, running_cat_loss / t_index, running_cont_loss / t_index))

        val_writer.add_scalar('losses/total_loss', running_loss / t_index, e)
        val_writer.add_scalar('losses/categorical_loss', running_cat_loss / t_index, e)
        val_writer.add_scalar('losses/continuous_loss', running_cont_loss / t_index, e)

        scheduler.step()

    print('completed training')
    emotic_model.to("cpu")
    model_context.to("cpu")
    model_body.to("cpu")
    torch.save(emotic_model, os.path.join(model_path, 'model_emotic1.pth'))
    torch.save(model_context, os.path.join(model_path, 'model_context1.pth'))
    torch.save(model_body, os.path.join(model_path, 'model_body1.pth'))
    print('saved models')


def train_emotic(result_path, model_path, train_log_path, val_log_path, ind2cat, ind2vad, context_norm, body_norm,
                 args):
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

    print('train ', 'context ', train_context.shape, 'body', train_body.shape, 'cat ', train_cat.shape, 'cont',
          train_cont.shape)
    print('val ', 'context ', val_context.shape, 'body', val_body.shape, 'cat ', val_cat.shape, 'cont', val_cont.shape)

    # Initialize Dataset and DataLoader
    train_transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(),
                                          transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                          transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

    train_dataset = Emotic_PreDataset(train_context, train_body, train_cat, train_cont, train_transform, context_norm,
                                      body_norm)
    val_dataset = Emotic_PreDataset(val_context, val_body, val_cat, val_cont, test_transform, context_norm, body_norm)

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)

    print('train loader ', len(train_loader), 'val loader ', len(val_loader))

    # Prepare models
    model_context, model_body = prep_models(context_model=args.context_model, body_model=args.body_model,
                                            model_dir=model_path)
    use_model = 'skipnet'
    if use_model != 'skipnet':
        model_body = nn.Sequential(*(list(model_body.children())[:-1]))
        emotic_model = Emotic(list(model_context.children())[-1].in_features,
                              list(model_body.children())[-1].in_features)
    else:
        model_body = get_skipnet(num_classes=512, skip_w=8)

        model_body = torch.load(r'debug_exp/models/0.3542_13_model_body1.pth',
                                map_location=lambda storage, loc: storage)

        # state_dict = {str.replace(k, 'module.', ''): v for k, v in model_weight['state_dict'].items()}
        # model_body.load_state_dict(state_dict)

        emotic_model = Emotic(list(model_context.children())[-1].in_features,
                              list(model_body.children())[-1].out_features)

        emotic_model = torch.load(r'debug_exp/models/0.3542_13_model_emotic1.pth',
                                  map_location=lambda storage, loc: storage)

        # state_dict = {str.replace(k, 'module.', ''): v for k, v in model_weight['state_dict'].items()}
        # emotic_model.load_state_dict(state_dict)

    model_context = nn.Sequential(*(list(model_context.children())[:-1]))

    for param in emotic_model.parameters():
        param.requires_grad = True
    for param in model_context.parameters():
        param.requires_grad = True
    for param in model_body.parameters():
        param.requires_grad = True

    device = torch.device("cuda:%s" % (str(args.gpu)) if torch.cuda.is_available() else "cpu")
    # opt = optim.Adam((list(emotic_model.parameters()) + list(model_context.parameters()) + list(model_body.parameters())), lr=args.learning_rate, weight_decay=args.weight_decay)
    # scheduler = StepLR(opt, step_size=7, gamma=0.1)

    '''原opt = optim.SGD(
        (list(emotic_model.parameters()) + list(model_context.parameters()) + list(model_body.parameters())),
        momentum=0.9, lr=args.learning_rate, weight_decay=args.weight_decay)'''


    #添加SAM
    dataset = Emotic_PreDataset(args.batch_size, args.threads)
    log = Log(log_each=10)
    model = SkipNet(args.depth, args.width_factor).to(device)
    base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
    opt= SAM((list(emotic_model.parameters()) + list(model_context.parameters()) + list(model_body.parameters())),base_optimizer,
             lr=0.1,  momentum=0.9)
    scheduler = StepLR(opt, step_size=7, gamma=0.5)

    for epoch in range(args.epochs):
        model.train()
        log.train(len_dataset=len(dataset.train))

        for batch in dataset.train:
            inputs, targets = (b.to(device) for b in batch)

            # first forward-backward step
            enable_running_stats(model)
            predictions = model(inputs)
            #原loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
            disc_loss = DiscreteLoss(predictions, targets, DiscreteLoss=args.label_smoothing)
            disc_loss.mean().backward()
            opt.first_step(zero_grad=True)

            # second forward-backward step
            disable_running_stats(model)
            DiscreteLoss(model(inputs), targets, smoothing=args.label_smoothing).mean().backward()
            opt.second_step(zero_grad=True)

            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                log(model, disc_loss.cpu(), correct.cpu(), scheduler.lr())
                scheduler(epoch)

        model.eval()
        log.eval(len_dataset=len(dataset.test))

        with torch.no_grad():
            for batch in dataset.test:
                inputs, targets = (b.to(device) for b in batch)

                predictions = model(inputs)

                disc_loss= DiscreteLoss(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                log(model, disc_loss.cpu(), correct.cpu())




        #disc_loss = DiscreteLoss(args.discrete_loss_weight_type, device)

                if args.continuous_loss_type == 'Smooth L1':
                    cont_loss = ContinuousLoss_SL1()
                else:
                    cont_loss = ContinuousLoss_L2()
                opt.first_step(zero_grad=True)

                if args.continuous_loss_type == 'Smooth L1':
                    cont_loss = ContinuousLoss_SL1()
                else:
                    cont_loss = ContinuousLoss_L2()
    log.flush()

    train_writer = SummaryWriter(train_log_path)
    val_writer = SummaryWriter(val_log_path)

    # training
    train_data(opt, scheduler, [model_context, model_body, emotic_model], device, train_loader, val_loader, disc_loss,
               cont_loss, train_writer, val_writer, model_path, args, ind2cat, ind2vad, len(val_dataset))
    # validation
    test_data([model_context, model_body, emotic_model], device, val_loader, ind2cat, ind2vad, len(val_dataset),
              result_dir=result_path, test_type='val')
