from __future__ import print_function
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.model_zoo import load_url as load_state_dict_from_url
import timm
from tqdm import tqdm

from utils import *
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.metrics import precision_recall_fscore_support as score
import argparse
import torch.nn as nn
import torch.nn.functional as F              # needed for KD
from torchvision import transforms           # transforms used below


# Parse command-line arguments
parser = argparse.ArgumentParser(description='Organize Dataset')
parser.add_argument('--m', required=False, type=str, default='mobilenetv2_100', help='Model name')
parser.add_argument('--set', required=False, type=str, default='SFD',
                    choices=['AUCD', 'SFD'],
                    help='Dataset choice: AUCD or SFD')

args, unparsed = parser.parse_known_args()

seed = 0
seed_everything(seed)
input_size = 224


def freeze(net, freeze=True):
    """
    Freeze or unfreeze all parameters of a network.
    Args:
        net: the model to freeze/unfreeze
        freeze (bool): if True, freeze parameters; else unfreeze
    """
    if freeze:
        for param in net.parameters():
            param.requires_grad = False
        net.eval()
    else:
        for param in net.parameters():
            param.requires_grad = True
        net.train()


class NegativeL1Loss(nn.Module):
    """
    Negative L1 Loss with normalization to [-1, 1].
    Measures negative similarity between two tensors.
    """
    def __init__(self):
        super(NegativeL1Loss, self).__init__()

    def forward(self, x1, x2):
        epsilon = 1e-8  # small value to avoid division by zero

        # Normalize x1 to [0, 1]
        x1_min = x1.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        x1_max = x1.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        x1_range = x1_max - x1_min + epsilon
        x1_normalized = (x1 - x1_min) / x1_range

        # Normalize x2 to [0, 1]
        x2_min = x2.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        x2_max = x2.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        x2_range = x2_max - x2_min + epsilon
        x2_normalized = (x2 - x2_min) / x2_range

        # Rescale to [-1, 1]
        x1_normalized = 2 * x1_normalized - 1
        x2_normalized = 2 * x2_normalized - 1

        loss_fn = nn.L1Loss()
        loss = -loss_fn(x1_normalized, x2_normalized)
        return loss


class Generator(nn.Module):
    """
    Generator module that processes an input image and features,
    progressively upsamples the features and combines them with processed images.
    """
    def __init__(self, input_dim, num_layers, image_dim=6):
        super(Generator, self).__init__()

        # Initial image processing block
        image_processor = [
            nn.Conv2d(3, image_dim, 5, stride=1, padding=2),
            nn.InstanceNorm2d(image_dim),
            nn.ReLU(inplace=True)
        ]
        self.image_processor = nn.Sequential(*image_processor)

        # Progressive upsampling generator blocks
        layers = []
        for _ in range(num_layers):
            out_dim = input_dim // 2
            layers += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(input_dim, out_dim, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_dim),
                nn.ReLU(inplace=True)
            ]
            input_dim = out_dim

        layers += [nn.Conv2d(input_dim, image_dim, 3, stride=1, padding=1), nn.ReLU(inplace=True)]
        self.generator = nn.Sequential(*layers)

        # Output layer combining image and generated feature maps
        output_layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(image_dim * 2, 3, 7),
            nn.Tanh()
        ]
        self.output = nn.Sequential(*output_layers)

    def forward(self, im, feature):
        im_processed = self.image_processor(im)
        feature_processed = self.generator(feature)
        combined = torch.cat((im_processed, feature_processed), dim=1)
        output = self.output(combined)
        # Optional interpolation commented out:
        # output = F.interpolate(output, size=(input_size, input_size), mode='bilinear', align_corners=False)
        return output


class Features(nn.Module):
    """
    Feature extractor that splits the base network into meaningful blocks.
    Extracts features at different depths of the network.
    """
    def __init__(self, net_layers):
        super(Features, self).__init__()

        # Extract children of the 3rd layer group for finer layers
        net_layers_ = list(net_layers[2].children())

        # Define sequential blocks from network layers
        self.net_layer_0 = nn.Sequential(*net_layers[0:2])
        self.net_layer_1 = nn.Sequential(*net_layers_[0:3])
        self.net_layer_2 = nn.Sequential(*net_layers_[3:5])
        self.net_layer_3 = nn.Sequential(*net_layers_[5:7])
        self.net_layer_4 = nn.Sequential(*net_layers[3:5])

    def forward(self, x):
        x = self.net_layer_0(x)
        x = self.net_layer_1(x)
        x1 = x
        x = self.net_layer_2(x)
        x2 = x
        x = self.net_layer_3(x)
        x = self.net_layer_4(x)
        x3 = x
        return x1, x2, x3


class Network_Wrapper(nn.Module):
    """
    Wraps the feature extractor and multiple classifier blocks
    to produce multi-scale predictions.
    """
    def __init__(self, net_layers, num_class):
        super().__init__()

        self.Features = Features(net_layers)

        # Max pooling layers for spatial reduction
        self.max_pool1 = nn.MaxPool2d(kernel_size=28, stride=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=14, stride=1)
        self.max_pool3 = nn.MaxPool2d(kernel_size=7, stride=1)

        # Convolutional blocks followed by classifiers for each feature scale
        self.conv_block1 = nn.Sequential(
            BasicConv(32, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_class)
        )

        self.conv_block2 = nn.Sequential(
            BasicConv(96, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_class)
        )

        self.conv_block3 = nn.Sequential(
            BasicConv(1280, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_class)
        )

        # Classifier for concatenated features from all scales
        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(1024 * 3),
            nn.Linear(1024 * 3, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_class)
        )

    def forward(self, x):
        # Extract multi-scale features
        x1, x2, x3 = self.Features(x)

        # Process first scale
        map1 = x1.clone()
        x1_ = self.conv_block1(x1)
        x1_ = self.max_pool1(x1_)
        x1_f = x1_.view(x1_.size(0), -1)
        x1_c = self.classifier1(x1_f)

        # Process second scale
        map2 = x2.clone()
        x2_ = self.conv_block2(x2)
        x2_ = self.max_pool2(x2_)
        x2_f = x2_.view(x2_.size(0), -1)
        x2_c = self.classifier2(x2_f)

        # Process third scale
        map3 = x3.clone()
        x3_ = self.conv_block3(x3)
        x3_ = self.max_pool3(x3_)
        x3_f = x3_.view(x3_.size(0), -1)
        x3_c = self.classifier3(x3_f)

        # Concatenate all features and classify
        x_c_all = torch.cat((x1_f, x2_f, x3_f), dim=-1)
        f = x_c_all.clone()
        x_c_all = self.classifier_concat(x_c_all)

        return x1_c, x2_c, x3_c, x_c_all, map1, map2, map3, f



def inference(net, criterion, batch_size, test_path):
    """
    Perform inference on the test dataset.
    Args:
        net: the model to evaluate
        criterion: loss function
        batch_size: batch size for dataloader
        test_path: path to test dataset
    Returns:
        Multiple metrics: accuracy, f1_micro, f1_macro, auc_micro, auc_macro,
                         precision_micro, precision_macro, recall_micro, recall_macro
    """
    net.eval()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    net.to(device)

    transform_test = transforms.Compose([
        transforms.Resize((int(input_size / 0.875), int(input_size / 0.875))),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    test_set = torchvision.datasets.ImageFolder(root=test_path, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    test_loss = 0
    correct = 0
    total = 0
    score_list = []
    target_list = []
    pred_list = []

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if use_cuda:
            inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = Variable(inputs), Variable(targets)

        with torch.no_grad():
            output_1, output_2, output_3, output_concat, _, _, _, _ = net(inputs)
            outputs = output_1 + output_2 + output_3 + output_concat

        # Store softmax probabilities
        score_list.append(outputs.softmax(dim=1).cpu())
        loss = criterion(outputs, targets)
        test_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        pred_list.append(predicted.cpu().unsqueeze(0))
        target_list.append(targets.cpu().unsqueeze(0))

        total += targets.size(0)
        correct += predicted.eq(targets).cpu().sum().item()

        if batch_idx % 50 == 0 or batch_idx == len(test_loader) - 1:
            print(f'Step: {batch_idx} | Loss: {test_loss / (batch_idx + 1):.3f} | '
                  f'Acc: {100. * correct / total:.3f}% ({correct}/{total})')

    pred_list = torch.cat(pred_list, dim=-1).squeeze().numpy()
    target_list = torch.cat(target_list, dim=-1).squeeze().numpy()
    score_list = torch.cat(score_list, dim=0).squeeze().numpy()

    accuracy = accuracy_score(pred_list, target_list) * 100
    f1_micro = f1_score(target_list, pred_list, average='micro')
    f1_macro = f1_score(target_list, pred_list, average='macro')

    precision_micro, recall_micro, _, _ = score(torch.from_numpy(target_list), torch.from_numpy(pred_list), average='micro')
    precision_macro, recall_macro, _, _ = score(torch.from_numpy(target_list), torch.from_numpy(pred_list), average='macro')

    auc_micro = roc_auc_score(target_list, score_list, multi_class='ovr', average='micro')
    auc_macro = roc_auc_score(target_list, score_list, multi_class='ovr', average='macro')

    test_acc = 100. * correct / total
    test_loss /= (batch_idx + 1)
    print(f"Test Accuracy: {test_acc}%")

    return accuracy, f1_micro, f1_macro, auc_micro, auc_macro, precision_micro, precision_macro, recall_micro, recall_macro


def test(net, criterion, batch_size, test_path):
    """
    Evaluate the model on the test dataset with detailed accuracy calculation.
    Args:
        net: model to test
        criterion: loss function
        batch_size: batch size
        test_path: dataset path
    Returns:
        test_acc_en: ensemble accuracy
        test_loss: average loss
    """
    net.eval()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    net.to(device)

    transform_test = transforms.Compose([
        transforms.Resize((int(input_size / 0.875), int(input_size / 0.875))),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    testset = torchvision.datasets.ImageFolder(root=test_path, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4)

    test_loss = 0
    correct = 0
    correct_com = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            output_1, output_2, output_3, output_concat, _, _, _, _ = net(inputs)

            outputs_com = output_1 + output_2 + output_3 + output_concat

            loss = criterion(output_concat, targets)

        test_loss += loss.item()
        _, predicted = torch.max(output_concat.data, 1)
        _, predicted_com = torch.max(outputs_com.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets).cpu().sum().item()
        correct_com += predicted_com.eq(targets).cpu().sum().item()

    test_acc_en = 100. * correct_com / total
    test_loss /= (batch_idx + 1)

    return test_acc_en, test_loss


# --------------------- KD settings & function --------------------- #
KD_T = 2.0       # temperature
KD_ALPHA = 0.5   # distillation loss weight

def kd_loss(student_logits, teacher_logits, T=KD_T):
    """
    Standard knowledge distillation loss:
    KL( softmax(teacher/T) || log_softmax(student/T) ) * T^2
    """
    log_p_s = F.log_softmax(student_logits / T, dim=1)
    p_t = F.softmax(teacher_logits / T, dim=1).detach()
    return F.kl_div(log_p_s, p_t, reduction='batchmean') * (T * T)
# ----------------------------------------------------------------- #


def train(nb_epoch, batch_size, store_name, start_epoch=0, num_class=0, data_path=''):
    """
    Train the model for nb_epoch epochs.
    Args:
        nb_epoch: total number of epochs to train
        batch_size: batch size for training
        store_name: directory to save models and results
        start_epoch: epoch to start training from (for resuming)
        num_class: number of output classes
        data_path: root path for training data
    """
    exp_dir = store_name
    vis_dir = os.path.join(store_name, "visualizations")

    # Create directories if they don't exist
    try:
        os.stat(exp_dir)
    except:
        os.makedirs(exp_dir)

    use_cuda = torch.cuda.is_available()
    print(use_cuda)

    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.Resize((int(input_size / 0.875), int(input_size / 0.875))),
        transforms.RandomCrop(input_size, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = torchvision.datasets.ImageFolder(root=data_path + '/train', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Load backbone model
    net = timm.create_model(args.m, pretrained=True)
    net_layers = list(net.children())

    net = Network_Wrapper(net_layers, num_class)

    # Wrap model for multi-GPU training
    netp = torch.nn.DataParallel(net, device_ids=[0])

    # Instantiate generators
    g1 = Generator(32, 3).cuda()
    g2 = Generator(96, 4).cuda()
    g3 = Generator(1280, 5).cuda()

    device = torch.device("cuda")
    net.to(device)

    # Define loss functions
    CELoss = nn.CrossEntropyLoss()
    MAEloss = nn.L1Loss()
    NegLoss = NegativeL1Loss()
    BCELoss = nn.BCELoss()

    # Optimizer for network
    optimizer = optim.SGD([
        {'params': net.classifier_concat.parameters(), 'lr': 0.002},
        {'params': net.conv_block1.parameters(), 'lr': 0.002},
        {'params': net.classifier1.parameters(), 'lr': 0.002},
        {'params': net.conv_block2.parameters(), 'lr': 0.002},
        {'params': net.classifier2.parameters(), 'lr': 0.002},
        {'params': net.conv_block3.parameters(), 'lr': 0.002},
        {'params': net.classifier3.parameters(), 'lr': 0.002},
        {'params': net.Features.parameters(), 'lr': 0.0002}
    ],
        momentum=0.9, weight_decay=5e-4)

    # Optimizer for generators
    optimizer_g = optim.Adam(
        list(g1.parameters()) + list(g2.parameters()) + list(g3.parameters()),
        lr=0.0002, betas=(0.5, 0.999)
    )

    max_val_acc = 0
    lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]

    for epoch in tqdm(range(start_epoch, nb_epoch), desc="Epoch Progress"):
        net.train()
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss3 = 0
        train_loss4 = 0
        train_loss5 = 0
        correct = 0
        total = 0
        idx = 0

        for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader),
                                                total=len(trainloader),
                                                desc=f"Batch Progress (Epoch {epoch})",
                                                leave=False):
            idx = batch_idx
            if inputs.shape[0] < batch_size:
                continue
            if use_cuda:
                inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = Variable(inputs), Variable(targets)

            # Update learning rates with cosine annealing schedule
            for nlr in range(len(optimizer.param_groups)):
                optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, lr[nlr])

            # Freeze main net, train generators
            freeze(netp, True)
            freeze(g1, False)
            freeze(g2, False)
            freeze(g3, False)

            high_freq = compute_high_freq(inputs)  # High pass filter

            optimizer_g.zero_grad()
            inputs0 = inputs.clone().detach()

            output_1, output_2, output_3, _, map1, map2, map3, _ = netp(inputs0)
            low1 = g1(inputs, map1)
            comp1 = torch.clamp(high_freq + low1, -1, 1)
            low2 = g2(inputs, map2)
            comp2 = torch.clamp(high_freq + low2, -1, 1)
            low3 = g3(inputs, map3)
            comp3 = torch.clamp(high_freq + low3, -1, 1)

            # Compute losses for generator outputs on comp1
            output_1_im1, output_2_im1, output_3_im1, _, map1_im1, _, _, f_im1 = netp(comp1)
            loss_g = CELoss(output_2_im1, targets) + MAEloss(output_2_im1, output_2) + \
                     CELoss(output_3_im1, targets) + MAEloss(output_3_im1, output_3) + \
                     + NegLoss(map1_im1, map1)

            # For comp2 input
            output_1_im2, output_2_im2, output_3_im2, _, _, map2_im2, _, f_im2 = netp(comp2)
            loss_g = loss_g + CELoss(output_1_im2, targets) + MAEloss(output_1_im2, output_1) + \
                     CELoss(output_3_im2, targets) + MAEloss(output_3_im2, output_3) + \
                     + NegLoss(map2_im2, map2)

            # For comp3 input
            output_1_im3, output_2_im3, output_3_im3, _, _, _, map3_im3, f_im3 = netp(comp3)
            loss_g = loss_g + CELoss(output_2_im3, targets) + MAEloss(output_2_im3, output_2) + \
                     CELoss(output_1_im3, targets) + MAEloss(output_1_im3, output_3) + \
                     + NegLoss(map3_im3, map3)

            loss_g.backward()
            optimizer_g.step()

            # Freeze generators and main net accordingly
            freeze(netp, True)
            freeze(g1, True)
            freeze(g2, True)
            freeze(g3, True)

            # ====== fetch maps and teacher logits, then average as teacher ======
            with torch.no_grad():
                base_o1, base_o2, base_o3, base_oc, map1, map2, map3, _ = netp(inputs)
                # base_o_avg = (base_o1 + base_o2 + base_o3 + base_oc) / 4.0
                base_o_avg = base_oc
            # ====================================================================

            low1 = g1(inputs, map1)
            comp1 = torch.clamp(high_freq + low1, -1, 1)
            low2 = g2(inputs, map2)
            comp2 = torch.clamp(high_freq + low2, -1, 1)
            low3 = g3(inputs, map3)
            comp3 = torch.clamp(high_freq + low3, -1, 1)

            freeze(netp, False)
            freeze(g1, True)
            freeze(g2, True)
            freeze(g3, True)

            # ---- branch 1 ----
            optimizer.zero_grad()
            if epoch >= 5:
                output_1, _, _, _, _, _, _, _ = netp(comp1)
            else:
                output_1, _, _, _, _, _, _, _ = netp(inputs)
            # KD to averaged teacher
            loss1 = CELoss(output_1, targets) * 1 + KD_ALPHA * kd_loss(output_1, base_o_avg)
            loss1.backward()
            optimizer.step()

            # ---- branch 2 ----
            optimizer.zero_grad()
            if epoch >= 5:
                _, output_2, _, _, _, _, _, _ = netp(comp2)
            else:
                _, output_2, _, _, _, _, _, _ = netp(inputs)
            # KD to averaged teacher
            loss2 = CELoss(output_2, targets) * 1 + KD_ALPHA * kd_loss(output_2, base_o_avg)
            loss2.backward()
            optimizer.step()

            # ---- branch 3 ----
            optimizer.zero_grad()
            if epoch >= 5:
                _, _, output_3, _, _, _, _, _ = netp(comp3)
            else:
                _, _, output_3, _, _, _, _, _ = netp(inputs)
            # KD to averaged teacher
            loss3 = CELoss(output_3, targets) * 1 + KD_ALPHA * kd_loss(output_3, base_o_avg)
            loss3.backward()
            optimizer.step()

            # concat branch (unchanged, no KD here per your original design)
            optimizer.zero_grad()
            inputs_concat = inputs.clone().detach()
            _, _, _, output_concat, _, _, _, _ = netp(inputs_concat)
            concat_loss = CELoss(output_concat, targets) * 1
            concat_loss.backward()
            optimizer.step()

            _, predicted = torch.max(output_concat.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            train_loss += (loss1.item() + loss2.item() + loss3.item() + concat_loss.item())
            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            train_loss3 += loss3.item()
            train_loss4 += concat_loss.item()

            if batch_idx == (len(trainloader) - 2):
                last_batch_cache = {
                    'orig': inputs.detach(),
                    'high': high_freq.detach(),
                    'low_orig': (inputs - high_freq).detach(),
                    'g_lows': [low1.detach(), low2.detach(), low3.detach()],
                    'comps': [comp1.detach(), comp2.detach(), comp3.detach()],
                }

                save_combined_images(last_batch_cache['orig'],
                                     last_batch_cache['high'],
                                     last_batch_cache['low_orig'],
                                     last_batch_cache['g_lows'],
                                     last_batch_cache['comps'],
                                     vis_dir, epoch)

        train_acc = 100. * float(correct) / total
        train_loss = train_loss / (idx + 1)
        with open(exp_dir + '/results_train.txt', 'a') as file:
            file.write(
                'Iteration %d | train_acc = %.5f | train_loss = %.5f | Loss1: %.3f | Loss2: %.5f | Loss3: %.5f | Loss_ATT: %.5f | Loss_concat: %.5f |\n' % (
                    epoch, train_acc, train_loss, train_loss1 / (idx + 1), train_loss2 / (idx + 1), train_loss3 / (idx + 1),
                    train_loss4 / (idx + 1), train_loss5 / (idx + 1)))

        # Validation and model saving
        val_acc_com, val_loss = test(net, CELoss, 3, data_path + '/validation')
        if val_acc_com > max_val_acc:
            max_val_acc = val_acc_com
            net.cpu()
            torch.save(net, './' + store_name + '/model.pth')
            net.to(device)

            g1.cpu()
            torch.save(g1, f'./{store_name}/model_g1.pth')
            g1.to(device)

            g2.cpu()
            torch.save(g2, f'./{store_name}/model_g2.pth')
            g2.to(device)

            g3.cpu()
            torch.save(g3, f'./{store_name}/model_g3.pth')
            g3.to(device)

        with open(exp_dir + '/results_test.txt', 'a') as file:
            file.write('Iteration %d, test_acc_combined = %.5f, test_loss = %.6f\n' % (
                epoch, val_acc_com, val_loss))

    # Final inference after training
    trained_model = torch.load('./' + store_name + '/model.pth')

    if args.set == 'SFD':
        test_folders = ['StoA', 'StoB', 'StoL']
    elif args.set == 'AUCD':
        test_folders = ['AtoB', 'AtoS', 'AtoL']
    else:
        raise ValueError(f"Unknown dataset set: {args.set}")

    with open(exp_dir + '/results_test.txt', 'a') as file:
        for folder in test_folders:
            test_path = os.path.join(data_path, folder)
            accuracy, f1_micro, f1_macro, auc_micro, auc_macro, precision_micro, precision_macro, recall_micro, recall_macro = inference(
                trained_model, CELoss, 3, test_path)
            
            file.write(f'[{folder}]\n')
            file.write('Inference Results: Accuracy = %.5f, F1_micro = %.5f, F1_macro = %.5f, Auc_micro = %.5f, Auc_macro = %.5f\n' %
                    (accuracy, f1_micro, f1_macro, auc_micro, auc_macro))
            file.write('Inference Results: precision_micro = %.5f, precision_macro = %.5f, recall_micro = %.5f, recall_macro = %.5f\n\n' %
                    (precision_micro, precision_macro, recall_micro, recall_macro))

if __name__ == '__main__':
    # Define dataset root path based on args
    data_path = './datasets/organized/' + args.set
    num_class = 10  # Number of output classes

    # Create main results directory
    results_path = 'results'
    mk_dir(results_path)

    # Extract script name, remove extension and prefix
    pyname = os.path.basename(__file__).replace('.py', '').replace('train_models_', '')

    # Define classification task results folder
    task_result_path = os.path.join(results_path, 'classification')
    mk_dir(task_result_path)

    # Define specific experiment folder with details
    experiment_result_path = os.path.join(
        task_result_path,
        "{}_{}_seed_{}_input_size_{}_set_{}".format(pyname, args.m, seed, input_size, args.set)
    )
    mk_dir(experiment_result_path)

    # Start training with fixed parameters
    train(
        nb_epoch=100,                 # Total number of epochs
        batch_size=16,                # Batch size
        store_name=experiment_result_path,  # Directory to save models and logs
        start_epoch=0,                # Start from scratch
        num_class=num_class,
        data_path=data_path
    )
