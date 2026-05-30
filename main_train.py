import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import warnings
from datetime import datetime

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from dataloader import AVQA_dataset, ToTensor
from nets.net import TSPM
from configs.arguments import parser

TIMESTAMP = "{0:%Y-%m-%d-%H-%M-%S/}".format(datetime.now())
warnings.filterwarnings('ignore')

print("\n--------------- TSPM --------------- \n")


def train(args, model, train_loader, optimizer, criterion, writer, epoch, device):
    model.train()
    for batch_idx, sample in enumerate(train_loader):
        audios_feat = sample['audios_feat'].to(device)
        visual_feat = sample['visual_feat'].to(device)
        audios_patch_feat = sample['audios_patch_feat'].to(device)
        visual_patch_feat = sample['visual_patch_feat'].to(device)
        target = sample['answer_label'].to(device)
        question = sample['question_feat'].to(device)
        question_prompt = sample['question_prompt'].to(device)

        optimizer.zero_grad()
        output_qa = model(audios_feat, visual_feat, audios_patch_feat, visual_patch_feat, question, question_prompt)
        loss = criterion(output_qa, target)

        writer.add_scalar('run/both', loss.item(), epoch * len(train_loader) + batch_idx)

        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(audios_feat), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def eval(model, val_loader, writer, epoch, device):
    model.eval()
    total_qa = 0
    correct_qa = 0

    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            audios_feat = sample['audios_feat'].to(device)
            visual_feat = sample['visual_feat'].to(device)
            audios_patch_feat = sample['audios_patch_feat'].to(device)
            visual_patch_feat = sample['visual_patch_feat'].to(device)
            target = sample['answer_label'].to(device)
            question = sample['question_feat'].to(device)
            question_prompt = sample['question_prompt'].to(device)

            preds_qa = model(audios_feat, visual_feat, audios_patch_feat, visual_patch_feat, question, question_prompt)
            _, predicted = torch.max(preds_qa.data, 1)
            total_qa += preds_qa.size(0)
            correct_qa += (predicted == target).sum().item()

    print('Current Acc: %.2f %%' % (100 * correct_qa / total_qa))
    writer.add_scalar('metric/acc_qa', 100 * correct_qa / total_qa, epoch)
    return 100 * correct_qa / total_qa


def main():
    args = parser.parse_args()

    if not hasattr(args, 'AV_Attn_Module'):
        args.AV_Attn_Module = True
    if not hasattr(args, 'Temp_QTGM'):
        args.Temp_QTGM = True
    if not hasattr(args, 'log_interval'):
        args.log_interval = 10

    os.makedirs(args.model_save_dir, exist_ok=True)

    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    torch.manual_seed(args.seed)

    writer = SummaryWriter('runs/strn/' + TIMESTAMP + '_' + args.checkpoint)

    model = TSPM(args)
    if torch.cuda.is_available():
        model = nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    train_dataset = AVQA_dataset(
        label=args.label_train,
        args=args,
        audios_feat_dir=args.audios_feat_dir,
        visual_feat_dir=args.visual_feat_dir,
        audios_patch_dir=args.audios_patch_dir,
        visual_patch_dir=args.visual_patch_dir,
        qst_prompt_dir=args.qst_prompt_dir,
        qst_feat_dir=args.qst_feat_dir,
        transform=transforms.Compose([ToTensor()]),
        mode_flag='train'
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_dataset = AVQA_dataset(
        label=args.label_val,
        args=args,
        audios_feat_dir=args.audios_feat_dir,
        visual_feat_dir=args.visual_feat_dir,
        audios_patch_dir=args.audios_patch_dir,
        visual_patch_dir=args.visual_patch_dir,
        qst_prompt_dir=args.qst_prompt_dir,
        qst_feat_dir=args.qst_feat_dir,
        transform=transforms.Compose([ToTensor()]),
        mode_flag='val'
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        train(args, model, train_loader, optimizer, criterion, writer, epoch=epoch, device=device)
        scheduler.step()
        current_acc = eval(model, val_loader, writer, epoch, device)

        if current_acc >= best_acc:
            best_acc = current_acc
            best_epoch = epoch
            # Save properly handling DataParallel wrapper
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            save_path = os.path.join(args.model_save_dir, args.checkpoint + ".pt")
            torch.save(state_dict, save_path)
            print(f"Checkpoint saved to {save_path}")

        print("Best Acc: %.2f %%" % best_acc)
        print("Best Epoch: ", best_epoch)
        print("*" * 20)


if __name__ == '__main__':
    main()