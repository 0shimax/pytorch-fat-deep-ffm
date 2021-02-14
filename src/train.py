import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch; torch.manual_seed(555)
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

from model.optimizer.sam import SAM
from model.fat_deep_ffm import FieldAttentiveDeepFieldAwareFactorizationMachineModel as FatFFM
from data.data_loader import CtrDataset, loader
from feature.build import read_df


def main(args):
    model = FatFFM(
        len(args.numerical_cols.split(",")) if args.numerical_cols is not None else 0
        ,len(args.categorical_cols.split(","))
        ,list(map(int, args.num_ids.split(",")))
        ,args.embed_size
        ,args.deep_output_size
        ,list(map(int, args.deep_layer_sizes.split(",")))
        ,args.reduction
        ,args.ffm_dropout_p
        ,list(map(float, args.deep_dropout_p.split(",")))
        ,n_class=args.n_class)

    if Path(args.resume_model).exists():
        print("load model:", args.resume_model)
        model.load_state_dict(torch.load(args.resume_model))

    # setup optimizer
    # optimizer = SAM(model.parameters(), 
    #                 optim.Adam, 
    #                 lr=args.lr, 
    #                 betas=(args.beta1, 0.999))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    train_dataset = CtrDataset(read_df(args.root_dir, args.train_file_name, args.categorical_cols.split(",")),
                               args.numerical_cols.split(",") if args.numerical_cols is not None else [],
                               args.categorical_cols.split(","),
                               salt=True)
    test_dataset = CtrDataset(read_df(args.root_dir, args.test_file_name, args.categorical_cols.split(",")), 
                              args.numerical_cols.split(",") if args.numerical_cols is not None else [], 
                              args.categorical_cols.split(","),
                              salt=False)
    train_loader = loader(train_dataset, args.batch_size)
    test_loader = loader(test_dataset, 1, shuffle=False)

    train(args, model, optimizer, train_loader)
    test(args, model, test_loader)


def train(args, model, optimizer, data_loader):
    model.train()
    for epoch in range(args.epochs):
        for i, (numerical_data, categorical_data, target) in enumerate(data_loader):
            model.zero_grad()

            optimizer.zero_grad()
            if args.numerical_cols is None:
                output, attention = model(categorical_data)
            else:
                output, attention = model(categorical_data, numerical_data)
            n_batch = output.shape[0]
            loss = F.nll_loss(F.log_softmax(output), target.squeeze(1))
            loss.backward()

            optimizer.step()
            print('[{}/{}][{}/{}] Loss: {:.4f}'.format(
                  epoch, args.epochs, i,
                  len(data_loader), loss.item()))

        # do checkpointing
        torch.save(model.state_dict(),
                   '{}/fat_ffm_ckpt.pth'.format(args.out_dir))
    torch.save(model.state_dict(),
                '{}/fat_ffm_ckpt.pth'.format(args.out_dir))


def test(args, model, data_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i, (numerical_data, categorical_data, target) in enumerate(data_loader):
            if args.numerical_cols is None:
                output, attention = model(categorical_data)
            else:
                output, attention = model(categorical_data, numerical_data)

            # sum up batch loss
            test_loss += torch.mean(F.nll_loss(
                output, target.squeeze(1), size_average=False)).item()
            # get the index of the max log-probability
            pred = output.argmax(1)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
          .format(test_loss, correct, len(data_loader.dataset),
                  100. * correct / len(data_loader.dataset)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', default='./data/dummy', help='path to dataset')
    parser.add_argument('--embed-size', type=int, default=32)
    parser.add_argument('--deep-output_size', type=int, default=32)
    parser.add_argument('--deep-layer-sizes', default="256,128")
    parser.add_argument('--reduction', type=int, default=1)
    parser.add_argument('--ffm-dropout-p', type=float, default=0.5)
    parser.add_argument('--deep-dropout-p', default="0.5,0.5")
    parser.add_argument('--n-class', type=int, default=2, help='number of class')
    parser.add_argument('--train-file-name', default='train_dummy.csv', help='path to train data')
    parser.add_argument('--test-file-name', default='test_dummy.csv', help='path to test data')
    parser.add_argument('--resume-model', default='./results/fat_ffm_ckpt.pth_', help='path to trained model')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch-size', type=int, default=256, help='input batch size')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--numerical-cols', default=None, help='numerical column names (comma split as string)')
    parser.add_argument('--categorical-cols', default='creative_ad_type,imp_tagid,device,categories,slot_size,imp_banner_pos', help='categorical column names (comma split as string)')
    parser.add_argument('--num-ids', default="10,20000,4,100,100,10")    
    parser.add_argument('--out-dir', default='./results', help='folder to output images and model checkpoints')
    args = parser.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True),

    main(args)