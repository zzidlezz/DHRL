import argparse
from noisifier import *
from DHRL_model import DHRL
from utils import *
from data import *
import torch



def train(args, dset):
    print('Training Stage...')
    print('Train size: %d' % (dset.I_tr.shape[0]))
    assert dset.I_tr.shape[0] == dset.T_tr.shape[0]
    assert dset.I_tr.shape[0] == dset.L_tr.shape[0]

    dhrl = DHRL(args=args)
    dhrl.train().cuda()
    optimizer = torch.optim.Adam([{'params': dhrl.parameters(), 'lr': args.lr}])
    add_noise = Multi_Label_Noisifier()
    train_label = dset.L_tr
    train_label,_ =  add_noise.random_noise_per_sample( train_label, 0.8, 0.8)

    train_loader = data.DataLoader(my_dataset(dset.I_tr, dset.T_tr, train_label),
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   pin_memory=True)

    for epoch in range(args.epochs):
        for i, (idx, img_feat, txt_feat, label) in enumerate(train_loader):

            img_feat = img_feat.cuda()
            txt_feat = txt_feat.cuda()

            label = torch.Tensor( label).cuda()

            optimizer.zero_grad()
            H_I , H_T, pred_I , pred_T = dhrl(img_feat, txt_feat)



            error_loss_array_1, classes_1 = Ranking(label, pred_I, 0.2, 0.8)
            error_loss_array_2, classes_2 = Ranking(label, pred_T ,0.2, 0.8)


            lossLR = LRLoss(torch.Tensor([1.0]))
            lossLR.cuda()

            loss_array_1 = lossLR(pred_I, label)

            loss_array_2 = lossLR(pred_T ,label)


            brutto_loss_array_1 = error_loss_array_1
            brutto_loss_array_2 = error_loss_array_2

            low_loss_args_1 = torch.argsort(brutto_loss_array_1)[:int(128*0.75)]
            low_loss_args_2 = torch.argsort(brutto_loss_array_2)[:int(128*0.75)]

            low_loss_samples_1 = loss_array_1[low_loss_args_2]
            low_loss_samples_2 = loss_array_2[low_loss_args_1]

            loss_1 = torch.mean(low_loss_samples_1)
            loss_2 = torch.mean(low_loss_samples_2)



            loss_C = torch.mean((torch.square(H_I-H_T)))



            loss =loss_1+loss_2+loss_C

            loss.backward()

            optimizer.step()

            if (i + 1) == len(train_loader) and (epoch + 1) % 2 == 0:

                print('Epoch [%3d/%3d], Loss: %.4f, Loss1: %.4f,Loss2: %.4f,Loss_C:%.4f '
                      % (epoch + 1, args.epochs, loss.item(),

                         loss_1.item(),loss_2.item(),loss_C.item()
                       ))


    return dhrl

def eval(model, dset, args):
    model.eval()
    print('=' * 30)
    print('Testing Stage...')
    print('Retrieval size: %d' % (dset.I_db.shape[0]))
    assert dset.I_db.shape[0] == dset.T_db.shape[0]
    assert dset.I_db.shape[0] == dset.L_db.shape[0]

    retrieval_loader = data.DataLoader(my_dataset(dset.I_db, dset.T_db, dset.L_db),
                                       batch_size=args.eval_batch_size,
                                       shuffle=False,
                                       num_workers=0,
                                       pin_memory=True)

    retrievalI = []
    retrievalT = []

    for i, (idx, img_feat, txt_feat, _) in enumerate(retrieval_loader):
        img_feat = img_feat.cuda()
        txt_feat = txt_feat.cuda()
        H_I,H_T,_,_  = model(img_feat, txt_feat)
        retrievalI.append(H_I.data.cpu().numpy())
        retrievalT.append(H_T.data.cpu().numpy())



    retrievalH_I = np.concatenate(retrievalI)
    retrievalH_T = np.concatenate(retrievalT)


    retrievalCodeI = np.sign(retrievalH_I)
    retrievalCodeT= np.sign(retrievalH_T)



    print('Query size: %d' % (dset.I_te.shape[0]))
    assert dset.I_te.shape[0] == dset.T_te.shape[0]
    assert dset.I_te.shape[0] == dset.L_te.shape[0]

    val_loader = data.DataLoader(my_dataset(dset.I_te, dset.T_te, dset.L_te),
                                 batch_size=args.eval_batch_size,
                                 shuffle=False,
                                 num_workers=0,
                                 pin_memory=True)

    valH_I = []
    valH_T = []

    for i, (idx, img_feat, txt_feat, _) in enumerate(val_loader):
        img_feat = img_feat.cuda()
        txt_feat = txt_feat.cuda()
        H_I, H_T, _,_ = model(img_feat, txt_feat)
        valH_I.append(H_I.data.cpu().numpy())
        valH_T.append(H_T.data.cpu().numpy())

    valI = np.concatenate(valH_I)
    valT = np.concatenate(valH_T)

    valCodeI = np.sign(valI)
    valCodeT = np.sign(valT)



    MAP_I2T_1000 = calculate_top_map(qu_B= valCodeI, re_B=  retrievalCodeT, qu_L=dset.L_te, re_L=dset.L_db, topk=1000)
    MAP_T2I_1000 = calculate_top_map(qu_B= valCodeT, re_B=retrievalCodeI, qu_L=dset.L_te, re_L=dset.L_db, topk=1000)
    print('map(i->t): %3.4f' % (MAP_I2T_1000))
    print('map(t->i): %3.4f' % (MAP_T2I_1000))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## Net basic params
    parser.add_argument('--model', type=str, default='DHRL')
    parser.add_argument('--epochs', type=int, default=20, help='Number of student epochs to train.')
    parser.add_argument('--nbit', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.5, help='')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--eval_batch_size', type=int, default=128)

    ## Transformer params
    parser.add_argument('--nhead', type=int, default=1, help='"nhead" in Transformer.')
    parser.add_argument('--num_layer', type=int, default=2, help='"num_layer" in Transformer.')
    parser.add_argument('--trans_act', type=str, default='gelu', help='"activation" in Transformer.')

    ## Data params       MIR 80      WIKI 60    nus100
    parser.add_argument('--dataset', type=str, default='flickr', help='coco/nuswide/flickr')
    parser.add_argument('--classes', type=int, default=24)
    parser.add_argument('--image_dim', type=int, default=4096)
    parser.add_argument('--text_dim', type=int, default=1000)


    parser.add_argument('--img_hidden_dim', type=list, default=[2048, 128], help='Construct imageMLP')
    parser.add_argument('--txt_hidden_dim', type=list, default=[1024, 128], help='Construct textMLP')

    parser.add_argument('--seed', type=int, default=2021)
    args = parser.parse_args()

    seed_setting(args.seed)

    dset = load_data(args.dataset)
    print('Train size: %d, Retrieval size: %d, Query size: %d' % (dset.I_tr.shape[0], dset.I_db.shape[0], dset.I_te.shape[0]))
    print('Image dimension: %d, Text dimension: %d, Label dimension: %d' % (dset.I_tr.shape[1], dset.T_tr.shape[1], dset.L_tr.shape[1]))

    args.image_dim = dset.I_tr.shape[1]
    args.text_dim = dset.T_tr.shape[1]
    args.classes = dset.L_tr.shape[1]

    args.img_hidden_dim.insert(0, args.image_dim)
    args.txt_hidden_dim.insert(0, args.text_dim)



    model = train(args, dset)
            # eval the Model
    eval(model, dset, args)


