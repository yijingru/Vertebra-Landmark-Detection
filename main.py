import argparse
import train
import test
import eval

def parse_args():
    parser = argparse.ArgumentParser(description='CenterNet Modification Implementation')
    parser.add_argument('--num_epoch', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Number of epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--init_lr', type=float, default=1.25e-4, help='Init learning rate')
    parser.add_argument('--down_ratio', type=int, default=4, help='down ratio')
    parser.add_argument('--input_h', type=int, default=1024, help='input height')
    parser.add_argument('--input_w', type=int, default=512, help='input width')
    parser.add_argument('--K', type=int, default=100, help='maximum of objects')
    parser.add_argument('--conf_thresh', type=float, default=0.2, help='confidence threshold')
    parser.add_argument('--seg_thresh', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--num_classes', type=int, default=1, help='number of classes')
    parser.add_argument('--ngpus', type=int, default=0, help='number of gpus')
    parser.add_argument('--resume', type=str, default='model_last.pth', help='weights to be resumed')
    parser.add_argument('--data_dir', type=str, default='../../Datasets/spinal/', help='data directory')
    parser.add_argument('--phase', type=str, default='test', help='data directory')
    parser.add_argument('--dataset', type=str, default='spinal', help='data directory')
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    if args.phase == 'train':
        is_object = train.Network(args)
        is_object.train_network(args)
    elif args.phase == 'test':
        is_object = test.Network(args)
        is_object.test(args, save=False)
    elif args.phase == 'eval':
        is_object = eval.Network(args)
        is_object.eval(args, save=False)
        # is_object.eval_three_angles(args, save=False)