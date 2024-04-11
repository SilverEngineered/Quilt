import argparse
from models.ensemble import Ensemble
from models.pruned_ensemble import PrunedEnsemble
from models.multi_class_ensemble import MultiClassEnsemble
from models.crazy import Crazy
from models.crazy_combined import CrazyCombined
from models.crazy_combined_assisted import CrazyCombinedAssist
from models.final import Full
from models.final_divided import Full_D
parser = argparse.ArgumentParser()
parser.add_argument('--model', dest='model', default='Ensemble')
parser.add_argument('--dataset_name', dest='dataset_name', default='mushrooms', help='name of the dataset')
parser.add_argument('--num_wires', dest='num_wires', type=int, default=5)
parser.add_argument('--device_file', dest='device_file', default='dev_config0.json')
parser.add_argument('--training_epochs', dest='training_epochs', type=int, default=200)
parser.add_argument('--weights_file', dest='weights_file', default='var_rot_5_layer_100_epoch.npy')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=100)
parser.add_argument('--num_layers', dest='num_layers', type=int, default=5)
parser.add_argument('--run_type', dest='run_type', default='train')
parser.add_argument('--experts', dest='experts', type=int, default=3)
parser.add_argument('--load', dest='load', default=None)
parser.add_argument('--debug', dest='debug', type=int, default=0)
args = parser.parse_args()

if __name__ == '__main__':
    if args.model == 'Ensemble':
        network = Ensemble(args)
    elif args.model == 'Pruned':
        network = PrunedEnsemble(args)
    elif args.model == 'Multi_Class':
        network = MultiClassEnsemble(args)
    elif args.model == 'Crazy':
        network = Crazy(args)
    elif args.model == 'CrazyCombined':
        network = CrazyCombined(args)
    elif args.model == 'CrazyCombinedAssist':
        network = CrazyCombinedAssist(args)
    elif args.model == 'Final':
        network = Full(args)
    elif args.model == 'Final_Divide':
        network = Full_D(args)
    if args.run_type == 'test':
        network.run_inference()
    elif args.run_type == 'train':
        network.train()
    elif args.run_type == 'prune':
        network.prune()
