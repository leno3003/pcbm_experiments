import argparse
import os
import pickle
import numpy as np
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from torchvision.utils import save_image

import torchvision
from data import get_dataset
from concepts import ConceptBank
from models import PosthocLinearCBM, get_model
from training_tools import load_or_compute_projections, get_projections

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept-bank", required=True, type=str, help="Path to the concept bank")
    parser.add_argument("--model-checkpoint", required=False, type=str, help="Path to the concept bank")
    parser.add_argument("--out-dir", required=True, type=str, help="Output folder for model/run info.")
    parser.add_argument("--dataset", default="cub", type=str)
    parser.add_argument("--backbone-name", default="resnet18_cub", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--alpha", default=0.99, type=float, help="Sparsity coefficient for elastic net.")
    parser.add_argument("--lam", default=1e-5, type=float, help="Regularization strength.")
    parser.add_argument("--lr", default=1e-3, type=float)
    return parser.parse_args()

def main(args, concept_bank, backbone, preprocess):
    all_concepts = pickle.load(open(args.concept_bank, 'rb'))
    all_concept_names = list(all_concepts.keys())

    test_index = 50

    top_concept_number = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Employed device:', device)

    train_loader, test_loader, idx_to_class, classes = get_dataset(args, preprocess)
    
    # Get a clean conceptbank string
    # e.g. if the path is /../../cub_resnet-cub_0.1_100.pkl, then the conceptbank string is resnet-cub_0.1_100
    # which means a bank learned with 100 samples per concept with C=0.1 regularization parameter for the SVM. 
    # See `learn_concepts_dataset.py` for details.
    conceptbank_source = args.concept_bank.split("/")[-1].split(".")[0] 
    num_classes = len(classes)

    # Initialize the PCBM module.
    posthoc_layer = torch.load(args.model_checkpoint, map_location=device)
    posthoc_layer = posthoc_layer.to(args.device)
    posthoc_layer.eval()

    test_embs, test_projs, test_lbls = get_projections(args, backbone, posthoc_layer, test_loader)
    predictions, distributions = posthoc_layer(torch.from_numpy(test_embs).to(device), return_dist=True)
    print(predictions.shape)
    
    print(len(test_loader))
    for batch, (image, label) in enumerate(test_loader):
        correct_class = idx_to_class[label[test_index].item()]
        # save_image(torchvision.transforms.functional.adjust_brightness(image[0], 3), f'/home/cassano/pcbm_data/pcbm_test_image_{idx_to_class[label[0].item()]}.png')
        save_image(image[test_index], f'/home/cassano/pcbm_data/pcbm_test_image_{idx_to_class[label[test_index].item()]}.png')
        break

    print('First image prediction:', idx_to_class[torch.argmax(predictions[test_index]).item()])
    print(torch.argmax(predictions[test_index]).item())
    print('Correct class: ', correct_class)
    print(label[test_index].item())
    distributions = distributions.cpu().detach().numpy()

    index_max = np.argmax(np.array(distributions[test_index]))
    print(f'Top {top_concept_number} concepts: ')

    indices = np.argsort(distributions[test_index])[-top_concept_number:]
    for i in indices:
        print(all_concept_names[i] + ': ' + str(distributions[test_index][i]))


    correct = 0
    total = 0
    for batch_index, (image_batch, label_batch) in enumerate(test_loader):
        for i, label in enumerate(label_batch):
            if torch.argmax(predictions[i + batch_index*64]).item() == label.item():
                correct += 1
            total = total + 1
    print('Test accuracy: ')
    print(float(correct/total))
    print('End.')


if __name__ == "__main__":
    args = config()
    all_concepts = pickle.load(open(args.concept_bank, 'rb'))
    all_concept_names = list(all_concepts.keys())
    print(all_concept_names)
    print(f"Bank path: {args.concept_bank}. {len(all_concept_names)} concepts will be used.")
    concept_bank = ConceptBank(all_concepts, args.device)

    # Get the backbone from the model zoo.
    backbone, preprocess = get_model(args, backbone_name=args.backbone_name)
    backbone = backbone.to(args.device)
    backbone.eval()
    main(args, concept_bank, backbone, preprocess)