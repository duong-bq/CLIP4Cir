import multiprocessing
from argparse import ArgumentParser
from operator import itemgetter
from pathlib import Path
from statistics import mean
from typing import List, Tuple
import pandas as pd


import clip
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import squarepad_transform, FashionIQDataset, targetpad_transform, CIRRDataset
from combiner import Combiner
from utils import extract_index_features, extract_index_features_fclip, collate_fn, element_wise_sum, device
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoModel, AutoProcessor


def compute_fiq_val_metrics(relative_val_dataset: FashionIQDataset, clip_model: CLIPModel, clip_preprocess: CLIPProcessor, index_features: torch.tensor,
                            index_names: List[str], combining_function: callable) -> Tuple[float, float, float, float]:
    """
    Compute validation metrics on FashionIQ dataset
    :param relative_val_dataset: FashionIQ validation dataset in relative mode
    :param clip_model: CLIP model
    :param index_features: validation index features
    :param index_names: validation index names
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :return: the computed validation metrics
    """

    # Generate predictions
    predicted_features, target_names = generate_fiq_val_predictions(clip_model, clip_preprocess, relative_val_dataset,
                                                                    combining_function, index_names, index_features)

    print(f"Compute FashionIQ {relative_val_dataset.dress_types} validation metrics")

    # Normalize the index features
    index_features = F.normalize(index_features, dim=-1).float()

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at20 = (torch.sum(labels[:, :20]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100

    return round(recall_at5, 2), round(recall_at10, 2), round(recall_at20, 2), round(recall_at50, 2)


def generate_fiq_val_predictions(clip_model: CLIPModel, clip_preprocess: CLIPProcessor, relative_val_dataset: FashionIQDataset,
                                 combining_function: callable, index_names: List[str], index_features: torch.tensor) -> \
        Tuple[torch.tensor, List[str]]:
    """
    Compute FashionIQ predictions on the validation set
    :param clip_model: CLIP model
    :param relative_val_dataset: FashionIQ validation dataset in relative mode
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :param index_features: validation index features
    :param index_names: validation index names
    :return: predicted features and target names
    """
    print(f"Compute FashionIQ {relative_val_dataset.dress_types} validation predictions")

    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32,
                                     num_workers=multiprocessing.cpu_count(), pin_memory=True, collate_fn=collate_fn,
                                     shuffle=False)

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features))

    # Initialize predicted features and target names
    predicted_features = torch.empty((0, clip_model.projection_dim)).to(device, non_blocking=True)
    target_names = []

    for reference_names, batch_target_names, captions in tqdm(relative_val_loader):  # Load data
        input_captions = captions
        # text_inputs = clip.tokenize(input_captions, context_length=77).to(device, non_blocking=True)
        text_inputs = clip_preprocess(text=input_captions, return_tensors="pt", padding=True).to(device)

        # Compute the predicted features
        with torch.no_grad():
            # text_features = clip_model.encode_text(text_inputs)
            text_features = clip_model.get_text_features(**text_inputs)
            # Check whether a single element is in the batch due to the exception raised by torch.stack when used with
            # a single tensor
            if text_features.shape[0] == 1:
                reference_image_features = itemgetter(*reference_names)(name_to_feat).unsqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*reference_names)(
                    name_to_feat))  # To avoid unnecessary computation retrieve the reference image features directly from the index features
            batch_predicted_features = combining_function(reference_image_features, text_features)

        predicted_features = torch.vstack((predicted_features, F.normalize(batch_predicted_features, dim=-1)))
        target_names.extend(batch_target_names)

    return predicted_features, target_names


def fashioniq_val_retrieval(dress_type: str, combining_function: callable, clip_model: CLIPModel, preprocess: callable, clip_preprocess: CLIPProcessor) -> Tuple[float, float, float, float]:
    """
    Perform retrieval on FashionIQ validation set computing the metrics. To combine the features the `combining_function`
    is used
    :param dress_type: FashionIQ category on which perform the retrieval
    :param combining_function:function which takes as input (image_features, text_features) and outputs the combined
                            features
    :param clip_model: CLIP model
    :param preprocess: preprocess pipeline
    """
    print(type(combining_function))
    clip_model = clip_model.float().eval()

    # Define the validation datasets and extract the index features
    classic_val_dataset = FashionIQDataset('val', [dress_type], 'classic', preprocess)
    index_features, index_names = extract_index_features_fclip(classic_val_dataset, clip_model, device)
    relative_val_dataset = FashionIQDataset('val', [dress_type], 'relative', preprocess)

    return compute_fiq_val_metrics(relative_val_dataset, clip_model, clip_preprocess, index_features, index_names, combining_function)


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="should be either 'CIRR' or 'fashionIQ'")
    parser.add_argument("--combining-function", type=str, required=True,
                        help="Which combining function use, should be in ['combiner', 'sum']")
    parser.add_argument("--combiner-path", type=Path, help="path to trained Combiner")
    parser.add_argument("--projection-dim", default=640 * 4, type=int, help='Combiner projection dim')
    parser.add_argument("--hidden-dim", default=640 * 8, type=int, help="Combiner hidden dim")
    parser.add_argument("--clip-model-name", default="RN50x4", type=str, help="CLIP model to use, e.g 'RN50', 'RN50x4'")
    parser.add_argument("--clip-model-path", type=Path, help="Path to the fine-tuned CLIP model")
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str,
                        help="Preprocess pipeline, should be in ['clip', 'squarepad', 'targetpad'] ")
    parser.add_argument("--output-path", type=Path, help="Path to save the results")

    args = parser.parse_args()

    # clip_model, clip_preprocess = clip.load(args.clip_model_name, device=device, jit=False)
    # clip_model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").to(device)
    # clip_preprocess = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
    clip_model = AutoModel.from_pretrained('Marqo/marqo-fashionSigLIP', trust_remote_code=True)
    clip_preprocess = AutoProcessor.from_pretrained('Marqo/marqo-fashionSigLIP', trust_remote_code=True)
    print(clip_model.projection_dim)
    print(clip_model.text_embed_dim)
    print(clip_model.vision_embed_dim)
    input_dim = 224
    feature_dim = clip_model.projection_dim

    if args.clip_model_path:
        pass

    if args.transform == 'targetpad':
        print('Target pad preprocess pipeline is used')
        preprocess = targetpad_transform(args.target_ratio, input_dim)
    elif args.transform == 'squarepad':
        print('Square pad preprocess pipeline is used')
        preprocess = squarepad_transform(input_dim)
    else:
        print('CLIP default preprocess pipeline is used')
        preprocess = clip_preprocess

    if args.combining_function.lower() == 'sum':
        if args.combiner_path:
            print("Be careful, you are using the element-wise sum as combining_function but you have also passed a path"
                  " to a trained Combiner. Such Combiner will not be used")
        combining_function = element_wise_sum
    elif args.combining_function.lower() == 'combiner':
        combiner = Combiner(feature_dim, args.projection_dim, args.hidden_dim).to(device, non_blocking=True)
        state_dict = torch.load(args.combiner_path, map_location=device)
        combiner.load_state_dict(state_dict["Combiner"])
        combiner.eval()
        combining_function = combiner.combine_features
    else:
        raise ValueError("combiner_path should be in ['sum', 'combiner']")


    if args.dataset.lower() == 'fashioniq':
        average_recall5_list = []
        average_recall10_list = []
        average_recall20_list = []
        average_recall50_list = []

        shirt_recallat5, shirt_recallat10, shirt_recallat20, shirt_recallat50 = fashioniq_val_retrieval('shirt', combining_function, clip_model,
                                                                     preprocess, clip_preprocess)
        
        average_recall5_list.append(shirt_recallat5)
        average_recall10_list.append(shirt_recallat10)
        average_recall20_list.append(shirt_recallat20)
        average_recall50_list.append(shirt_recallat50)

        dress_recallat5, dress_recallat10, dress_recallat20, dress_recallat50 = fashioniq_val_retrieval('dress', combining_function, clip_model,
                                                                     preprocess, clip_preprocess)
        average_recall5_list.append(dress_recallat5)
        average_recall10_list.append(dress_recallat10)
        average_recall20_list.append(dress_recallat20)
        average_recall50_list.append(dress_recallat50)

        toptee_recallat5, toptee_recallat10, toptee_recallat20, toptee_recallat50 = fashioniq_val_retrieval('toptee', combining_function, clip_model,
                                                                       preprocess, clip_preprocess)
        average_recall5_list.append(toptee_recallat5)
        average_recall10_list.append(toptee_recallat10)
        average_recall20_list.append(toptee_recallat20)
        average_recall50_list.append(toptee_recallat50)

        # gemini_recallat5, gemini_recallat10, gemini_recallat20, gemini_recallat50 = fashioniq_val_retrieval('gemini', combining_function, clip_model,
        #                                                              preprocess)
        # average_recall5_list.append(gemini_recallat5)
        # average_recall10_list.append(gemini_recallat10)
        # average_recall20_list.append(gemini_recallat20)
        # average_recall50_list.append(gemini_recallat50)
        print()
        
        print(f"{dress_recallat5 = }")
        print(f"{dress_recallat10 = }")
        print(f"{dress_recallat20 = }")
        print(f"{dress_recallat50 = }")

        print(f"{shirt_recallat5 = }")
        print(f"{shirt_recallat10 = }")
        print(f"{shirt_recallat20 = }")
        print(f"{shirt_recallat50 = }")

        print(f"{toptee_recallat5 = }")
        print(f"{toptee_recallat10 = }")
        print(f"{toptee_recallat20 = }")
        print(f"{toptee_recallat50 = }")

        # print(f"{gemini_recallat5 = }")
        # print(f"{gemini_recallat10 = }")
        # print(f"{gemini_recallat20 = }")
        # print(f"{gemini_recallat50 = }")

        print(f"average recall10 = {mean(average_recall10_list)}")
        print(f"average recall50 = {mean(average_recall50_list)}")

        # Tạo DataFrame
        data = {
            'R@5': [dress_recallat5, shirt_recallat5, toptee_recallat5, round(mean(average_recall5_list), 2)],
            'R@10': [dress_recallat10, shirt_recallat10, toptee_recallat10, round(mean(average_recall10_list), 2)],
            'R@20': [dress_recallat20, shirt_recallat20, toptee_recallat20, round(mean(average_recall20_list), 2)],
            'R@50': [dress_recallat50, shirt_recallat50, toptee_recallat50, round(mean(average_recall50_list), 2)]
        }

        df = pd.DataFrame(data, index=['Dress', 'Shirt', 'Toptee', 'Average'])

        # Xuất DataFrame ra file CSV
        if args.output_path:
            df.to_csv(args.output_path)
        else:
            df.to_csv('recall_metrics.csv')
    else:
        raise ValueError("Dataset should be either 'CIRR' or 'FashionIQ")


if __name__ == '__main__':
    main()
