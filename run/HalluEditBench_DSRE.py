import os
import os.path as path
import json
import random
import sys
import argparse
from typing import List, Any, Union
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import yaml


sys.path.append(os.getcwd()+'/EasyEdit')
sys.path.append(os.getcwd()+'/EasyEdit/run_bishe')

from multiarea_dataset import MultiAreaDataset

try:
    from EasyEdit.easyeditor import (
        ZZZHyperParams
        )

    from EasyEdit.easyeditor import BaseEditor
    from EasyEdit.easyeditor.models.ike import encode_ike_facts
    from sentence_transformers import SentenceTransformer
    from EasyEdit.easyeditor import KnowEditDataset
    from EasyEdit.easyeditor.models.zzz.router import KnowRouter

except ImportError:
    from easyeditor import (
        ZZZHyperParams
        )

    from easyeditor import BaseEditor
    from easyeditor.models.ike import encode_ike_facts
    from sentence_transformers import SentenceTransformer
    from easyeditor import KnowEditDataset
    from easyeditor.models.zzz.router import KnowRouter

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_dataset_configs(config_str, all_files=None):
    if all_files is None:
        all_files = [
            "art_sculpture", "business_brand", "business_corporation",
            "business_industry", "entertainment_anime", "entertainment_music_genre",
            "entertainment_song", "event_film", "event_history",
            "event_sport", "geography_forest", "geography_glacier",
            "geography_volcano", "health_disease", "health_medication",
            "health_symptom", "human_athlete", "human_entrepreneur",
            "human_scientist", "human_writer", "places_city",
            "places_country", "places_landmark", "technology_database",
            "technology_programming_language", "technology_software"
        ]
    config_dict = {}
    if not config_str:
        return config_dict

    config_str = config_str.strip()

    if config_str.startswith("ALL:"):
        total = int(config_str.split(":")[1])
        num_files = len(all_files)
        base = total // num_files
        remainder = total % num_files

        for i, name in enumerate(all_files):
            k = base + (1 if i < remainder else 0)
            filename = name if name.endswith(".json") else name + ".json"
            config_dict[filename] = k
    else:
        for entry in config_str.split(','):
            filename, k = entry.split(':')
            if '.json' not in filename:
                filename = filename + '.json'
            k = int(k) if k != "None" else None
            config_dict[filename.strip()] = k

    return config_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--hparams_dir', required=True, type=str)

    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--data_configs', type=str, required=True)
    parser.add_argument('--random_sample', default=False, type=str2bool)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--metrics_save_dir', default='./output', type=str)
    parser.add_argument('--output_dir', default='./outputs', type=str)
    parser.add_argument('--router_save_path', default='./router', type=str)
    parser.add_argument('--router_load_path', default='./router', type=str)
    parser.add_argument('--retrain', default=False, type=str2bool)
    parser.add_argument('--sbert_path', default='sentence-transformers/all-MiniLM-L6-v2', type=str)
    parser.add_argument('--sequential_edit', default=True, type=str2bool)
    parser.add_argument('--two_stages', default=False, type=str2bool)
    parser.add_argument('--boundary_model_name', default=None, type=str)
    parser.add_argument('--boundary_threshold', default=None, type=float)
    parser.add_argument('--use_clustering', default=True, type=str2bool)
    parser.add_argument('--use_multi_ffn', default=True, type=str2bool)
    parser.add_argument('--edit_layer', default=12, type=int)

    args = parser.parse_args()

    if args.editing_method == 'DSRE':
        editing_hparams = ZZZHyperParams
    else:
        raise NotImplementedError

    dataset_configs = parse_dataset_configs(args.data_configs)

    multiarea_dataset = MultiAreaDataset(
        root_dir=args.data_dir,
        dataset_configs=dataset_configs,
        seed=222,
        random_sample=args.random_sample
    )

    prompts, rephrase_prompts, target_new, subjects, locality_inputs, _ = multiarea_dataset.to_edit_dataset()
    locality_prompts = locality_inputs['neighborhood']['prompt']
    hparams = editing_hparams.from_hparams(args.hparams_dir)
    hparams.two_stages = args.two_stages
    hparams.boundary_model_name = args.boundary_model_name
    hparams.boundary_threshold = args.boundary_threshold
    hparams.use_clustering = args.use_clustering
    hparams.use_multi_ffn = args.use_multi_ffn
    hparams.inner_params[0] = f'model.layers[{args.edit_layer}].mlp.down_proj.weight'

    if args.sbert_path != 'sentence-transformers/all-MiniLM-L6-v2':
        hparams.sbert_path = args.sbert_path
        hparams.embedding.model_name = args.sbert_path
    
    if args.router_load_path and not args.retrain:
        try:
            router = KnowRouter.load(args.router_load_path)
        except Exception as e:
            router = KnowRouter(cfg=hparams)
            router.build_route_table(prompt_list=prompts)
    else:
        router = KnowRouter(cfg=hparams)
        print(hparams.clustering)
        router.build_route_table(prompt_list=prompts)
        if args.router_save_path:
            try:
                router.save(args.router_save_path)
            except Exception as e:
                print(f"Router Fail: {str(e)}")
    correct_locality_routing = 0
    total_locality = len(prompts)
    for i in range(total_locality):
        original_prompt = prompts[i]
        locality_prompt = locality_prompts[i]
        original_cluster_id = router.route_table.get(original_prompt, -99)
        if original_cluster_id == -99:
            continue

        predicted_locality_cluster_id, locality_confidence = router.route_with_confidence(locality_prompt)

        is_correct = (predicted_locality_cluster_id != original_cluster_id)

        if is_correct:
            correct_locality_routing += 1

        print(f"Original Prompt (Idx {i}): '{original_prompt}' -> Target Cluster: {original_cluster_id}")
        print(f"Locality Prompt (Idx {i}): '{locality_prompt}' -> Routed Cluster: {predicted_locality_cluster_id}, Confidence: {locality_confidence:.4f}")
        print(f"  -> Locality Routing Correct? {'Yes' if is_correct else 'No'}")
        print("-" * 20)

    locality_accuracy = correct_locality_routing / total_locality if total_locality > 0 else 0

    correct_rephrase_routing = 0
    total_rephrase = len(prompts)
    for i in range(total_rephrase):
        original_prompt = prompts[i]
        rephrase_prompt = rephrase_prompts[i]

        original_cluster_id = router.route_table.get(original_prompt, -99)
        if original_cluster_id == -99:
            continue

        predicted_rephrase_cluster_id, rephrase_confidence = router.route_with_confidence(rephrase_prompt)

        is_correct = (predicted_rephrase_cluster_id == original_cluster_id)

        if is_correct:
            correct_rephrase_routing += 1

        print(f"Original Prompt (Idx {i}): '{original_prompt}' -> Target Cluster: {original_cluster_id}")
        print(f"Rephrase Prompt (Idx {i}): '{rephrase_prompt}' -> Routed Cluster: {predicted_rephrase_cluster_id}, Confidence: {rephrase_confidence:.4f}")
        print(f"  -> Rephrase Routing Correct? {'Yes' if is_correct else 'No'}")
        print("-" * 20)

    rephrase_accuracy = correct_rephrase_routing / total_rephrase if total_rephrase > 0 else 0

    print(f"\nLocality Routing Accuracy (Routed to different cluster): {correct_locality_routing}/{total_locality} = {locality_accuracy:.4f}")
    print(f"\nRephrase Routing Accuracy (Routed to same cluster): {correct_rephrase_routing}/{total_rephrase} = {rephrase_accuracy:.4f}")

    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        rephrase_prompts=rephrase_prompts,
        target_new=target_new,
        subject=subjects,
        locality_inputs=locality_inputs,
        sequential_edit=args.sequential_edit,
        router=router
    )

    print('Method: {}'.format(args.editing_method))
    print('Model: {}'.format(hparams.model_name.split("/")[-1]))
    print('Layer: {}'.format(hparams.inner_params[0]))
    print('Sequential: {}'.format(args.sequential_edit))
