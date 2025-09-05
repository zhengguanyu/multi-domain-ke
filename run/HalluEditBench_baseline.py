import os
import json
import sys
import argparse

sys.path.append('..')
from easyeditor import (
        GraceHyperParams,
        WISEHyperParams,
        ROMEHyperParams,
        FTHyperParams,
        MEMITHyperParams,
        AlphaEditHyperParams,
        EMMETHyperParams,
        summary_metrics,
        )

from easyeditor import BaseEditor
from easyeditor.models.ike import encode_ike_facts
from sentence_transformers import SentenceTransformer
from easyeditor import KnowEditDataset

from multiarea_dataset import MultiAreaDataset


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
        if not all_files:
            raise ValueError(" ALL:<total>, must all_files !")
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
            if ':' not in entry:
                raise ValueError(f"Fail: '{entry}'")
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

    parser.add_argument('--sequential_edit', default=True, type=str2bool)

    args = parser.parse_args()
    dataset_configs = parse_dataset_configs(args.data_configs)

    if args.editing_method == 'GRACE':
        editing_hparams = GraceHyperParams
    elif args.editing_method == 'WISE':
        editing_hparams = WISEHyperParams
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    elif args.editing_method == 'FT':
        editing_hparams = FTHyperParams
    elif args.editing_method == 'MEMIT':
        editing_hparams = MEMITHyperParams
    elif args.editing_method == 'AlphaEdit':
        editing_hparams = AlphaEditHyperParams
    elif args.editing_method == 'EMMET':
        editing_hparams = EMMETHyperParams,
    else:
        raise NotImplementedError
    
    multiarea_dataset = MultiAreaDataset(
        root_dir=args.data_dir,
        dataset_configs=dataset_configs,
        seed=42,
        random_sample=args.random_sample
    )
    
    prompts, rephrase_prompts, target_new, subjects, locality_inputs, _ = multiarea_dataset.to_edit_dataset()

    hparams = editing_hparams.from_hparams(args.hparams_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir,
        f'{hparams.model_name.split("/")[-1]}_{args.editing_method}_Sequential={args.sequential_edit}.json'
        )

    print("See results at: ", output_file)
    if args.editing_method == 'WISE':
        loc_filepath = '/data/zsre_mend_train.json'
        loc_data = json.load(
            open(loc_filepath, 'r', encoding='utf-8')
        )[:len(multiarea_dataset)]
        loc_prompts = [edit_data_['loc'] + ' ' + edit_data_['loc_ans'] for edit_data_ in loc_data]
        print('Len of loc_prompts: ', len(loc_prompts))
    else:
        loc_prompts = None


    editor = BaseEditor.from_hparams(hparams)
    if args.editing_method == 'WISE':
        metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            rephrase_prompts=rephrase_prompts,
            target_new=target_new,
            subject=subjects,
            locality_inputs=locality_inputs,
            sequential_edit=args.sequential_edit,
            loc_prompts=loc_prompts,
        )
    else:
        metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            rephrase_prompts=rephrase_prompts,
            target_new=target_new,
            subject=subjects,
            locality_inputs=locality_inputs,
            sequential_edit=args.sequential_edit,
        )

    if len(metrics) > 0:
        summary_metrics(metrics)
