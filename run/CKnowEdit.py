import os
import os.path
import sys
import json
import random

sys.path.append(os.getcwd() + '/EasyEdit')
sys.path.append('..')

from easyeditor import (
    FTHyperParams,
    MEMITHyperParams,
    ROMEHyperParams,
    GraceHyperParams,
    WISEHyperParams,
    ZZZHyperParams,
)
from easyeditor import BaseEditor
from easyeditor import CKnowEditDataset
from easyeditor.models.zzz.router import KnowRouter

import argparse

all_subset = [
    'classical_chinese_results_reviewed',
    'phonetic_notation_results_reviewed',
    'ruozhiba'
]

def load_cknowedit(filepath, ds_size):
    datas = CKnowEditDataset(filepath, ds_size)
    prompts = [data['prompt'] for data in datas]
    target_new = [data['target_new'] for data in datas]
    ground_truth = [data['target_old'] for data in datas]
    subject = [data['subject'] for data in datas]
    rephrase_prompts = [data['rephrase'] for data in datas]
    portability_data = [data['portability'] for data in datas]
    locality_data = [data['locality'] for data in datas]

    portability_prompts = []
    portability_answers = []
    for item in portability_data:
        if item is None or len(item) == 0:
            portability_prompts.append(None)
            portability_answers.append(None)
        else:
            temp_prompts = []
            temp_answers = []
            for pr in item:
                prompt = pr['prompt']
                an = pr['answer']
                temp_prompts.append(prompt)
                temp_answers.append(an)
            portability_prompts.append(temp_prompts)
            portability_answers.append(temp_answers)

    locality_prompts = []
    locality_answers = []
    for item in locality_data:
        if item is None or len(item) == 0:
            locality_prompts.append(None)
            locality_answers.append(None)
        else:
            temp_prompts = []
            temp_answers = []
            for pr in item:
                if 'prompt' in pr.keys():
                    prompt = pr["prompt"]
                    an = pr["answer"]
                    temp_prompts.append(prompt)
                    temp_answers.append(an)
            locality_prompts.append(temp_prompts)
            locality_answers.append(temp_answers)

    locality_inputs = {
        'loc_hop': {
            'prompt': locality_prompts,
            'ground_truth': locality_answers
        }
    }
    portability_inputs = {
        'por_hop': {
            'prompt': portability_prompts,
            'ground_truth': portability_answers
        }
    }

    return prompts, target_new, ground_truth, subject, rephrase_prompts, locality_inputs, portability_inputs

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--data_dir', required=True, type=str, help="Directory containing cknowedit subset json files")
    parser.add_argument('--ds_size', default=None, type=int, help="Total number of samples to load from cknowedit subsets")
    parser.add_argument('--metrics_save_dir', default='./output', type=str)
    parser.add_argument('--datatype', default='CKnowEdit', type=str, help="General datatype identifier for filename")
    parser.add_argument('--chinese_ds_type', default='all_subsets_sampled', type=str, help="Specific CKnowEdit subset type for filename")

    parser.add_argument('--random_sample', default=False, type=str2bool)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--router_save_path', default='./router', type=str)
    parser.add_argument('--router_load_path', default='./router', type=str)
    parser.add_argument('--retrain', default=False, type=str2bool)
    parser.add_argument('--sbert_path', default='sentence-transformers/all-MiniLM-L6-v2', type=str)
    parser.add_argument('--two_stages', default=True, type=str2bool)
    parser.add_argument('--boundary_model_name', default=None, type=str)
    parser.add_argument('--boundary_threshold', default=None, type=float)
    parser.add_argument('--use_clustering', default=True, type=str2bool)
    parser.add_argument('--use_multi_ffn', default=True, type=str2bool)

    args = parser.parse_args()

    combined_prompts = []
    combined_target_new = []
    combined_ground_truth_old = []
    combined_subject = []
    combined_rephrase_prompts = []

    combined_locality_inputs = {'loc_hop': {'prompt': [], 'ground_truth': []}}
    combined_portability_inputs = {'por_hop': {'prompt': [], 'ground_truth': []}}

    num_subsets = len(all_subset)

    shuffled_subsets = all_subset[:]

    base_samples_per_subset = args.ds_size // num_subsets
    remainder_samples = args.ds_size % num_subsets
    samples_per_subset_list = [base_samples_per_subset] * num_subsets
    for i in range(remainder_samples):
        samples_per_subset_list[i] += 1

    for i, subset_name in enumerate(shuffled_subsets):
        current_subset_size_to_load = samples_per_subset_list[i]

        subset_filepath = os.path.join(args.data_dir, subset_name + '.json')

        num_to_load_str = "all" if current_subset_size_to_load is None else str(current_subset_size_to_load)
        print(f"Loading data from {subset_name} (file: {subset_filepath}), aiming for {num_to_load_str} samples...")

        s_prompts, s_target_new, s_ground_truth_old, s_subject, s_rephrase_prompts, \
            s_locality_inputs, s_portability_inputs = load_cknowedit(subset_filepath, current_subset_size_to_load)

        combined_prompts.extend(s_prompts)
        combined_target_new.extend(s_target_new)
        combined_ground_truth_old.extend(s_ground_truth_old)
        combined_subject.extend(s_subject)
        combined_rephrase_prompts.extend(s_rephrase_prompts)

        if s_locality_inputs['loc_hop']['prompt']:
            combined_locality_inputs['loc_hop']['prompt'].extend(s_locality_inputs['loc_hop']['prompt'])
            combined_locality_inputs['loc_hop']['ground_truth'].extend(s_locality_inputs['loc_hop']['ground_truth'])


        if s_portability_inputs['por_hop']['prompt']:
            combined_portability_inputs['por_hop']['prompt'].extend(s_portability_inputs['por_hop']['prompt'])
            combined_portability_inputs['por_hop']['ground_truth'].extend(s_portability_inputs['por_hop']['ground_truth'])

        print(f"Loaded {len(s_prompts)} samples from {subset_name}. Total samples so far: {len(combined_prompts)}")

    prompts = combined_prompts
    target_new = combined_target_new
    subject = combined_subject
    rephrase_prompts = combined_rephrase_prompts
    locality_inputs = combined_locality_inputs
    portability_inputs = combined_portability_inputs


    print(f"\nTotal CKnowEdit samples loaded across all subsets: {len(prompts)}")

    if args.editing_method == 'FT':
        editing_hparams = FTHyperParams
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    elif args.editing_method == 'GRACE':
        editing_hparams = GraceHyperParams
    elif args.editing_method == 'WISE':
        editing_hparams = WISEHyperParams
    elif args.editing_method == 'MEMIT':
        editing_hparams = MEMITHyperParams
    elif args.editing_method == 'DSRE':
        editing_hparams = ZZZHyperParams

    else:
        raise NotImplementedError

    loc_prompts = None
    if args.editing_method == 'WISE':
        loc_filepath = './data/zsre_mend_train.json'

        loc_data = json.load(
            open(loc_filepath, 'r', encoding='utf-8')
        )[:int(len(prompts))]
        loc_prompts = [edit_data_['loc'] + ' ' + edit_data_['loc_ans'] for edit_data_ in loc_data]

    hparams = editing_hparams.from_hparams(args.hparams_dir)
    editor = BaseEditor.from_hparams(hparams)

    print(f"\nStarting editing with method: {args.editing_method}")

    print(f"Number of edits to perform: {len(prompts)}")

    if args.editing_method == 'DSRE':
        hparams.two_stages = args.two_stages
        hparams.boundary_model_name = args.boundary_model_name
        hparams.boundary_threshold = args.boundary_threshold
        hparams.use_clustering = args.use_clustering
        hparams.use_multi_ffn = args.use_multi_ffn

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

    if args.editing_method == 'WISE':
        metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            target_new=target_new,
            ground_truth=target_new,
            rephrase_prompts=rephrase_prompts,
            locality_inputs=locality_inputs,
            portability_inputs=portability_inputs,
            subject=subject,
            keep_original_weight=True,
            sequential_edit=True,
            loc_prompts=loc_prompts
        )
    elif args.editing_method == 'DSRE':
        metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            target_new=target_new,
            ground_truth=target_new,
            rephrase_prompts=rephrase_prompts,
            locality_inputs=locality_inputs,
            portability_inputs=portability_inputs,
            subject=subject,
            keep_original_weight=True,
            sequential_edit=True,
            router=router
        )

    else:
        metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            target_new=target_new,
            ground_truth=target_new,
            rephrase_prompts=rephrase_prompts,
            locality_inputs=locality_inputs,
            portability_inputs=portability_inputs,
            subject=subject,
            keep_original_weight=True,
            sequential_edit=True,
        )
