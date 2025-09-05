import argparse
import os.path
import sys
import json
import datetime

sys.path.append('..')

from easyeditor import (
    WISEHyperParams,
    FTHyperParams,
    ROMEHyperParams,
    MEMITHyperParams,
    EMMETHyperParams,
    summary_metrics,
    GraceHyperParams,
    ZZZHyperParams
)

from easyeditor import BaseEditor
from easyeditor import BaseEditor
from easyeditor.models.ike import encode_ike_facts
from sentence_transformers import SentenceTransformer
from easyeditor import KnowEditDataset
from easyeditor.models.zzz.router import KnowRouter

from multiarea_dataset import MultiAreaDataset


def preprocess_coutnerfact(edit_filepath, loc_filepath, N):

    datas = KnowEditDataset(edit_filepath, size=N)
    loc_data = json.load(
        open(loc_filepath, 'r', encoding='utf-8')
    )[:N]
    loc_prompts = [edit_data_['loc'] + ' ' + edit_data_['loc_ans'] for edit_data_ in loc_data]

    prompts = [data['prompt'] for data in datas]
    subjects = [data['subject'] for data in datas]
    target_new = [data['target_new'] for data in datas]

    portability_r = [data['portability_r'] for data in datas]
    portability_s = [data['portability_s'] for data in datas]
    portability_l = [data['portability_l'] for data in datas]

    portability_reasoning_prompts = []
    portability_reasoning_ans = []
    portability_Logical_Generalization_prompts = []
    portability_Logical_Generalization_ans = []
    portability_Subject_Aliasing_prompts = []
    portability_Subject_Aliasing_ans = []

    portability_data = [portability_r, portability_s, portability_l]
    portability_prompts = [portability_reasoning_prompts, portability_Subject_Aliasing_prompts, portability_Logical_Generalization_prompts]
    portability_answers = [portability_reasoning_ans, portability_Subject_Aliasing_ans, portability_Logical_Generalization_ans]
    for data, portable_prompts, portable_answers in zip(portability_data, portability_prompts, portability_answers):
        for item in data:
            if item is None:
                portable_prompts.append(None)
                portable_answers.append(None)
            else:
                temp_prompts = []
                temp_answers = []
                for pr in item:
                    prompt = pr["prompt"]
                    an = pr["ground_truth"]
                    while isinstance(an, list):
                        an = an[0]
                    if an.strip() == "":
                        continue
                    temp_prompts.append(prompt)
                    temp_answers.append(an)
                portable_prompts.append(temp_prompts)
                portable_answers.append(temp_answers)
    assert len(prompts) == len(portability_reasoning_prompts) == len(portability_Logical_Generalization_prompts) == len(portability_Subject_Aliasing_prompts)

    locality_rs = [data['locality_rs'] for data in datas]
    locality_f = [data['locality_f'] for data in datas]
    locality_Relation_Specificity_prompts = []
    locality_Relation_Specificity_ans = []
    locality_Forgetfulness_prompts = []
    locality_Forgetfulness_ans = []

    locality_data = [locality_rs, locality_f]
    locality_prompts = [locality_Relation_Specificity_prompts, locality_Forgetfulness_prompts]
    locality_answers = [locality_Relation_Specificity_ans, locality_Forgetfulness_ans]
    for data, local_prompts, local_answers in zip(locality_data, locality_prompts, locality_answers):
        for item in data:
            if item is None:
                local_prompts.append(None)
                local_answers.append(None)
            else:
                temp_prompts = []
                temp_answers = []
                for pr in item:
                    prompt = pr["prompt"]
                    an = pr["ground_truth"]
                    while isinstance(an, list):
                        an = an[0]
                    if an.strip() == "":
                        continue
                    temp_prompts.append(prompt)
                    temp_answers.append(an)
                local_prompts.append(temp_prompts)
                local_answers.append(temp_answers)
    assert len(prompts) == len(locality_Relation_Specificity_prompts) == len(locality_Forgetfulness_prompts)

    locality_inputs = {
        'Relation_Specificity': {
            'prompt': locality_Relation_Specificity_prompts,
            'ground_truth': locality_Relation_Specificity_ans
        },
        'Forgetfulness': {
            'prompt': locality_Forgetfulness_prompts,
            'ground_truth': locality_Forgetfulness_ans
        }
    }
    portability_inputs = {
        'Subject_Aliasing': {
            'prompt': portability_Subject_Aliasing_prompts,
            'ground_truth': portability_Subject_Aliasing_ans
        },
        'reasoning': {
            'prompt': portability_reasoning_prompts,
            'ground_truth': portability_reasoning_ans
        },
        'Logical_Generalization': {
            'prompt': portability_Logical_Generalization_prompts,
            'ground_truth': portability_Logical_Generalization_ans
        }
    }

    return prompts, subjects, portability_inputs, target_new, locality_inputs, loc_prompts

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    hyperparams_maps = {
        'WISE': WISEHyperParams,
        'FT': FTHyperParams,
        'ROME': ROMEHyperParams,
        'MEMIT': MEMITHyperParams,
        'EMMET': EMMETHyperParams,
        'GRACE': GraceHyperParams,
        'DSRE': ZZZHyperParams
    }

    data_processor_maps = {
        'counterfact': preprocess_coutnerfact
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--sequential_edit', default=True, type=str2bool)
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--ds_size', default=None, type=int)
    parser.add_argument('--data_type', default=None, type=str)
    parser.add_argument('--evaluation_type', type=str)
    parser.add_argument('--api_key', default=None, type=str)
    parser.add_argument('--output_dir', default='./outputs', type=str)
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

    repeat = False
    if args.ds_size == 1:
        repeat = True

    start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    hparams = hyperparams_maps[args.editing_method].from_hparams(args.hparams_dir)

    if args.data_type == 'counterfact':

        prompts, subject,portability_inputs, target_new, locality_inputs, loc_prompts = data_processor_maps[args.data_type](
            edit_filepath=args.data_dir,
            loc_filepath='/data/zsre_mend_train.json',
            N=args.ds_size
        )
    else:
        raise NotImplementedError

    if repeat:
        prompts = prompts * 10
        subject = subject * 10
        target_new = target_new * 10
        for key in locality_inputs.keys():
            locality_inputs[key]['prompt'] = locality_inputs[key]['prompt'] * 10
            locality_inputs[key]['ground_truth'] = locality_inputs[key]['ground_truth'] * 10
        for key in portability_inputs.keys():
            portability_inputs[key]['prompt'] = portability_inputs[key]['prompt'] * 10
            portability_inputs[key]['ground_truth'] = portability_inputs[key]['ground_truth'] * 10
        
        args.ds_size = 10 * args.ds_size

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
    
    editor = BaseEditor.from_hparams(hparams)
    if args.editing_method == 'WISE':
        metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            target_new=target_new,
            subject=subject,
            locality_inputs=locality_inputs,
            portability_inputs=portability_inputs,
            keep_original_weight=True,
            sequential_edit=args.sequential_edit,
            loc_prompts=loc_prompts,
        )
    elif args.editing_method == 'DSRE':
        metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            target_new=target_new,
            subject=subject,
            locality_inputs=locality_inputs,
            portability_inputs=portability_inputs,
            keep_original_weight=True,
            sequential_edit=args.sequential_edit,
            router=router,
        )
    else:
        metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            target_new=target_new,
            subject=subject,
            locality_inputs=locality_inputs,
            portability_inputs=portability_inputs,
            keep_original_weight=True,
            sequential_edit=args.sequential_edit,
        )

    end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('Method: {}'.format(args.editing_method))
    print('Data: {}'.format(args.data_type))
    print('Size: {}'.format(args.ds_size))
    print('Model: {}'.format(hparams.model_name.split("/")[-1]))
    print('Evaluation: {}'.format(args.evaluation_type))
    print('Sequential: {}'.format(args.sequential_edit))
    print('from {} to {}'.format(start_time, end_time))
