import os
import sys
import json
import random
import argparse
import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'EasyEdit'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.dirname(__file__))

from easyeditor import (
    FTHyperParams, ROMEHyperParams, MEMITHyperParams,
    GraceHyperParams, WISEHyperParams, ASMemHyperParams,
    AlphaEditHyperParams, EMMETHyperParams,
)
from easyeditor.models.ndedit import NDEditHyperParams
from easyeditor.models.deltaedit import DeltaEditHyperParams
from easyeditor.models.eamet import EAMETHyperParams
from easyeditor.models.blue import BLUEHyperParams
from easyeditor import BaseEditor, CKnowEditDataset, KnowEditDataset
from easyeditor.models.asmem.router import KnowRouter
from multiarea_dataset import MultiAreaDataset

HPARAMS_MAP = {
    'FT':        FTHyperParams,
    'ROME':      ROMEHyperParams,
    'MEMIT':     MEMITHyperParams,
    'GRACE':     GraceHyperParams,
    'WISE':      WISEHyperParams,
    'ASMem':       ASMemHyperParams,           
    'AlphaEdit': AlphaEditHyperParams,
    'NDEdit':    NDEditHyperParams,
    'DeltaEdit': DeltaEditHyperParams,
    'BLUE':      BLUEHyperParams,
    'RECT':      BLUEHyperParams,
}

def load_hallu(args):

    dataset_configs = parse_hallu_configs(args.data_configs)
    dataset = MultiAreaDataset(
        root_dir=args.data_dir,
        dataset_configs=dataset_configs,
        seed=args.seed,
        random_sample=args.random_sample,
    )
    prompts, rephrase_prompts, target_new, subjects, locality_inputs, source_files =\
        dataset.to_edit_dataset()

    data = dict(
        prompts=prompts,
        target_new=target_new,
        subject=subjects,
        rephrase_prompts=rephrase_prompts,
        locality_inputs=locality_inputs,
        portability_inputs=None,
    )

                                                                         
                                                                                
    if args.interleave:
        data = interleave_by_domain(data, source_files)
    return data


def parse_hallu_configs(config_str):

    ALL_FILES = [
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
    config_str = config_str.strip()

    if config_str.startswith("ALL:"):
        total = int(config_str.split(":")[1])
        base = total // len(ALL_FILES)
        remainder = total % len(ALL_FILES)
        for i, name in enumerate(ALL_FILES):
            k = base + (1 if i < remainder else 0)
            config_dict[name + ".json"] = k
    else:
        for entry in config_str.split(','):
            entry = entry.strip()
            if ':' not in entry:
                raise ValueError(f"Invalid config: '{entry}', expected 'name:count'")
            name, k = entry.split(':')
            name = name.strip()
            if not name.endswith('.json'):
                name += '.json'
            config_dict[name] = int(k)

    return config_dict


def load_cknowedit(args):

    ALL_SUBSETS = [
        'ancient_poetry_reviewed',
        'classical_chinese_results_reviewed',
        'geography_results_reviewed',
        'history',
        'phonetic_notation_results_reviewed',
        'proverb_results_reviewed',
    ]

    ds_size = args.ds_size
    if ds_size is None:
        raise ValueError("--ds_size is required for CKnowEdit dataset")

    num_subsets = len(ALL_SUBSETS)
    base = ds_size // num_subsets
    remainder = ds_size % num_subsets
    samples_per_subset = [base + (1 if i < remainder else 0) for i in range(num_subsets)]

    combined = dict(
        prompts=[], target_new=[], subject=[], rephrase_prompts=[],
        locality_inputs={'loc_hop': {'prompt': [], 'ground_truth': []}},
        portability_inputs={'por_hop': {'prompt': [], 'ground_truth': []}},
    )
    source_files = []                                        

    for i, subset_name in enumerate(ALL_SUBSETS):
        filepath = os.path.join(args.data_dir, subset_name + '.json')
        n = samples_per_subset[i]
        print(f"Loading {subset_name}: {n} samples from {filepath}")

        datas = CKnowEditDataset(filepath, n)

        n_loaded = len(datas)
        combined['prompts'].extend([d['prompt'] for d in datas])
        combined['target_new'].extend([d['target_new'] for d in datas])
        combined['subject'].extend([d['subject'] for d in datas])
        combined['rephrase_prompts'].extend([d['rephrase'] for d in datas])
        source_files.extend([subset_name] * n_loaded)

                     
        for d in datas:
            port = d['portability']
            if port is None or len(port) == 0:
                combined['portability_inputs']['por_hop']['prompt'].append(None)
                combined['portability_inputs']['por_hop']['ground_truth'].append(None)
            else:
                combined['portability_inputs']['por_hop']['prompt'].append(
                    [p['prompt'] for p in port])
                combined['portability_inputs']['por_hop']['ground_truth'].append(
                    [p['answer'] for p in port])

                                                            
        for d in datas:
            loc = d['locality']
            if loc is None or len(loc) == 0:
                combined['locality_inputs']['loc_hop']['prompt'].append(None)
                combined['locality_inputs']['loc_hop']['ground_truth'].append(None)
            else:
                loc_with_prompt = [p for p in loc if 'prompt' in p]
                if loc_with_prompt:
                    combined['locality_inputs']['loc_hop']['prompt'].append(
                        [p['prompt'] for p in loc_with_prompt])
                    combined['locality_inputs']['loc_hop']['ground_truth'].append(
                        [p['answer'] for p in loc_with_prompt])
                else:
                    combined['locality_inputs']['loc_hop']['prompt'].append(None)
                    combined['locality_inputs']['loc_hop']['ground_truth'].append(None)

                                                                                 
    combined['ground_truth'] = combined['target_new']

    print(f"Total CKnowEdit samples: {len(combined['prompts'])}")

    if args.interleave:
        combined = interleave_by_domain(combined, source_files)
    return combined


def load_counterfact(args):

    ds_size = args.ds_size
    datas = KnowEditDataset(args.data_dir, size=ds_size)

    prompts = [d['prompt'] for d in datas]
    subjects = [d['subject'] for d in datas]
    target_new = [d['target_new'] for d in datas]

                         
    def _extract_nested(data_list):
        extracted_prompts, extracted_answers = [], []
        for item in data_list:
            if item is None:
                extracted_prompts.append(None)
                extracted_answers.append(None)
            else:
                ps, ans = [], []
                for pr in item:
                    a = pr['ground_truth']
                    while isinstance(a, list):
                        a = a[0]
                    if a.strip() == '':
                        continue
                    ps.append(pr['prompt'])
                    ans.append(a)
                extracted_prompts.append(ps)
                extracted_answers.append(ans)
        return extracted_prompts, extracted_answers

    port_r_p, port_r_a = _extract_nested([d['portability_r'] for d in datas])
    port_s_p, port_s_a = _extract_nested([d['portability_s'] for d in datas])
    port_l_p, port_l_a = _extract_nested([d['portability_l'] for d in datas])

    portability_inputs = {
        'reasoning':              {'prompt': port_r_p, 'ground_truth': port_r_a},
        'Subject_Aliasing':       {'prompt': port_s_p, 'ground_truth': port_s_a},
        'Logical_Generalization': {'prompt': port_l_p, 'ground_truth': port_l_a},
    }

                      
    loc_rs_p, loc_rs_a = _extract_nested([d['locality_rs'] for d in datas])
    loc_f_p, loc_f_a = _extract_nested([d['locality_f'] for d in datas])

    locality_inputs = {
        'Relation_Specificity': {'prompt': loc_rs_p, 'ground_truth': loc_rs_a},
        'Forgetfulness':        {'prompt': loc_f_p,  'ground_truth': loc_f_a},
    }

    return dict(
        prompts=prompts,
        target_new=target_new,
        subject=subjects,
        rephrase_prompts=None,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
    )


def load_zsre(args):

    ds_size = args.ds_size
    edit_data = json.load(open(args.data_dir, 'r', encoding='utf-8'))[:ds_size]

    prompts = [d['src'] for d in edit_data]
    subjects = [d['subject'] for d in edit_data]
    rephrase_prompts = [d['rephrase'] for d in edit_data]
    target_new = [d['alt'] for d in edit_data]
    locality_prompts = [d['loc'] for d in edit_data]
    locality_ans = [d['loc_ans'] for d in edit_data]

    locality_inputs = {
        'neighborhood': {
            'prompt': locality_prompts,
            'ground_truth': locality_ans,
        },
    }

    return dict(
        prompts=prompts,
        target_new=target_new,
        subject=subjects,
        rephrase_prompts=rephrase_prompts,
        locality_inputs=locality_inputs,
        portability_inputs=None,
    )


DATASET_LOADERS = {
    'hallu':       load_hallu,
    'cknowedit':   load_cknowedit,
}

def interleave_by_domain(data, source_files):

    if source_files is None or len(source_files) == 0:
        return data

                                                             
    from collections import OrderedDict
    domain_indices = OrderedDict()
    for idx, domain in enumerate(source_files):
        domain_indices.setdefault(domain, []).append(idx)

                            
    domains = list(domain_indices.keys())
    K = len(domains)
    iterators = [iter(domain_indices[d]) for d in domains]
    interleaved_order = []
    exhausted = set()
    while len(exhausted) < K:
        for i, it in enumerate(iterators):
            if i in exhausted:
                continue
            try:
                interleaved_order.append(next(it))
            except StopIteration:
                exhausted.add(i)

                        
    def _reorder_list(lst, order):
        if lst is None:
            return None
        return [lst[i] for i in order]

    def _reorder_dict_of_lists(d, order):
        if d is None:
            return None
        result = {}
        for key, val in d.items():
            if isinstance(val, dict):
                result[key] = _reorder_dict_of_lists(val, order)
            elif isinstance(val, list):
                result[key] = _reorder_list(val, order)
            else:
                result[key] = val
        return result

    reordered = {}
    for key, val in data.items():
        if isinstance(val, list) and len(val) == len(source_files):
            reordered[key] = _reorder_list(val, interleaved_order)
        elif isinstance(val, dict):
            reordered[key] = _reorder_dict_of_lists(val, interleaved_order)
        else:
            reordered[key] = val

    return reordered

def load_wise_loc_prompts(n, loc_path=None):

    if loc_path is None:
        loc_path = os.path.join(
            os.path.dirname(__file__), '..', 'EasyEdit', 'data', 'wise',
            'ZsRE', 'zsre_mend_train.json')
        if not os.path.exists(loc_path):
            loc_path = os.path.join(
                os.path.dirname(__file__), '..', 'data', 'wise',
                'ZsRE', 'zsre_mend_train.json')
    if not os.path.exists(loc_path):
        raise FileNotFoundError(
            f"WISE locality data not found at: {loc_path}\n"
            f"Use --loc_data_path to specify the correct path to zsre_mend_train.json")
    loc_data = json.load(open(loc_path, 'r', encoding='utf-8'))[:n]
    return [d['loc'] + ' ' + d['loc_ans'] for d in loc_data]

def build_or_load_router(hparams, prompts, args):

                                          
    if args.two_stages is not None:
        hparams.two_stages = args.two_stages
    if args.boundary_model_name is not None:
        hparams.boundary_model_name = args.boundary_model_name
    if args.boundary_threshold is not None:
        hparams.boundary_threshold = args.boundary_threshold
    if args.use_clustering is not None:
        hparams.use_clustering = args.use_clustering
    if args.use_multi_ffn is not None:
        hparams.use_multi_ffn = args.use_multi_ffn
    if args.random_routing is not None:
        hparams.random_routing = args.random_routing
    if args.sbert_path:
        hparams.sbert_path = args.sbert_path
        hparams.embedding.model_name = args.sbert_path
    if args.edit_layer is not None:
        hparams.inner_params[0] = f'model.layers[{args.edit_layer}].mlp.down_proj.weight'

                          
    router = None
    if args.router_load_path and not args.retrain:
        try:
            router = KnowRouter.load(args.router_load_path)
            print(f"Router loaded from {args.router_load_path}, "
                  f"clusters: {router.get_num_clusters()}")
        except Exception as e:
            print(f"Failed to load router: {e}, will retrain...")

    if router is None:
        print("Training router...")
        router = KnowRouter(cfg=hparams)
        router.build_route_table(prompt_list=prompts)
        print(f"Router trained, clusters: {router.get_num_clusters()}")
        if args.router_save_path:
            try:
                router.save(args.router_save_path)
                print(f"Router saved to {args.router_save_path}")
            except Exception as e:
                print(f"Failed to save router: {e}")

    return router


def run_editing(editor, data, method, args, router=None, loc_prompts=None):

                         
    kwargs = dict(
        prompts=data['prompts'],
        target_new=data['target_new'],
        subject=data['subject'],
        keep_original_weight=True,
        sequential_edit=args.sequential_edit,
    )

    if data.get('rephrase_prompts'):
        kwargs['rephrase_prompts'] = data['rephrase_prompts']
    if data.get('locality_inputs'):
        kwargs['locality_inputs'] = data['locality_inputs']
    if data.get('portability_inputs'):
        kwargs['portability_inputs'] = data['portability_inputs']
    if data.get('ground_truth'):
        kwargs['ground_truth'] = data['ground_truth']

                            
    if method == 'WISE' and loc_prompts is not None:
        kwargs['loc_prompts'] = loc_prompts
    if method == 'ASMem' and router is not None:
        kwargs['router'] = router

                                                                                   
    if method in ('EAMET',) or args.batch_edit:
        metrics, edited_model, weights_copy = editor.batch_edit(**kwargs)
    else:
        metrics, edited_model, weights_copy = editor.edit(**kwargs)

    return metrics, edited_model, weights_copy

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser(
        description="Unified Knowledge Editing Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

                      
    parser.add_argument('--dataset', required=True, type=str,
                        choices=['hallu', 'cknowedit', 'counterfact', 'zsre'],
                        help="Dataset: hallu (HalluEditBench), cknowedit, counterfact, zsre")
    parser.add_argument('--method', required=True, type=str,
                        choices=list(HPARAMS_MAP.keys()),
                        help="Editing method")
    parser.add_argument('--hparams_dir', required=True, type=str,
                        help="Path to hparams YAML file")
    parser.add_argument('--data_dir', required=True, type=str,
                        help="Data directory or file path")

                          
    parser.add_argument('--ds_size', default=None, type=int,
                        help="Number of samples (for cknowedit/counterfact/zsre)")
    parser.add_argument('--data_configs', default=None, type=str,
                        help="HalluEditBench config: 'ALL:1000' or 'domain:N,...'")

                             
    parser.add_argument('--sequential_edit', default=True, type=str2bool,
                        help="Use sequential (lifelong) editing")
    parser.add_argument('--batch_edit', default=False, type=str2bool,
                        help="Force batch_edit mode (all edits at once, e.g. MEMIT-MASS)")

                      
    parser.add_argument('--random_sample', default=False, type=str2bool)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--interleave', default=False, type=str2bool,
                        help="Round-robin domain interleaving per Eq.(9). "
                             "Default False to match submitted experiments.")

                    
    parser.add_argument('--output_dir', default='./outputs', type=str)

                                
    parser.add_argument('--router_save_path', default='./router', type=str)
    parser.add_argument('--result_tag', default='', type=str,
                        help='Optional suffix appended to method_tag in result filename (for ablation disambiguation)')
    parser.add_argument('--router_load_path', default='./router', type=str)
    parser.add_argument('--retrain', default=False, type=str2bool,
                        help="Force router retraining")
    parser.add_argument('--sbert_path', default=None, type=str,
                        help="SBERT model path for router")
    parser.add_argument('--two_stages', default=None, type=str2bool,
                        help="Two-stage routing")
    parser.add_argument('--boundary_model_name', default=None, type=str,
                        help="Fine-tuned SBERT for boundary detection")
    parser.add_argument('--boundary_threshold', default=None, type=float)
    parser.add_argument('--use_clustering', default=None, type=str2bool)
    parser.add_argument('--use_multi_ffn', default=None, type=str2bool)
    parser.add_argument('--random_routing', default=None, type=str2bool,
                        help="Ablation: random cluster assignment instead of anchor routing")
    parser.add_argument('--edit_layer', default=None, type=int,
                        help="Override editing layer index")

                           
    parser.add_argument('--loc_data_path', default=None, type=str,
                        help="WISE locality data path (default: auto-detect)")

    args = parser.parse_args()
    random.seed(args.seed)

                      
    if args.dataset == 'hallu' and args.data_configs is None:
        parser.error("--data_configs is required for HalluEditBench (e.g., 'ALL:1000')")
    if args.dataset in ('cknowedit', 'counterfact', 'zsre') and args.ds_size is None:
        parser.error(f"--ds_size is required for {args.dataset}")
    if args.method not in HPARAMS_MAP:
        parser.error(f"Unknown method: {args.method}")

    start_time = datetime.datetime.now()
    print(f"{'=' * 60}")
    print(f"Dataset: {args.dataset} | Method: {args.method}")
    print(f"Hparams: {args.hparams_dir}")
    print(f"Data: {args.data_dir}")
    print(f"Sequential edit: {args.sequential_edit}")
    print(f"Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}\n")

                          
    hparams = HPARAMS_MAP[args.method].from_hparams(args.hparams_dir)

                                                       
    if args.edit_layer is not None and args.method != 'ASMem':
        hparams.inner_params[0] = f'model.layers[{args.edit_layer}].mlp.down_proj.weight'

                          
    print(f"Loading dataset: {args.dataset}...")
    data = DATASET_LOADERS[args.dataset](args)
    n_edits = len(data['prompts'])
    print(f"Loaded {n_edits} editing instances\n")

                                     
    router = None
    if args.method == 'ASMem':
        router = build_or_load_router(hparams, data['prompts'], args)

                                   
    loc_prompts = None
    if args.method == 'WISE':
        loc_prompts = load_wise_loc_prompts(n_edits, args.loc_data_path)
        print(f"Loaded {len(loc_prompts)} WISE locality prompts\n")

                                 
    editor = BaseEditor.from_hparams(hparams)

    metrics, edited_model, weights_copy = run_editing(
        editor, data, args.method, args,
        router=router, loc_prompts=loc_prompts,
    )

                                                         
    end_time = datetime.datetime.now()
    duration = end_time - start_time

                                                                           
    method_tag = args.method
    if hasattr(hparams, 'blue') and getattr(hparams, 'blue', False) and args.method != 'BLUE':
        method_tag = f"{args.method}-BLUE"
    if args.batch_edit:
        method_tag += "-MASS"
    if args.random_routing:
        method_tag += "_random"
    if args.result_tag:
        method_tag += f"_{args.result_tag}"

    model_tag = hparams.model_name.split('/')[-1]
    result_name = f"{method_tag}_{args.dataset}_T{n_edits}_{model_tag}"
    os.makedirs(args.output_dir, exist_ok=True)

                                                                        
    src_log = os.path.join(os.getcwd(), 'logs', 'results.json')
    if os.path.exists(src_log):
        import shutil
        dst = os.path.join(args.output_dir, f"{result_name}.json")
        shutil.copy2(src_log, dst)
        print(f"\nResults saved to: {dst}")
    else:
                                         
        dst = os.path.join(args.output_dir, f"{result_name}.json")
        with open(dst, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)
        print(f"\nResults saved to: {dst}")

                     
    print(f"\n{'=' * 60}")
    print(f"Done!")
    print(f"  Method:     {method_tag}")
    print(f"  Dataset:    {args.dataset}")
    print(f"  Edits:      {n_edits}")
    print(f"  Model:      {model_tag}")
    layer_info = hparams.inner_params[0] if hasattr(hparams, 'inner_params') else getattr(hparams, 'layers', 'N/A')
    print(f"  Layer:      {layer_info}")
    print(f"  Sequential: {args.sequential_edit}")
    print(f"  Duration:   {duration}")
    print(f"  Output:     {dst}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
