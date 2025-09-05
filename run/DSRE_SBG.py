import os
import sys
from typing import List, Dict, Optional, Any, Tuple

from sentence_transformers.evaluation import SentenceEvaluator
from sklearn.model_selection import train_test_split
from multiarea_dataset import MultiAreaDataset
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
import math
import json
import datasets
from sentence_transformers.losses.BatchHardTripletLoss import BatchHardTripletLossDistanceFunction
sys.path.append(os.getcwd() + '/EasyEdit')
sys.path.append(os.getcwd() + '/EasyEdit/run_bishe')

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

def prepare_triplet_data(root_dir: str,
                         dataset_configs: Dict[str, int],
                         validation_split_ratio: float = 0.1,
                         seed: int = 1234,
                         random_sample: bool = False) -> tuple[datasets.Dataset, datasets.Dataset]:

    multiarea_dataset = MultiAreaDataset(
        root_dir=root_dir,
        dataset_configs=dataset_configs,
        seed=seed,
        random_sample=random_sample
    )

    prompts, rephrase_prompts, _, _, locality_inputs, _ = multiarea_dataset.to_edit_dataset()
    locality_prompts = locality_inputs['neighborhood']['prompt']

    min_len = min(len(prompts), len(rephrase_prompts), len(locality_prompts))
    if not (len(prompts) == len(rephrase_prompts) == len(locality_prompts)):
        prompts = prompts[:min_len]
        rephrase_prompts = rephrase_prompts[:min_len]
        locality_prompts = locality_prompts[:min_len]

    if min_len == 0:
        empty_data = {'anchor': [], 'positive': [], 'negative': []}
        return datasets.Dataset.from_dict(empty_data), datasets.Dataset.from_dict(empty_data)

    train_anchors, val_anchors, \
        train_positives, val_positives, \
        train_negatives, val_negatives = train_test_split(
        prompts,
        rephrase_prompts,
        locality_prompts,
        test_size=validation_split_ratio,
        random_state=seed,
        shuffle=True
    )

    train_data_dict = {
        'anchor': train_anchors,
        'positive': train_positives,
        'negative': train_negatives
    }
    val_data_dict = {
        'anchor': val_anchors,
        'positive': val_positives,
        'negative': val_negatives
    }

    features = datasets.Features({
        'anchor': datasets.Value('string'),
        'positive': datasets.Value('string'),
        'negative': datasets.Value('string')
    })

    train_dataset = datasets.Dataset.from_dict(train_data_dict, features=features)
    val_dataset = datasets.Dataset.from_dict(val_data_dict, features=features)

    if train_dataset:
        train_dataset.info.dataset_name = "multi_area_triplet_train"
        train_dataset.info.description = "Training dataset for multi-area triplet loss fine-tuning."
    if val_dataset:
        val_dataset.info.dataset_name = "multi_area_triplet_validation"
        val_dataset.info.description = "Validation dataset for multi-area triplet loss fine-tuning."

    return train_dataset, val_dataset

def finetune_sentence_transformer(
        data_root_dir: str,
        data_configs: Dict[str, int],
        base_model_name: str,
        output_model_dir: str,
        final_model_subdir: str,
        distance_metric_name: str,
        triplet_margin: float,
        num_train_epochs: int,
        train_batch_size: int,
        learning_rate: float,
        warmup_ratio: float,
        weight_decay: float,
        logging_steps: int,
        save_strategy: str,
        evaluation_strategy: str,
        eval_steps: int,
) -> Optional[str]:

    train_dataset, eval_dataset = prepare_triplet_data(root_dir=data_root_dir, dataset_configs=data_configs)

    model = SentenceTransformer(base_model_name)

    if distance_metric_name.upper() == "COSINE":
        distance_metric = losses.SiameseDistanceMetric.COSINE_DISTANCE
    elif distance_metric_name.upper() == "EUCLIDEAN":
        distance_metric = losses.SiameseDistanceMetric.EUCLIDEAN
    else:
        distance_metric = losses.SiameseDistanceMetric.COSINE_DISTANCE

    loss_func = losses.TripletLoss(model=model, distance_metric=distance_metric, triplet_margin=triplet_margin)

    steps_per_epoch = math.ceil(len(train_dataset) / train_batch_size)
    total_steps = steps_per_epoch * num_train_epochs
    warmup_steps = math.ceil(total_steps * warmup_ratio)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_model_dir,

        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=train_batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,

        logging_dir=os.path.join(output_model_dir, 'logs'),
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        evaluation_strategy=evaluation_strategy if eval_dataset else "no",
        eval_steps= eval_steps if evaluation_strategy == "steps" and eval_dataset else None,
        load_best_model_at_end=True if eval_dataset else False,

        report_to="tensorboard",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss_func,
    )

    trainer.train()

    final_model_path = os.path.join(output_model_dir, final_model_subdir)

    os.makedirs(final_model_path, exist_ok=True)
    model.save(final_model_path)

    return final_model_path

if __name__ == "__main__":

    ALL_FILES = [
        "art_sculpture.json", "business_brand.json", "business_corporation.json",
        "business_industry.json", "entertainment_anime.json", "entertainment_music_genre.json",
        "entertainment_song.json", "event_film.json", "event_history.json",
        "event_sport.json", "geography_forest.json", "geography_glacier.json",
        "geography_volcano.json", "health_disease.json", "health_medication.json",
        "health_symptom.json", "human_athlete.json", "human_entrepreneur.json",
        "human_scientist.json", "human_writer.json", "places_city.json",
        "places_country.json", "places_landmark.json", "technology_database.json",
        "technology_programming_language.json", "technology_software.json"
    ]

    dataset_configs = {file: 1000 for file in ALL_FILES}

    multi_area_root_dir = '/data/HalluEditBench'

    train_hparams = {
        "base_model_name": 'sentence-transformers/all-MiniLM-L6-v2',
        "output_model_dir": './finetuned_sbert_triplet',
        "final_model_subdir": 'final_model',
        "distance_metric_name": "COSINE",
        "triplet_margin": 0.9,
        "num_train_epochs": 3,
        "train_batch_size": 8,
        "learning_rate": 1e-5,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "logging_steps": 50,
        "save_strategy": "epoch",
        "evaluation_strategy": "epoch",
        "eval_steps": 100,
    }

    final_model_path = finetune_sentence_transformer(
        data_root_dir=multi_area_root_dir,
        data_configs=dataset_configs,
        **train_hparams
    )
