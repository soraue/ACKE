import os.path
import sys
import json
import argparse
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# CUDA_VISIBLE_DEVICES=5

sys.path.append('..')
from easyeditor import (
    FTHyperParams,
    GraceHyperParams,
    MEMITHyperParams,
    ROMEHyperParams,
    MENDHyperParams,
    WISEHyperParams,
    ACKEHyperParams,
    #BaseEditor,
    summary_metrics,
)
from easyeditor.editors.editor import BaseEditor as BaseEditor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--editing_method', required=True, type=str)
    # parser.add_argument('--hparams_dir', required=True, type=str)
    # parser.add_argument('--data_dir', required=True, type=str)
    # parser.add_argument('--data_type', required=True, type=str,
    #                     choices=['ZsRE', 'temporal', 'hallucination'])
    # parser.add_argument('--output_dir', default='./outputs', type=str)
    # parser.add_argument('--ds_size', default=3, type=int)
    # parser.add_argument('--sequential_edit', action="store_true")

    args = parser.parse_args()

    args.editing_method='ACKE'
    args.ds_size=10
    args.data_type='ZsRE'
    args.sequential_edit=True
    #args.sequential_edit=False
    #args.output_dir='./outputs'

    args.hparams_dir='/root/autodl-tmp/EasyEdit-main/EasyEdit-main/hparams/ACKE/llama-7b.yaml'
    args.data_dir='/root/autodl-tmp/EasyEdit-main/EasyEdit-main/data'
    args.output_dir='/root/autodl-tmp/EasyEdit-main/EasyEdit-main/outputs'

    if args.editing_method == 'FT':
        editing_hparams = FTHyperParams
    elif args.editing_method == 'MEMIT':
        editing_hparams = MEMITHyperParams
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    elif args.editing_method == 'MEND':
        editing_hparams = MENDHyperParams
    elif args.editing_method == 'GRACE':
        editing_hparams = GraceHyperParams
    elif args.editing_method == 'WISE':
        editing_hparams = WISEHyperParams
    elif args.editing_method == 'ACKE':
        editing_hparams = ACKEHyperParams
    else:
        raise NotImplementedError

    K = args.ds_size

    #edit_data = json.load(open(f'{args.data_dir}/zsre_mend_eval.json', 'r', encoding='utf-8'))[:K]
    edit_data = json.load(open(f'{args.data_dir}/zsre_mend_edit.json', 'r', encoding='utf-8'))[:K]
    loc_data = json.load(open(f'{args.data_dir}/zsre_mend_train.json', 'r', encoding='utf-8'))[:K]
    loc_prompts = [edit_data_['loc'] + ' ' + edit_data_['loc_ans'] for edit_data_ in loc_data]

    prompts = [edit_data_['src'] for edit_data_ in edit_data]
    subject = [edit_data_['subject'] for edit_data_ in edit_data]
    rephrase_prompts = [edit_data_['rephrase'] for edit_data_ in edit_data]
    target_new = [edit_data_['alt'] for edit_data_ in edit_data]
    locality_prompts = [edit_data_['loc'] for edit_data_ in edit_data]
    locality_ans = [edit_data_['loc_ans'] for edit_data_ in edit_data]
    locality_inputs = {
        'neighborhood':{
            'prompt': locality_prompts,
            'ground_truth': locality_ans
        },
    }

    hparams = editing_hparams.from_hparams(f'{args.hparams_dir}')
    
    train_ds = None

    # qwen用
    # te = ['transformer.h.{}.mlp.fc_out.weight']

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir,
        f'{hparams.model_name.split("/")[-1]}_{args.editing_method}_N={args.ds_size}_Sequential={args.sequential_edit}.json'
        )

    print("See results at: ", output_file)

    eval_metric = {
        'ZsRE': 'token em',
        'hallucination': 'ppl',
        'temporal': 'ood_ppl'
    }

    editor = BaseEditor.from_hparams(hparams)

    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        rephrase_prompts=rephrase_prompts,
        target_new=target_new,
        loc_prompts=loc_prompts,
        subject=subject,
        locality_inputs=locality_inputs,
        sequential_edit=args.sequential_edit,
        eval_metric=eval_metric[args.data_type]
    )
    
    
    
    edited_model = edited_model.model
    
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=4)

    if len(metrics) > 0:
        summary_metrics(metrics)

