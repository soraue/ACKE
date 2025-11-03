from typing import Any, Dict, List, Tuple
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer
from .ACKE import ACKE, ACKEMultimodal
from .ACKE_act import ACKE as ACKE_act
from .utils import tokenize, multimodal_tokenize, get_context_templates
from .acke_hparams import ACKEHyperParams
ACKEload = False

def apply_acke_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: ACKEHyperParams,
        copy=False,
        **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    if copy:
        model = deepcopy(model)
    device = f'cuda:{hparams.device}'
    context_templates = get_context_templates(model, tok, length_params=[[5,5], [10,5]], device=device)
    editor = ACKE(model=model, config=hparams, device=device)
    editor_act = ACKE_act(model=model, config=hparams, device=device)
    import os
    global ACKEload
    if hasattr(hparams, 'load_path') and hparams.load_path and os.path.exists(hparams.load_path) and ACKEload:
        print("Start loading the ACKE model!")
        editor.load(hparams.load_path)
        ACKEload=False
    print(f"Executing ACKE algorithm for the update: ")
    for request in requests:
        print(
            f"[{request['prompt']}] -> [{request['target_new']}]"
        )
    tokens, act_mask, deact_mask = tokenize(requests, tokenizer=tok, device=device, context_templates=context_templates, hparams=hparams)
    editor_act.edit(config=hparams, tokens=tokens, act_mask=act_mask, deact_mask=deact_mask)
    editor.edit(config=hparams, tokens=tokens, act_mask=act_mask, deact_mask=deact_mask)

    weights_copy = editor.reset_layer

    return editor, weights_copy

def apply_acke_to_model_act(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: ACKEHyperParams,
        copy=False,
        **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    if copy:
        model = deepcopy(model)
    device = f'cuda:{hparams.device}'
    context_templates = get_context_templates(model, tok, length_params=[[5,5], [10,5]], device=device)
    editor = ACKE_act(model=model, config=hparams, device=device)
    import os
    global ACKEload
    if ACKEload:
        print("Start loading the ACKE model!")
        #editor.load("./acke_checkpoint/llama_act1000.pt")
        #ACKEload=False
    print(f"Executing ACKE algorithm for the update: ")
    for request in requests:
        print(
            f"[{request['prompt']}] -> [{request['target_new']}]"
        )
    tokens, act_mask, deact_mask = tokenize(requests, tokenizer=tok, device=device, context_templates=context_templates, hparams=hparams)
    editor.edit(config=hparams, tokens=tokens, act_mask=act_mask, deact_mask=deact_mask)

    weights_copy = editor.reset_layer

    return editor, weights_copy


def apply_acke_to_multimodal_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: ACKEHyperParams,
        copy=False,
        **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    device = f'cuda:{hparams.device}'    
    if copy:
        model = deepcopy(model)
        model.to(device)

    editor = ACKEMultimodal(model=model, config=hparams, device=device)
    import os
    global ACKEload
    if hasattr(hparams, 'load_path') and hparams.load_path and os.path.exists(hparams.load_path) and ACKEload:
        print("Start loading the ACKE model!")
        editor.load(hparams.load_path)
        ACKEload=False
    print(f"Executing ACKE algorithm for the update: ")
    for request in requests:
        print(
            f"[{request['prompt']}] -> [{request['target']}]"
        )

    multimodal_inputs, text_tokens, ans_token_len, act_mask, deact_mask = multimodal_tokenize(requests, processor=tok, device=device, context_templates=None, hparams=hparams)
    editor.edit(config=hparams, multimodal_inputs=multimodal_inputs, ans_token_len=ans_token_len, text_tokens=text_tokens, act_mask=act_mask, deact_mask=deact_mask)
    weights_copy = editor.reset_layer
    return editor, weights_copy

