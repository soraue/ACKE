import globalact
import copy
import random

import torch
from torch.nn import functional as F
from .utils import parent_module, brackets_to_periods, EarlyStopMeter, EditingMeanAct
import transformers
import numpy as np
from torch import Tensor
from torch.nn import CrossEntropyLoss
from transformers.activations import ACT2FN
from .merge import slerp, GTA, linear
import torch.nn as nn
import gc

ACKElayer = True
ACKEmerge = True


merge_dict = {
    'slerp': slerp(),
    'ties': GTA('magnitude', 'sum', normalize=True),
    'magnitude_norm': GTA('magnitude', None, normalize=True),
    'magnitude': GTA('magnitude', None, normalize=False),
    'sign': GTA(None, 'sum', normalize=True),
    'dare_ties': GTA('rescaled_random', 'sum'),
    'dare_linear': GTA('random', None),
    'linear': linear()
}

edit_history = []
merge_group_edit_history = []

def euc(query, key, config, act_mask=None, infer=False):
    # Euclidean distance

    act_fn = ACT2FN[config.hidden_act]
    l2_norm = torch.norm(act_fn(key) - act_fn(query), dim=-1)
    if infer and l2_norm.size(1) > 100:
        topk = torch.topk(l2_norm, k=1, largest=True)
        return topk.values.mean()

    if act_mask is not None:
        return torch.sum(l2_norm * act_mask, dim=1) / torch.sum(act_mask, dim=1)
    else:
        return torch.mean(l2_norm, dim=-1)

class ACKE(torch.nn.Module):
    def __init__(self, config, model, device):
        super(ACKE, self).__init__()
        self.config = config
        self.model = model
        self.config = config
        if hasattr(self.model.config, 'hidden_act'):
            self.config.hidden_act = self.model.config.hidden_act
        elif hasattr(self.model.config, 'activation_function'):
            self.config.hidden_act = self.model.config.activation_function
        
        # 支持多个层的编辑
        if isinstance(config.inner_params, list):
            self.layers = config.inner_params
        else:
            self.layers = [config.inner_params]
        print(self.layers)
        global ACKElayer
        if ACKElayer:
            self.layers.append('model.layers[26].mlp.down_proj.weight')
            self.layers.append('model.layers[25].mlp.down_proj.weight')

        print(self.layers)
        
        self.device = device
        self.adapter_layers = {}  # 存储多个adapter层
        self.original_layers = {}  # 存储多个原始层

        # --- ensure proper formatting (ACKE edits weights matrices) ---
        suffixes = [".weight", ".bias"]

        
        for layer in self.layers:
            layer_name = layer.rsplit(".", 1)[0] if any(layer.endswith(x) for x in suffixes) else layer
            
            for n, p in self.model.named_parameters():
                p.requires_grad = False

            if isinstance(self.model, transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel):
                transpose = False
            else:
                transpose = True

            # --- Add ACKE to chosen layers ---
            edit_module = parent_module(self.model, brackets_to_periods(layer_name))
            layer_name_short = layer_name.rsplit(".", 1)[-1]
            self.adapter_layers[layer] = getattr(edit_module, layer_name_short)

            if type(self.adapter_layers[layer]) is not ACKEAdapter:
                setattr(edit_module, layer_name_short, ACKEAdapter(config, self.adapter_layers[layer], transpose=transpose))
                self.original_layers[layer] = copy.deepcopy(self.adapter_layers[layer])
                self.adapter_layers[layer] = getattr(edit_module, layer_name_short)
                print(f"New weights successfully inserted into {layer}")

        
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        

    # Forward
    def __call__(self, **kwargs):
        '''if not self.config.retrieve:
            global ACKEmerge
            if ACKEmerge:
                ACKEmerge = False
                print("final merge")
                for layer_name in self.layers:
                    self.get_adapter_layer(layer_name).merge_weight()
                    print(f'Merge Weight of (New, Original) Matrix for layer {layer_name}... with {self.config.merge_alg}')'''
        return self.model(**kwargs)

    def reset_layer(self, layer_name):
        layer_name_short = layer_name.rsplit(".", 1)[-1]
        layer = getattr(adapter_layer.edit_module, layer_name_short)
        del layer
        setattr(adapter_layer[layer_name].edit_module, layer_name_short, adapter_layer[layer_name].original_layer)

    def get_adapter_layer(self, layer_name=None):
        """
        获取指定层的adapter层
        
        Args:
            layer_name: 层名称，如果为None则返回第一个adapter层（向后兼容）
            
        Returns:
            对应的ACKEAdapter实例，如果不存在则返回None
        """
        
        if layer_name is None:
            # 向后兼容：如果没有指定layer_name，返回第一个adapter层
            print("没有接收到layername")
            if self.adapter_layers:
                return next(iter(self.adapter_layers.values())).to(self.model.device)
            return None
        
        # 返回指定层的adapter层
        if layer_name in self.adapter_layers:
                return self.adapter_layers[layer_name].to(self.model.device)

            #return self.adapter_layers[layer_name].to(self.model.device)
        else:
            raise ValueError(f"Layer '{layer_name}' not found in adapter_layers. Available layers: {list(self.adapter_layers.keys())}")

    # TODO: generation
    def generate(self, *args, **kwargs):
        # This method is no longer directly useful for multiple layers,
        # but keeping it for compatibility if called elsewhere.
        # For multiple layers, you'd need to pass the layer name.
        # For now, returning the first adapter for simplicity, but this might need refinement.
        if self.adapter_layers:
            layer_name = next(iter(self.adapter_layers.keys()))
            setattr(eval(f"self.model.{layer_name}"), "key_id", -1)
            return self.model.generate(*args, **kwargs)
        return self.model.generate(*args, **kwargs)

    def edit(self, config, tokens, act_mask=None, deact_mask=None):
        
        # for retrieve ##
        global edit_history
        global merge_group_edit_history
        edit_history.append([{f"{k1}" : v1.to('cpu') for k1, v1 in tokens.items()}, False])
        # for retrieve ##
        last_prompt_token_loc = (tokens["labels"] == -100).sum(dim=-1) - 1

        for layer_name in self.layers:
            print(layer_name)

            self.get_adapter_layer(layer_name).training = True
            self.get_adapter_layer(layer_name).editing = True
            self.get_adapter_layer(layer_name).set_parameter_tunable()

            if self.get_adapter_layer(layer_name).editing_total_cnt % self.config.save_freq == 0:
                self.get_adapter_layer(layer_name).generate_activation_mask(self.config.mask_ratio)

        # --- train Wise value ---
        loss_meter = EarlyStopMeter()
        for i in range(config.n_iter):

            if i == 0:

                params_to_optimize = []
                for layer_name in self.layers:
                    params_to_optimize.append(self.get_adapter_layer(layer_name).new_weight)
                
                optimizer = torch.optim.SGD(params_to_optimize, config.edit_lr, weight_decay=1e-5)

            ft_loss = self._cal_ft_loss(tokens, last_prompt_token_loc)
            loss = ft_loss


            if loss_meter.stop():
                for layer_name in self.layers:
                    self.get_adapter_layer(layer_name).save_editing_activation()
                break
            if i == config.n_iter - 1:
                for layer_name in self.layers:
                    self.get_adapter_layer(layer_name).save_editing_activation()

            optimizer.zero_grad()
            loss.backward()

            # 计算分层权重
            layer_weights = self._cal_adaptive_layer_weights()
            
            # 应用分层权重到梯度
            for layer_name in self.layers:
                adapter = self.get_adapter_layer(layer_name)
                if adapter.new_weight.grad is not None:
                    adapter.new_weight.grad *= layer_weights[layer_name]

            # 对所有层应用梯度掩码
            for layer_name in self.layers:
                self.get_adapter_layer(layer_name).mask_new_weight_gradient()

            print(f"loss {np.round(loss.item(), 3)} = {np.round(ft_loss.item(), 3)}")

            optimizer.step()
            loss_meter.update(loss.item())

            # 对所有层应用范数约束
            if type(self.config.norm_constraint) is float:
                self._norm_constraint(self.config.norm_constraint)

        # --- 退出编辑模式 ---
        for layer_name in self.layers:
            #setattr(eval(f"self.model.{layer_name}"), "editing", False)
            #setattr(eval(f"self.model.{layer_name}"), "training", False)
            self.get_adapter_layer(layer_name).training = False
            self.get_adapter_layer(layer_name).editing = False

            
            #editing_total_cnt = getattr(eval(f"self.model.{layer_name}"), "editing_total_cnt") + 1
            #setattr(eval(f"self.model.{layer_name}"), "editing_total_cnt", editing_total_cnt)
            editing_total_cnt = self.get_adapter_layer(layer_name).editing_total_cnt + 1
            self.get_adapter_layer(layer_name).editing_total_cnt = editing_total_cnt
            
            # 保存权重到记忆
            if self.config.save_freq is not None and editing_total_cnt % self.config.save_freq == 0:
                self.get_adapter_layer(layer_name).save_weight()
                print(f'Add New Weight to Memory for layer {layer_name}...')
            
            # 定期合并权重
            if editing_total_cnt % self.config.merge_freq == 0:
                self.get_adapter_layer(layer_name).merge_weight()
                print(f'Merge Weight of (New, Original) Matrix for layer {layer_name}... with {self.config.merge_alg}')

        # 更新编辑历史
        if any(self.get_adapter_layer(layer_name).editing_total_cnt % self.config.merge_freq == 0 for layer_name in self.layers):
            merge_group_edit_history.append(edit_history)
            edit_history = []
            
        global ACKElayer
        if ACKElayer:
            ACKElayer = False

    def _norm_constraint(self, norm_constraint):
        for layer_name in self.layers:
            adapter_layer = self.get_adapter_layer(layer_name)
            new_weight = adapter_layer.new_weight
            original_weight = adapter_layer.weight
            with torch.no_grad():
                new_weight[...] = torch.clamp(
                    new_weight, min=original_weight - norm_constraint, max=original_weight + norm_constraint
                )

    def _cal_ft_loss(self, tokens, last_prompt_token_loc):
        if hasattr(self.model.config, 'batch_size'):
            k = self.config.batch_size
        else:
            k = 1
        bs = tokens["input_ids"].shape[0] - k
        logits = self.model(**tokens).logits
        shift_logits = logits[:-k, :-1, :].contiguous()
        shift_labels = tokens['labels'][:-k, 1:].contiguous()

        loss_fct = CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(bs, -1)

        label_mask = torch.zeros_like(loss, dtype=torch.bool)

        for i, col_index in enumerate(last_prompt_token_loc[:-k]):
            label_mask[i, col_index - 1:] = True

        ft_loss = ((loss * label_mask).sum(1) / label_mask.sum(1)).mean()
        return ft_loss
    
    def _cal_activation_loss(self, original_layer_output, new_weight_layer_output, config=None, act_mask=None,
                              deact_mask=None):
        if hasattr(self.model.config, 'batch_size'):
            k = self.config.batch_size
        else:
            k = 1
        total_loss = []
        len_temp = original_layer_output.shape[0] / k - 1
        for i,act_mk in enumerate(act_mask):
            if act_mk is not None:
                in_scope_dist = euc(original_layer_output[int(i*len_temp):int((i+1)*len_temp), ...], new_weight_layer_output[int(i*len_temp):int((i+1)*len_temp), ...], config,
                                    act_mask=act_mk)
                out_scope_dist = euc(original_layer_output[int(i*len_temp):int((i+1)*len_temp), ...], new_weight_layer_output[int(i*len_temp):int((i+1)*len_temp), ...], config,
                                    act_mask=deact_mask[i])
            else:
                in_scope_dist = euc(original_layer_output[int(i*len_temp):int((i+1)*len_temp), ...], new_weight_layer_output[int(i*len_temp):int((i+1)*len_temp), ...], config)
                if (i==k-1):
                    out_scope_dist = euc(original_layer_output[int(i-k):, ...], new_weight_layer_output[int(i-k):, ...], config)
                else:
                    out_scope_dist = euc(original_layer_output[int(i-k):int(i+1-k), ...], new_weight_layer_output[int(i-k):int(i+1-k), ...], config)

            loss = out_scope_dist.view(-1,1) - in_scope_dist + config.gamma
            loss2 = out_scope_dist - config.alpha
            loss3 = config.beta - in_scope_dist
            loss3 = torch.mean(loss3[loss3 > 0]) if min(loss3[loss3 > 0].size()) > 0 else torch.tensor(0.).to(original_layer_output.device)
            loss2 = torch.mean(loss2[loss2 > 0]) if min(loss2[loss2 > 0].size()) > 0 else torch.tensor(0.).to(original_layer_output.device)
            loss = torch.mean(loss[loss > 0]) if min(loss[loss > 0].size()) > 0 else torch.tensor(0.).to(original_layer_output.device)
            total_loss.append(loss + loss2 + loss3)
        return sum(total_loss) / len(total_loss)

    def _cal_adaptive_layer_weights(self):
        """
        根据每层的重要性自适应调整权重
        """
        layer_weights = {}
        total_importance = 0
        
        for layer_name in self.layers:
            adapter = self.get_adapter_layer(layer_name)
            
            # 计算层的编辑重要性（基于梯度范数）
            if adapter.new_weight.grad is not None:
                importance = torch.norm(adapter.new_weight.grad)
            else:
                importance = 1.0  # 默认重要性
            
            layer_weights[layer_name] = importance
            total_importance += importance
        
        # 归一化权重，可能要改
        for layer_name in layer_weights:
            layer_weights[layer_name] /= total_importance
        
        return layer_weights

    def _cal_memory_pos_activation_loss(self, original_layer_output, new_weight_layer_output, config=None, act_mask=None,
                              deact_mask=None):
        if hasattr(self.model.config, 'batch_size'):
            k = self.config.batch_size
        else:
            k = 1
        in_scope_dist = euc(original_layer_output[:-k, ...], new_weight_layer_output[:-k, ...], config)
        loss4 = 20 - in_scope_dist

        return torch.mean(loss4[loss4 > 0]) if min(loss4[loss4 > 0].size()) > 0 else torch.tensor(0.)

    def _cal_memory_neg_activation_loss(self, original_layer_output, new_weight_layer_output, config=None, act_mask=None,
                              deact_mask=None):
        if hasattr(self.model.config, 'batch_size'):
            k = self.config.batch_size
        else:
            k = 1
        in_scope_dist = euc(original_layer_output[:-k, ...], new_weight_layer_output[:-k, ...], config)
        loss4 = in_scope_dist - 5

        return torch.mean(loss4[loss4 > 0]) if min(loss4[loss4 > 0].size()) > 0 else torch.tensor(0.)

    def save(self, save_path):
        import os
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)  # Create the directory if it doesn't exist

        # Save additional information, such as memory_weight, memory_mean_act, etc.
        additional_info = {}
        for layer_name, adapter_layer in self.adapter_layers.items():
            additional_info[layer_name] = {
                'memory_weight': adapter_layer.memory_weight,
                'memory_mean_act': adapter_layer.memory_mean_act,
                'merge_cnt': adapter_layer.merge_cnt,
                'editing_mean_act': adapter_layer.editing_mean_act,
                'editing_total_cnt': adapter_layer.editing_total_cnt,
                'weight_mask': adapter_layer.weight_mask,
                # Add other variables that need to be saved
            }
            if hasattr(adapter_layer, 'key_id') and adapter_layer.key_id is not None:
                additional_info[layer_name]['key_id'] = adapter_layer.key_id
        # Save all information to the file
        torch.save({
            'adapter_state_dict': {k: v.state_dict() for k, v in self.adapter_layers.items()},
            'config': self.config,
            'additional_info': additional_info,
            'edit_history': edit_history,
            'merge_group_edit_history': merge_group_edit_history
        }, save_path)

    def load(self, load_path):
        import os
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Checkpoint file not found: {load_path}")

        # Load all previously saved information
        saved_data = torch.load(load_path)
        if hasattr(self.model.config, 'hidden_act'):
            saved_data['config'].hidden_act = self.model.config.hidden_act
        elif hasattr(self.model.config, 'activation_function'):
            saved_data['config'].hidden_act = self.model.config.activation_function
        if saved_data['config'] != self.config:
            print("Warning: The loaded ACKE config is different from the original config")

        # Restore the state dictionary of the ACKE Adapter instance
        for layer_name, adapter_layer in self.adapter_layers.items():
            adapter_layer.load_state_dict(saved_data['adapter_state_dict'][layer_name])
        # Restore additional information
        for layer_name, adapter_layer in self.adapter_layers.items():
            for key, value in saved_data['additional_info'][layer_name].items():
                setattr(adapter_layer, key, value)
        
        # Restore editing history
        global edit_history, merge_group_edit_history
        edit_history = saved_data['edit_history']
        merge_group_edit_history = saved_data['merge_group_edit_history']
        print(f"Model configuration and ACKE state loaded from {load_path}")



class ACKEAdapter(torch.nn.Module):
    def __init__(self, config, layer, transpose):
        super(ACKEAdapter, self).__init__()

        self.layer = layer
        self.weight = self.layer.weight
        self.device = layer.weight.device
        self.config = config
        self.new_weight = copy.deepcopy(self.weight)
        self.original_layer = copy.deepcopy(self.layer)
        self.memory_weight = []
        self.memory_mean_act = []
        if 'gpt2' in self.config.model_name:
            self.bias = self.layer.bias # For Conv1D
        else:
            self.bias = None
        self.merge_cnt = 0  # only for retrieve
        assert not self.weight.requires_grad, print('Original Layer can not be tunable....')

        self.used_mask = None 

        if transpose:
            self.key_shape = layer.weight.shape[1]
            self.value_shape = layer.weight.shape[0]
        else:
            self.key_shape = layer.weight.shape[0]
            self.value_shape = layer.weight.shape[1]
        self.training = False
        self.editing = False

        self.editing_mean_act = EditingMeanAct()
        self.editing_total_cnt = 0

    def set_parameter_tunable(self):
        self.new_weight.requires_grad = True

    def save_weight(self):
        self.memory_weight.append(copy.deepcopy(self.new_weight))
        self.new_weight = copy.deepcopy(self.original_layer.weight)
        if self.config.retrieve:
            self.memory_mean_act.append(copy.deepcopy(self.editing_mean_act))
            self.editing_mean_act = EditingMeanAct()

    def merge_weight(self):
        if self.config.save_freq is not None:  # for ties dare dare_ties
            if not self.config.retrieve:
                merge_alg = merge_dict[self.config.merge_alg]
                if self.original_layer.weight.equal(self.layer.weight):
                    cur_new_weight = merge_alg.execute([self.config.weights / len(self.memory_weight) for _ in range(len(self.memory_weight))], self.original_layer.weight, self.memory_weight, densities=self.config.densities)
                else:
                    cur_new_weight = merge_alg.execute([0.4 / len(self.memory_weight) for _ in range(len(self.memory_weight))] + [0.6], self.original_layer.weight, self.memory_weight + [self.layer.weight], densities=self.config.densities)
                self.layer.weight = torch.nn.Parameter(cur_new_weight.to(self.layer.weight.device), requires_grad=False)
                self.new_weight = copy.deepcopy(self.original_layer.weight)
                del self.memory_weight
                self.memory_weight = []
            else:
                merge_alg = merge_dict[self.config.merge_alg]
                merge_num = self.config.merge_freq // self.config.save_freq
                assert len(self.memory_weight) >= merge_num
                new_merge_weight = merge_alg.execute([self.config.weights / merge_num for _ in range(merge_num)], self.original_layer.weight, self.memory_weight[-merge_num:], densities=self.config.densities)
                min_a = 1e9
                for _ in range(merge_num):
                    self.memory_weight.pop()
                    edit_act = self.memory_mean_act.pop()
                    min_a = min(min_a, edit_act.min_act())
                self.new_weight = copy.deepcopy(self.original_layer.weight)
                self.memory_weight.append(new_merge_weight)
                self.memory_mean_act.append(EditingMeanAct(min_a=min_a))
                print(len(self.memory_weight))
                assert len(self.memory_mean_act) == len(self.memory_weight)
                self.merge_cnt += 1
        else:
            merge_alg = merge_dict[self.config.merge_alg]
            cur_new_weight = merge_alg.execute(0.5, self.layer.weight, [self.new_weight],
                                               densities=self.config.densities)
            self.layer.weight = torch.nn.Parameter(cur_new_weight.to(self.layer.weight.device), requires_grad=False)
            self.new_weight = copy.deepcopy(self.original_layer.weight)

    def save_editing_activation(self):
        in_scope_dist = euc(self.original_layer_output[:-1, ...], self.new_weight_layer_output[:-1, ...], self.config)
        self.editing_mean_act.update(in_scope_dist.mean().item())

    def generate_activation_mask(self, mask_ratio):
        # 写的很好懂了，就是随机生成mask_ratio比例个1的权重mask(就是1的比例是mask_ratio的，1是随机分布的和权重矩阵同形状的01矩阵)
        p_grad = self.new_weight.reshape(-1)
        p_mask = np.random.choice([1, 0], size=p_grad.size()[0], p=[mask_ratio, 1 - mask_ratio])
        p_mask = torch.from_numpy(p_mask).to(p_grad.device)
        self.weight_mask = p_mask

    # 没被用过，那mask应该就是没变动过
    def generate_non_overlapping_mask(self, mask_ratio):
        p_grad = self.new_weight.reshape(-1)
        mask_size = int(mask_ratio * p_grad.size()[0])
        if self.used_mask is None:
            self.used_mask = np.zeros(p_grad.size()[0], dtype=bool)
        available_indices = np.where(~self.used_mask)[0]  # 获取未被遮罩的元素索引
        if len(available_indices) < mask_size:
            raise ValueError("Not enough unused elements to generate a new mask.")
        chosen_indices = np.random.choice(available_indices, size=mask_size, replace=False)
        mask_array = np.zeros(p_grad.size()[0], dtype=int)
        mask_array[chosen_indices] = 1
        self.used_mask[chosen_indices] = True  # 更新遮罩状态
        self.weight_mask = torch.from_numpy(mask_array).to(p_grad.device)

    def new_weight_forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.new_weight) if self.bias is None else torch.addmm(self.bias, input.view(-1, input.size(-1)), self.new_weight).view(input.size()[:-1] + (self.layer.nf,))

    def mask_new_weight_gradient(self):
        assert self.new_weight.grad is not None, print('Gradient Collection for New Weight error, gradient not found')
        # Add gradient mask after the loss updates
        p_size = self.new_weight.grad.size()
        p_grad = self.new_weight.grad.reshape(-1)

        # mask = torch.from_numpy(np.random.choice([0, 1], size=p_grad.size()[0], p=[.1, .9])).cuda()
        p_grad = p_grad * self.weight_mask
        self.new_weight.grad = p_grad.view(p_size).to(self.new_weight.grad.dtype)

    def forward(self, *args):
        if self.editing:
            layer_out = self.new_weight_forward(*args)
            self.new_weight_layer_output = layer_out
            self.original_layer_output = self.original_layer(*args)
        else:
            #no merge
            original_layer_output = self.original_layer(*args)
            #layer_out = self.new_weight_forward(*args)
            
            #merge
            layer_out = self.layer(*args)
            
            if globalact.actglo == False:
                with open ('./nnocheck.txt','a') as file:
                    file.write('ori\n')
                    file.close()
                layer_out = original_layer_output
            else:
                with open ('./nnocheck.txt','a') as file:
                    file.write('new\n')
                    file.close()
                
        return layer_out


class ACKEMultimodal(ACKE):
    def edit(self, config, multimodal_inputs, text_tokens, ans_token_len, act_mask=None, deact_mask=None):
        global edit_history
        global merge_group_edit_history
        edit_history.append([{f"{k1}" : v1.to('cpu') for k1, v1 in text_tokens.items()}, False])
        last_prompt_token_loc = (text_tokens["labels"] == -100).sum(dim=-1) - 1
        
        setattr(eval(f"self.model.{self.layers[0]}"), "training", True)
        setattr(eval(f"self.model.{self.layers[0]}"), "editing", True)
        self.adapter_layers[self.layers[0]].set_parameter_tunable()
        if getattr(eval(f"self.model.{self.layers[0]}"), "editing_total_cnt") % self.config.save_freq == 0:
            self.adapter_layers[self.layers[0]].generate_activation_mask(self.config.mask_ratio)        
        
        # --- train Wise value ---
        loss_meter = EarlyStopMeter()
        for i in range(config.n_iter):
            if i == 0:
                # --- we only need to create an optimizer for the first iteration (but forward pass instantiates the key, so optimzer is passed after first inference) ---
                optimizer = torch.optim.SGD([super().get_adapter_layer().new_weight], config.edit_lr, weight_decay=1e-5)

            ft_loss = self._cal_ft_loss(multimodal_inputs, text_tokens, last_prompt_token_loc, ans_token_len)

            act_loss = super()._cal_activation_loss(super().get_adapter_layer().original_layer_output, super().get_adapter_layer().new_weight_layer_output,
                                                  config=config, act_mask=act_mask, deact_mask=deact_mask)
            loss = ft_loss + act_loss.to(ft_loss.device)

            if loss_meter.stop():
                super().get_adapter_layer().save_editing_activation()  # add last gradient
                break
            if i == config.n_iter - 1:
                super().get_adapter_layer().save_editing_activation()  # add last gradient

            if self.config.retrieve and super().get_adapter_layer().merge_cnt > 0 and self.config.replay:
                memory_loss = []
                for _ in merge_group_edit_history:
                    idx = 0
                    while True:
                        memo_input, is_used = _[idx]
                        if not is_used:
                            _[idx][1] = True
                            break
                        idx += 1
                        if idx == len(_): ## re Assign
                            for m in range(len(_)):
                                _[m][1] = False
                            idx = 0

                    memo_input = {f"{k1}" : v1.to(self.config.device) for k1, v1 in memo_input.items()}
                    self.model(**memo_input)

                    memory_act_loss = super()._cal_memory_neg_activation_loss(super().get_adapter_layer().original_layer_output,
                                                    super().get_adapter_layer().new_weight_layer_output, config=config,
                                                    act_mask=act_mask, deact_mask=deact_mask)
                    memory_loss.append(memory_act_loss.to(ft_loss.device))
                    del memo_input
                neg_memo_loss = torch.stack(memory_loss).mean()
                loss += neg_memo_loss
                if len(edit_history) > 0:
                    memo_input = random.choice(edit_history)[0]
                    memo_input = {f"{k1}" : v1.to(self.config.device) for k1, v1 in memo_input.items()}
                    self.model(**memo_input)

                    pos_memo_loss = super()._cal_memory_pos_activation_loss(super().get_adapter_layer().original_layer_output,
                                                    super().get_adapter_layer().new_weight_layer_output, config=config,
                                                    act_mask=act_mask, deact_mask=deact_mask)
                    del memo_input
                    loss += pos_memo_loss.to(ft_loss.device)
            # for replay Appendix B.3

            optimizer.zero_grad()

            loss.backward()
            super().get_adapter_layer().mask_new_weight_gradient()

            if self.config.retrieve and super().get_adapter_layer().merge_cnt > 0 and self.config.replay:
                print(
                    f"loss {np.round(loss.item(), 3)} = {np.round(ft_loss.item(), 3)} + {np.round(act_loss.item(), 3)} + {np.round(neg_memo_loss.item(), 3)} + {np.round(pos_memo_loss.item(), 3)}"
                )
            else:
                print(
                    f"loss {np.round(loss.item(), 3)} = {np.round(ft_loss.item(), 3)} + {np.round(act_loss.item(), 3)}"
                )

            optimizer.step()
            loss_meter.update(loss.item())

            if type(self.config.norm_constraint) is float:
                super()._norm_constraint(self.config.norm_constraint)

        # --- pull out info we want to log from the Wise layer ---
        setattr(eval(f"self.model.{self.layers[0]}"), "editing", False)
        setattr(eval(f"self.model.{self.layers[0]}"), "training", False)

        editing_total_cnt = getattr(eval(f"self.model.{self.layers[0]}"), "editing_total_cnt") + 1
        setattr(eval(f"self.model.{self.layers[0]}"), "editing_total_cnt", editing_total_cnt)
        if self.config.save_freq is not None and editing_total_cnt % self.config.save_freq == 0:
            super().get_adapter_layer().save_weight()
            print(f'Add New Weight to Memory...')
        if editing_total_cnt % self.config.merge_freq == 0:
            # for retrieve ##
            merge_group_edit_history.append(edit_history)
            edit_history = []
            # for retrieve ##

            super().get_adapter_layer().merge_weight()
            print(f'Merge Weight of (New, Original) Matrix... with {self.config.merge_alg}')

    def _cal_ft_loss(self, multimodal_inputs, text_tokens, last_prompt_token_loc, ans_token_len):
        if hasattr(self.model.config, 'batch_size'):
            k = self.config.batch_size
        else:
            k = 1
        
        if k != 1:
            raise AssertionError("Not support Batch Edit")
        
        bs = text_tokens["input_ids"].shape[0] - k
        logits = self.model(**multimodal_inputs).logits
        shift_logits = logits[:-k, :-1, :].contiguous()
        shift_labels = multimodal_inputs['input_ids'][:-k, 1:].contiguous()
        # only cal loss of target text tokens
        loss_fct = CrossEntropyLoss(reduction='none')
        a = shift_logits.view(-1, shift_logits.size(-1))
        b = shift_labels.view(-1)[-ans_token_len:]
        a = a[-b.size(0):,:]
        loss = loss_fct(a, b)
        loss = loss.view(bs, -1)
        label_mask = torch.ones_like(loss, dtype=torch.bool)        
        ft_loss = ((loss * label_mask).sum(1) / label_mask.sum(1)).mean()
        return ft_loss
    