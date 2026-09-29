"""Default recipe: 1,200 steps x 32 sequences x 256 targets = 9,830,400 tokens."""
import argparse
import json
import math
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from common import PROTOCOL, ROOT, autocast, device_metrics, make_model, setup, sha
from dev_data import load_development_data
from evaluate import score
from ema import ModelEMA


def make_parameter_groups(model, weight_decay, policy):
    """Group unique trainable parameters; keep tied embeddings only once."""
    named = [(name, parameter) for name, parameter in model.named_parameters()
             if parameter.requires_grad]
    if policy == 'all':
        partitions = [('all', named, weight_decay)]
    elif policy == 'matrix':
        partitions = [
            ('decay', [(name, parameter) for name, parameter in named
                       if parameter.ndim >= 2], weight_decay),
            ('no_decay', [(name, parameter) for name, parameter in named
                          if parameter.ndim < 2], 0.0),
        ]
    else:
        raise ValueError(f'Unknown weight decay policy: {policy}')
    groups = []
    summary = []
    for label, parameters, decay in partitions:
        if not parameters:
            continue
        groups.append({'params': [parameter for _, parameter in parameters],
                       'weight_decay': decay})
        summary.append({'name': label, 'weight_decay': decay,
                        'tensors': len(parameters),
                        'parameters': sum(parameter.numel() for _, parameter in parameters),
                        'parameter_names': [name for name, _ in parameters]})
    return groups, summary


def main():
    total_started = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation', default='student')
    p.add_argument('--config', type=Path, default=ROOT/'configs/baseline.json')
    p.add_argument('--run-dir', type=Path, default=ROOT/'runs/baseline-s17')
    p.add_argument('--device', default='cpu')
    p.add_argument('--precision', choices=['auto','fp32','bf16'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--learning-rate', type=float, default=.001)
    p.add_argument('--warmup-steps', type=int, default=100)
    p.add_argument('--min-lr-ratio', type=float, default=.1)
    p.add_argument('--weight-decay', type=float, default=.1)
    p.add_argument('--weight-decay-policy', choices=['all','matrix'], default='all',
                   help='all: original recipe; matrix: decay matrices, exempt biases and norm scales.')
    p.add_argument('--adam-beta2', type=float, default=.999)
    p.add_argument('--sampling', choices=['random','epoch'], default='random',
                   help='Random crops, or shuffled non-overlapping windows each epoch.')
    p.add_argument('--keep-best', action='store_true',
                   help='Restore the best intermediate validation checkpoint after training.')
    p.add_argument('--ema-decay', type=float, default=0.0,
                   help='EMA decay in (0, 1); 0 disables EMA.')
    p.add_argument('--ema-start-step', type=int, default=None,
                   help='Initialize EMA after this update; default is halfway through training.')
    p.add_argument('--eval-every', type=int, default=0,
                   help='Optional validation-curve interval; 0 evaluates only after training.')
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        p.error('Batch size and step count must be positive.')
    if args.learning_rate <= 0 or args.warmup_steps < 1 or args.weight_decay < 0:
        p.error('Learning rate and warmup must be positive; weight decay cannot be negative.')
    if not 0 <= args.min_lr_ratio <= 1:
        p.error('Minimum learning-rate ratio must be between 0 and 1.')
    if not 0 < args.adam_beta2 < 1:
        p.error('Adam beta2 must be between 0 and 1.')
    if not 0 <= args.ema_decay < 1:
        p.error('EMA decay must be in [0, 1).')
    if args.ema_start_step is None:
        args.ema_start_step = max(1, args.steps // 2)
    if args.ema_decay > 0 and not 1 <= args.ema_start_step < args.steps:
        p.error('EMA start step must be at least 1 and smaller than total steps.')
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        p.error('Run directory already contains results. Use a new --run-dir.')
    device, precision = setup(args.device, args.precision, args.threads)
    torch.manual_seed(args.seed)
    prepared = time.perf_counter()
    data = load_development_data()
    config = json.loads(args.config.read_text())
    model, implementation_sha = make_model(args.implementation, config, device)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    parameter_groups, parameter_group_summary = make_parameter_groups(
        model, args.weight_decay, args.weight_decay_policy)
    optimizer = torch.optim.AdamW(parameter_groups, lr=args.learning_rate,
                                  betas=(.9, args.adam_beta2),
                                  weight_decay=args.weight_decay)
    print(json.dumps({'weight_decay_policy':args.weight_decay_policy,
                      'parameter_groups':[{key:value for key,value in group.items()
                                           if key != 'parameter_names'}
                                          for group in parameter_group_summary]}),flush=True)
    tokens = data['train'][0].to(device)
    rng = torch.Generator().manual_seed(args.seed)
    epoch_schedule = None
    if args.sampling == 'epoch':
        window_starts = torch.arange(0, len(tokens)-256, 256)
        needed = args.steps * args.batch_size
        permutations = []
        while sum(part.numel() for part in permutations) < needed:
            permutations.append(torch.randperm(len(window_starts), generator=rng))
        epoch_schedule = window_starts[torch.cat(permutations)[:needed]]
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    preparation_seconds = time.perf_counter()-prepared
    started = time.perf_counter()
    history = []
    validation_history = []
    best_validation_bpb = float('inf')
    best_model_state = None
    selected_step = args.steps
    selected_source = 'raw'
    ema = None
    best_by_source = {}
    intermediate_validation_seconds = 0.
    for step in range(args.steps):
        if epoch_schedule is None:
            starts = torch.randint(len(tokens)-257, (args.batch_size,), generator=rng).to(device)
        else:
            begin = step * args.batch_size
            starts = epoch_schedule[begin:begin+args.batch_size].to(device)
        batch = tokens[starts[:,None]+torch.arange(257,device=device)]
        learning_rate = args.learning_rate * min(1., (step+1)/args.warmup_steps) * (
            args.min_lr_ratio + (1-args.min_lr_ratio)*.5*(1+math.cos(math.pi*step/args.steps))
        )
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, precision):
            loss = (model.training_loss(batch) if hasattr(model, 'training_loss') else
                    F.cross_entropy(model(batch[:,:-1]).flatten(0,1).float(),batch[:,1:].flatten()))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if args.ema_decay > 0:
            if step + 1 == args.ema_start_step:
                ema = ModelEMA(model, args.ema_decay)
            elif ema is not None:
                ema.update(model)
        if (step+1)%100 == 0 or step+1 == args.steps:
            row = {'step':step+1,'loss':loss.item(),'seconds':time.perf_counter()-started-intermediate_validation_seconds}
            history.append(row)
            print(json.dumps(row),flush=True)
        scheduled_validation = args.eval_every > 0 and (step+1)%args.eval_every == 0
        final_ema_comparison = args.ema_decay > 0 and step+1 == args.steps
        if scheduled_validation or final_ema_comparison:
            validation_started = time.perf_counter()
            candidates = [('raw', model)]
            if ema is not None and ema.updates > 0:
                candidates.append(('ema', ema.model))
            # Without --keep-best, compare only the candidates at the last step.
            if not args.keep_best and final_ema_comparison:
                best_validation_bpb = float('inf')
            for source, candidate in candidates:
                intermediate = score(candidate,*data['validation'],device,'fp32')
                intermediate.pop('window_nll_nats')
                validation_history.append({'step':step+1,'source':source,**intermediate})
                print(json.dumps({'validation':validation_history[-1]}),flush=True)
                if intermediate['bpb'] < best_by_source.get(source, {}).get('bpb', float('inf')):
                    best_by_source[source] = {'step':step+1,'bpb':intermediate['bpb']}
                if (args.keep_best or final_ema_comparison) and intermediate['bpb'] < best_validation_bpb:
                    best_validation_bpb = intermediate['bpb']
                    best_model_state = {name: value.detach().cpu().clone()
                                        for name, value in candidate.state_dict().items()}
                    selected_step = step + 1
                    selected_source = source
            intermediate_validation_seconds += time.perf_counter()-validation_started
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter()-started-intermediate_validation_seconds
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    validation = score(model,*data['validation'],device,'fp32')
    validation.pop('window_nll_nats')
    checkpoint = args.run_dir/'checkpoint.pt'
    checkpoint_config = dict(config)
    checkpoint_state = model.cpu().state_dict()
    if int(config.get('mtp_heads', 0)) > 0:
        checkpoint_state = {name:value for name,value in checkpoint_state.items()
                            if not name.startswith('mtp_heads.')}
        checkpoint_config['mtp_heads'] = 0
        checkpoint_config['mtp_weight'] = 0.0
    torch.save({'protocol':PROTOCOL,'implementation':args.implementation,'config':checkpoint_config,
                'model':checkpoint_state,'seed':args.seed,
                'train_tokens':args.steps*args.batch_size*256,
                'selected_step':selected_step,'selected_source':selected_source,
                'weight_decay_policy':args.weight_decay_policy,
                'ema':{'decay':args.ema_decay,'start_step':args.ema_start_step,
                       'updates':ema.updates if ema is not None else 0}},checkpoint)
    result = {'protocol':PROTOCOL,'implementation':args.implementation,'config':config,'seed':args.seed,
              'parameters':sum(p.numel() for p in model.parameters()),'precision':precision,
              'train_tokens':args.steps*args.batch_size*256,'preparation_seconds':preparation_seconds,
              'selected_step':selected_step,
              'selected_source':selected_source,'best_validation_by_source':best_by_source,
              'ema':{'decay':args.ema_decay,'start_step':args.ema_start_step,
                     'updates':ema.updates if ema is not None else 0},
              'sampling':args.sampling,'checkpoint_config':checkpoint_config,
              'optimizer':{'name':'AdamW','learning_rate':args.learning_rate,
                           'betas':[.9,args.adam_beta2],'weight_decay':args.weight_decay,
                           'weight_decay_policy':args.weight_decay_policy,
                           'parameter_groups':parameter_group_summary,
                           'warmup_steps':args.warmup_steps,'min_lr_ratio':args.min_lr_ratio},
              'train_seconds':train_seconds,'validation':validation,'history':history,
              'validation_history':validation_history,
              'intermediate_validation_seconds':intermediate_validation_seconds,
              'process_seconds':time.perf_counter()-total_started,
              'torch_version':str(torch.__version__),'threads':args.threads,
              'checkpoint_sha256':sha(checkpoint),'implementation_sha256':implementation_sha,
              'trainer_sha256':sha(Path(__file__)),
              'ema_implementation_sha256':sha(ROOT/'ema.py'),
              **device_metrics(device)}
    (args.run_dir/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result|{'history':[]},indent=2),flush=True)


if __name__ == '__main__':
    main()
