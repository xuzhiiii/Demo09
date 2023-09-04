import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss,KLDivLoss
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from models.model_distill import ALBEF, DistillALBEF
from models.tokenization_bert import BertTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer

def get_kd_loss(teacher_output, student_output, device, is_atten=False, is_hidden=False):
    # attention_loss
    distill_per_num = len(teacher_output[0]) // len(student_output[0])
    mse_loss = MSELoss()
    if is_atten:
        atten_loss = 0.
        for idx, (stu_att, teac_att) in enumerate(zip(student_output, teacher_output)):
            new_teac_att = [teac_att[i * distill_per_num + distill_per_num - 1] for i in range(len(stu_att))]
            for s_att, t_att in zip(stu_att, new_teac_att):
                s_att = torch.where(s_att <= -1e2, torch.zeros_like(s_att).to(device), s_att)
                t_att = torch.where(t_att <= -1e2, torch.zeros_like(t_att).to(device), t_att)
                atten_loss += mse_loss(s_att, t_att)
        return atten_loss

    # hidden_loss
    if is_hidden:
        hidden_loss = 0.
        for idx, (stu_hidd, teac_hidd) in enumerate(zip(student_output, teacher_output)):
            new_teac_hidd = [teac_hidd[i * distill_per_num] for i in range(len(stu_hidd))]
            for s_hidd, t_hidd in zip(stu_hidd, new_teac_hidd):
                hidden_loss += mse_loss(s_hidd, t_hidd)
        return hidden_loss


def soft_cross_entropy(predicts, targets):
    kl_loss = KLDivLoss(reduction='batchmean')
    student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
    targets_prob = torch.nn.functional.softmax(targets, dim=-1)

    return kl_loss(student_likelihood.view(-1,predicts.shape[-1]),targets_prob.view(-1,targets.shape[-1]))


def train(teacher_model, student_model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config):
    # train
    student_model.train()

    output_bool = config["is_kd_distill"]
    loss_emd_weight = 1.0           # todo
    loss_logits_weight = 1.0        # todo
    loss_hidden_weight = config["loss_hidden_weight"]
    loss_atten_weight = config["loss_atten_weight"]
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    if config["is_kd_distill"]:
        metric_logger.add_meter('loss_att', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_hid', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_logits', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if config["add_emd"]:
        metric_logger.add_meter('loss_emd', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps*step_size  
    
    for i,(image, text, idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)
        text_input = tokenizer(text, padding='longest', max_length=30, return_tensors="pt").to(device)
            
        if epoch>0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader))

        # teacher model
        with torch.no_grad():
            teacher_loss = teacher_model(image, text_input, idx, output_attentions=output_bool, output_hidden_states=output_bool)

        student_loss = student_model(image, text_input, alpha, idx, output_attentions=output_bool, output_hidden_states=output_bool)

        # hidden
        loss_hidd, loss_atten = 0.,0.
        teacher_hidden = [
            teacher_loss["visual_hiddens"], teacher_loss["text_output.hidden_states"],
            teacher_loss["output_pos.hidden_states"], teacher_loss["output_neg.hidden_states"]
        ]
        teacher_atten = [
            teacher_loss["visual_attens"], teacher_loss["text_output.attentions"],
            teacher_loss["output_pos.attentions"], teacher_loss["output_neg.attentions"]
        ]
        student_hidden = [
            student_loss["visual_hiddens"], student_loss["text_output.hidden_states"],
            student_loss["output_pos.hidden_states"], student_loss["output_neg.hidden_states"]
        ]
        student_atten = [
            student_loss["visual_attens"], student_loss["text_output.attentions"],
            student_loss["output_pos.attentions"], student_loss["output_neg.attentions"]
        ]
        for t_hideen, s_hidden in zip(teacher_hidden, student_hidden):
            loss_hidd += get_kd_loss(t_hideen, s_hidden, device, is_hidden=True)
        for t_atten, s_atten in zip(teacher_atten, student_atten):
            loss_atten += get_kd_loss(t_atten, s_atten, device, is_atten=True)

        ## itm_logits
        teacher_logits = teacher_loss["loss_logits"]
        student_logits = student_loss["loss_logits"]
        temperature = 1.0                                     # todo
        loss_logits = soft_cross_entropy(student_logits / temperature,
                                             teacher_logits / temperature)
        # all loss
        loss_kd = loss_logits * loss_logits_weight + loss_hidd * loss_hidden_weight + loss_atten * loss_atten_weight
        loss = student_loss["loss_ita"] + student_loss["loss_itm"] + loss_kd
        if config["add_emd"]:
            loss += student_loss["loss_emd"] * loss_emd_weight

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        metric_logger.update(loss_itm= student_loss["loss_itm"].item())
        metric_logger.update(loss_ita= student_loss["loss_ita"].item())

        if config["is_kd_distill"]:
            metric_logger.update(loss_att = loss_atten * loss_atten_weight)
            metric_logger.update(loss_hid = loss_hidd * loss_hidden_weight)
            metric_logger.update(loss_logits = loss_logits * loss_logits_weight)
        if config["add_emd"]:
            metric_logger.update(loss_emd = student_loss["loss_emd"] * loss_emd_weight)

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)         

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config):
    # test
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'    
    
    print('Computing features for evaluation...')
    start_time = time.time()  

    texts = data_loader.dataset.text   
    num_text = len(texts)
    text_bs = 256
    text_feats = []
    text_embeds = []  
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i+text_bs)]
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(device) 
        text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
        text_feat = text_output.last_hidden_state
        text_embed = F.normalize(model.text_proj(text_feat[:,0,:]))
        text_embeds.append(text_embed)   
        text_feats.append(text_feat)
        text_atts.append(text_input.attention_mask)
    text_embeds = torch.cat(text_embeds,dim=0)
    text_feats = torch.cat(text_feats,dim=0)
    text_atts = torch.cat(text_atts,dim=0)
    
    image_feats = []
    image_embeds = []
    for image, img_id in data_loader: 
        image = image.to(device) 
        image_feat = model.visual_encoder(image)        
        image_embed = model.vision_proj(image_feat[:,0,:])            
        image_embed = F.normalize(image_embed,dim=-1)      
        
        image_feats.append(image_feat)
        image_embeds.append(image_embed)
     
    image_feats = torch.cat(image_feats,dim=0)
    image_embeds = torch.cat(image_embeds,dim=0)
    
    sims_matrix = image_embeds @ text_embeds.t()
    score_matrix_i2t = torch.full((len(data_loader.dataset.image),len(texts)),-100.0).to(device)
    
    num_tasks = utils.get_world_size()
    rank = utils.get_rank() 
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)

    for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)): 
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)

        encoder_output = image_feats[start+i].repeat(config['k_test'],1,1)
        encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device)
        output = model.text_encoder(encoder_embeds = text_feats[topk_idx], 
                                    attention_mask = text_atts[topk_idx],
                                    encoder_hidden_states = encoder_output,
                                    encoder_attention_mask = encoder_att,                             
                                    return_dict = True,
                                    mode = 'fusion'
                                   )
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_i2t[start+i,topk_idx] = score
        
    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((len(texts),len(data_loader.dataset.image)),-100.0).to(device)
    
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)    
    
    for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)): 
        
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_feats[topk_idx]
        encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device)
        output = model.text_encoder(encoder_embeds = text_feats[start+i].repeat(config['k_test'],1,1), 
                                    attention_mask = text_atts[start+i].repeat(config['k_test'],1),
                                    encoder_hidden_states = encoder_output,
                                    encoder_attention_mask = encoder_att,                             
                                    return_dict = True,
                                    mode = 'fusion'
                                   )
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_t2i[start+i,topk_idx] = score

    if args.distributed:
        dist.barrier()   
        torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM) 
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)        
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str)) 

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


            
@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    
    #Images->Text 
    ranks = np.zeros(scores_i2t.shape[0])
    for index,score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
  
    #Text->Images 
    ranks = np.zeros(scores_t2i.shape[0])
    
    for index,score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)        

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result =  {'txt_r1': tr1,
                    'txt_r5': tr5,
                    'txt_r10': tr10,
                    'txt_r_mean': tr_mean,
                    'img_r1': ir1,
                    'img_r5': ir5,
                    'img_r10': ir10,
                    'img_r_mean': ir_mean,
                    'r_mean': r_mean}
    return eval_result




def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating retrieval dataset")
    if "test_xmai_file" in config:
        train_dataset, val_dataset, test_dataset, test_xmai_dataset = create_dataset('re', config)
    else:
        train_dataset, val_dataset, test_dataset = create_dataset('re', config)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if "test_xmai_file" in config:
            samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None, None]
        else:
            samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None, None] if "test_xmai_file" not in config else [None, None, None, None]

    if "test_xmai_file" in config:
        train_loader, val_loader, test_loader, test_xmai_loader = create_loader([train_dataset, val_dataset, test_dataset, test_xmai_dataset],
                                                            samplers, batch_size=[config['batch_size_train']] + [ config['batch_size_test']] * 3,
                                                              num_workers=[4, 4, 4, 4],
                                                              is_trains=[True, False, False, False],
                                                              collate_fns=[None, None, None, None])
    else:
        train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                          batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2,
                                                          num_workers=[4,4,4],
                                                          is_trains=[True, False, False], 
                                                          collate_fns=[None,None,None])

    tokenizer = BertTokenizer.from_pretrained(config["text_encoder"])

    #### LOAD Teacher Model ####
    print("Loading teacher model")
    teacher_model = ALBEF(config=config, text_encoder=config["text_encoder"], tokenizer=tokenizer)
    teacher_model.init_params(args.teacher_ckpt)

    for name, param in teacher_model.named_parameters():
        param.requires_grad = False
    teacher_model = teacher_model.to(device)

    # load and initial student model todo
    student_model = DistillALBEF(config=config, text_encoder=config["text_encoder"], tokenizer=tokenizer,
                              num_heads=config["distill_layer_num"])
    if args.evaluate:
        checkpoint = torch.load(args.student_ckpt, map_location='cpu')
        state_dict = checkpoint["model"]
        student_model.load_state_dict(state_dict, strict=False)
    else:
        student_model.init_params(args.student_ckpt, ckpt_layer_num=12)

    student_model.to(device)

    s_model_without_ddp = student_model
    if args.distributed:
        student_model = torch.nn.parallel.DistributedDataParallel(student_model, device_ids=[args.gpu],
                                                                  find_unused_parameters=True)  #
        s_model_without_ddp = student_model.module

    print("### Total student_model Params: ", sum(p.numel() for p in student_model.parameters() if p.requires_grad))
    print("### Total teacher_model Params: ", sum(p.numel() for p in teacher_model.parameters() if p.requires_grad))

    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, student_model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)  
    
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    best = 0
    best_epoch = 0

    print("Start training")
    start_time = time.time()    
    for epoch in range(0, max_epoch):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = train(teacher_model, student_model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config)
            
        score_val_i2t, score_val_t2i, = evaluation(s_model_without_ddp, val_loader, tokenizer, device, config)
        score_test_i2t, score_test_t2i = evaluation(s_model_without_ddp, test_loader, tokenizer, device, config)
        if "test_xmai_file" in config:
            score_test_xmai_i2t, score_test_xmai_t2i = evaluation(s_model_without_ddp, test_xmai_loader, tokenizer, device, config)
    
        if utils.is_main_process():  
      
            val_result = itm_eval(score_val_i2t, score_val_t2i, val_loader.dataset.txt2img, val_loader.dataset.img2txt)
            print(val_result)
            test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt)
            print(test_result)
            if "test_xmai_file" in config:
                test_xmai_result = itm_eval(score_test_xmai_i2t, score_test_xmai_t2i, test_loader.dataset.txt2img,
                                   test_loader.dataset.img2txt)
                print(test_xmai_result)

            if args.evaluate:                
                log_stats = {**{f'val_{k}': v for k, v in val_result.items()},
                             **{f'test_{k}': v for k, v in test_result.items()},
                             'epoch': epoch,
                            }
                if "test_xmai_file" in config:
                    log_stats.update( **{f'test_xmai_{k}': v for k, v in test_xmai_result.items()} )
                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")     
            else:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in val_result.items()},
                             **{f'test_{k}': v for k, v in test_result.items()},
                             'epoch': epoch,
                            }
                if "test_xmai_file" in config:
                    log_stats.update( **{f'test_xmai_{k}': v for k, v in test_xmai_result.items()} )
                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")   
                    
                if val_result['r_mean']>best:
                    save_obj = {
                        'model': s_model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'config': config,
                        'epoch': epoch,
                    }
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))  
                    best = val_result['r_mean']    
                    best_epoch = epoch
                    
        if args.evaluate: 
            break
           
        lr_scheduler.step(epoch+warmup_steps+1)  
        dist.barrier()     
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 

    if utils.is_main_process():   
        with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
            f.write("best epoch: %d"%best_epoch)               

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--config', default='./configs/Distill.yaml')
    parser.add_argument('--output_dir', default='output/Distill_flickr/2L_kd_lr1e-4')
    parser.add_argument('--teacher_ckpt', default='/home/facelesswei/0-DC/pretrained/albef/flickr30k.pth')
    parser.add_argument('--student_ckpt', default='/home/facelesswei/0-DC/pretrained/albef/flickr30k.pth')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=2, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)
