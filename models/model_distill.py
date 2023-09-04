from functools import partial
from models.vit import VisionTransformer
from models.xbert import BertConfig, BertModel

import torch
from torch import nn
import torch.nn.functional as F

import json
import re

import utils

layer_num_pattern = re.compile(r'\.[0-9]+\.')  #   .10.  ->  .5.

def load_pretrained(ckpt_rpath, ckpt_layer_num=None, layer_num=None):

	checkpoint = torch.load(ckpt_rpath, map_location='cpu')
	state_dict = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint

	# student layer init, (12 Layer -> 6/2 Layer)
	if ckpt_layer_num and layer_num and ckpt_layer_num != layer_num:
		layer_step = int(ckpt_layer_num//layer_num)
		dict_map = {"."+str(i * layer_step)+".": "."+str(i)+"." for i in range(layer_num)}

		for k in list(state_dict.keys()):
			res = layer_num_pattern.search(k)
			if res and res.group() in dict_map:
				new_k = k.replace(res.group(), dict_map[res.group()])
				state_dict[new_k] = state_dict[k]
				del state_dict[k]
	return state_dict


class ALBEF(nn.Module):
	def __init__(self,
	             text_encoder=None,
	             tokenizer=None,
	             config=None,
	             ):
		super().__init__()

		self.tokenizer = tokenizer
		embed_dim = config['embed_dim']
		vision_width = config['vision_width']
		self.visual_encoder = VisionTransformer(
			img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12,
			mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

		bert_config = BertConfig.from_json_file(config['bert_config'])
		self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)

		text_width = self.text_encoder.config.hidden_size
		self.vision_proj = nn.Linear(vision_width, embed_dim)
		self.text_proj = nn.Linear(text_width, embed_dim)

		self.itm_head = nn.Linear(text_width, 2)

	def init_params(self, ckpt_rpath):
		state_dict = load_pretrained(ckpt_rpath)
		msg = self.load_state_dict(state_dict, strict=False)

		print('load checkpoint from %s' % ckpt_rpath)
		print("missing_keys: ", msg.missing_keys)
		print("unexpected_keys: ", msg.unexpected_keys)

	def forward(self, image, text, idx , output_attentions=True, output_hidden_states=True):

		visual_attens, visual_hiddens = self.visual_encoder(image, need_intermediate=True)
		image_embeds = visual_hiddens[-1]
		image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

		image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)

		text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask,
		                                return_dict=True, mode='text', output_hidden_states=output_hidden_states, output_attentions=output_attentions)
		text_embeds = text_output.last_hidden_state
		text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)

		idx = idx.view(-1, 1)
		idx_all = idx.t()
		pos_idx = torch.eq(idx, idx_all).float()
		sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

		sim_i2t = image_feat @ text_feat.t()
		sim_t2i = text_feat @ image_feat.t()

		loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_targets, dim=1).mean()
		loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_targets, dim=1).mean()

		loss_ita = (loss_i2t + loss_t2i) / 2

		###=================================###
		# forward the positve image-text pair
		output_pos = self.text_encoder(encoder_embeds=text_embeds,
		                               attention_mask=text.attention_mask,
		                               encoder_hidden_states=image_embeds,
		                               encoder_attention_mask=image_atts,
		                               return_dict=True,
		                               mode='fusion',
		                               output_hidden_states=output_hidden_states,
		                               output_attentions=output_attentions
		                               )
		with torch.no_grad():
			bs = image.size(0)
			weights_i2t = F.softmax(sim_i2t[:, :bs] + 1e-4, dim=1)
			weights_t2i = F.softmax(sim_t2i[:, :bs] + 1e-4, dim=1)

			mask = torch.eq(idx, idx.T)
			weights_i2t.masked_fill_(mask, 0)
			weights_t2i.masked_fill_(mask, 0)

		# select a negative image for each text
		image_embeds_neg = []
		for b in range(bs):
			neg_idx = torch.multinomial(weights_t2i[b], 1).item()
			image_embeds_neg.append(image_embeds[neg_idx])
		image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

		# select a negative text for each image
		text_embeds_neg = []
		text_atts_neg = []
		for b in range(bs):
			neg_idx = torch.multinomial(weights_i2t[b], 1).item()
			text_embeds_neg.append(text_embeds[neg_idx])
			text_atts_neg.append(text.attention_mask[neg_idx])
		text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
		text_atts_neg = torch.stack(text_atts_neg, dim=0)

		text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
		text_atts_all = torch.cat([text.attention_mask, text_atts_neg], dim=0)

		image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
		image_atts_all = torch.cat([image_atts, image_atts], dim=0)

		output_neg = self.text_encoder(encoder_embeds=text_embeds_all,
		                               attention_mask=text_atts_all,
		                               encoder_hidden_states=image_embeds_all,
		                               encoder_attention_mask=image_atts_all,
		                               return_dict=True,
		                               mode='fusion',
		                               )

		vl_embeddings = torch.cat([output_pos.last_hidden_state[:, 0, :], output_neg.last_hidden_state[:, 0, :]], dim=0)
		vl_output = self.itm_head(vl_embeddings)

		itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
		                       dim=0).to(image.device)
		loss_itm = F.cross_entropy(vl_output, itm_labels)

		res = {}
		res["loss_ita"] = loss_ita
		res["loss_itm"] = loss_itm
		res["loss_logits"] = vl_output

		if output_hidden_states:
			res.update( { 'visual_attens': visual_attens,'visual_hiddens': visual_hiddens,
			       'text_output.attentions': text_output.attentions, 'text_output.hidden_states': text_output.hidden_states,
			       'output_pos.attentions':output_pos.attentions, 'output_pos.hidden_states': output_pos.hidden_states,
			       'output_neg.attentions':output_pos.attentions, 'output_neg.hidden_states': output_pos.hidden_states} )

		return res


class DistillALBEF(nn.Module):
	def __init__(self,
	             text_encoder=None,
	             tokenizer=None,
	             config=None,
	             num_heads = 6
	             ):
		super().__init__()

		self.tokenizer = tokenizer
		embed_dim = config['embed_dim']
		vision_width = config['vision_width']
		vision_layer = num_heads
		self.visual_encoder = VisionTransformer(
			img_size=config['image_res'], patch_size=16, embed_dim=768, depth=vision_layer, num_heads=12,
			mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
			add_perturbed=config["perturbed"], eps= config["eps"], is_uniformity=config["is_uniformity"])

		self.layer_num = num_heads
		bert_config_dict = json.load(open(config['bert_config']))
		bert_config_dict["num_hidden_layers"] = vision_layer
		bert_config_dict["fusion_layer"] = vision_layer // 2
		distill_bert_config = BertConfig.from_dict(bert_config_dict)
		if text_encoder:
			self.text_encoder = BertModel.from_pretrained(text_encoder, config=distill_bert_config, add_pooling_layer=False,
					add_perturbed=config["perturbed"], eps= config["eps"], is_uniformity=config["is_uniformity"])

		text_width = self.text_encoder.config.hidden_size
		self.vision_proj = nn.Linear(vision_width, embed_dim)
		self.text_proj = nn.Linear(text_width, embed_dim)

		self.temp = nn.Parameter(torch.ones([]) * config['temp'])  # for contrastive learning loss_ita

		self.itm_head = nn.Linear(text_width, 2)

		# create momentum models
		self.momentum_distill = config['momentum_distill']
		if self.momentum_distill:
			self.momentum = config['momentum']
			self.queue_size = config['queue_size']

			self.visual_encoder_m = VisionTransformer(
				img_size=config['image_res'], patch_size=16, embed_dim=768, depth=vision_layer, num_heads=12,
				mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
			self.vision_proj_m = nn.Linear(vision_width, embed_dim)
			self.text_encoder_m = BertModel.from_pretrained(text_encoder, config=distill_bert_config, add_pooling_layer=False)
			self.text_proj_m = nn.Linear(text_width, embed_dim)

			self.model_pairs = [[self.visual_encoder, self.visual_encoder_m],
			                    [self.vision_proj, self.vision_proj_m],
			                    [self.text_encoder, self.text_encoder_m],
			                    [self.text_proj, self.text_proj_m],
			                    ]
			self.copy_params()

			# create the queue
			self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
			self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
			self.register_buffer("idx_queue", torch.full((1, self.queue_size), -100))
			self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

			self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
			self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

		# Robust Distill
		self.emd = {"solver": config["solver"], "metric": config["metric"]} if config["add_emd"] else None
		self.temperature = config["temperature"]

	def init_params(self, ckpt_rpath, ckpt_layer_num):

		state_dict = load_pretrained(ckpt_rpath, ckpt_layer_num, self.layer_num)
		msg = self.load_state_dict(state_dict, strict=False)
		if self.momentum_distill:
			self.copy_params()

		print('load checkpoint from %s' % ckpt_rpath)
		print("missing_keys: ", msg.missing_keys)
		print("unexpected_keys: ", msg.unexpected_keys)

	def forward(self, image, text, alpha, idx, output_attentions=True,output_hidden_states=True):

		visual_attens, visual_hiddens = self.visual_encoder(image, need_intermediate=True)
		image_embeds = visual_hiddens[-1]
		image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

		image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)

		text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask,
		                                return_dict=True, mode='text',
		                               output_hidden_states = output_hidden_states, output_attentions = output_attentions )
		text_embeds = text_output.last_hidden_state
		text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)

		idx = idx.view(-1, 1)
		if self.momentum_distill:
			idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()], dim=1)
			pos_idx = torch.eq(idx, idx_all).float()
			sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)
		else:
			pos_idx = torch.eq(idx, idx.t()).float()
			sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

		with torch.no_grad():
			if self.momentum_distill:
				self._momentum_update()
				image_embeds_m = self.visual_encoder_m(image)
				image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)
				image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)
				text_output_m = self.text_encoder_m(text.input_ids, attention_mask=text.attention_mask,
				                                    return_dict=True, mode='text')
				text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1)
				text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)

				sim_i2t_m = image_feat_m @ text_feat_all / self.temp
				sim_t2i_m = text_feat_m @ image_feat_all / self.temp

				sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
				sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets
			else:
				image_feat_all = image_feat.t()
				text_feat_all = text_feat.t()

		sim_i2t = image_feat @ text_feat_all / self.temp
		sim_t2i = text_feat @ image_feat_all / self.temp

		if self.momentum_distill:
			loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
			loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()

			self._dequeue_and_enqueue(image_feat_m, text_feat_m, idx)

		else:
			loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_targets, dim=1).mean()
			loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_targets, dim=1).mean()

		loss_ita = (loss_i2t + loss_t2i) / 2

		# EMD
		if self.emd:
			pos_idx = torch.eq(idx, idx.t()).float()
			sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

			emd_logits = self.calcu_emd( image_embeds, text_embeds, self.emd["solver"], self.emd["metric"])
			loss_emd = -torch.sum(F.log_softmax(emd_logits, dim=1) * sim_targets, dim=1).mean()

		###=================================###
		# forward the positve image-text pair
		output_pos = self.text_encoder(encoder_embeds=text_embeds,
		                               attention_mask=text.attention_mask,
		                               encoder_hidden_states=image_embeds,
		                               encoder_attention_mask=image_atts,
		                               return_dict=True,
		                               mode='fusion',
		                               output_hidden_states = output_hidden_states, output_attentions = output_attentions
		                               )
		with torch.no_grad():
			bs = image.size(0)
			weights_i2t = F.softmax(sim_i2t[:, :bs] + 1e-4, dim=1)
			weights_t2i = F.softmax(sim_t2i[:, :bs] + 1e-4, dim=1)

			mask = torch.eq(idx, idx.T)
			weights_i2t.masked_fill_(mask, 0)
			weights_t2i.masked_fill_(mask, 0)

		# select a negative image for each text
		image_embeds_neg = []
		for b in range(bs):
			neg_idx = torch.multinomial(weights_t2i[b], 1).item()
			image_embeds_neg.append(image_embeds[neg_idx])
		image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

		# select a negative text for each image
		text_embeds_neg = []
		text_atts_neg = []
		for b in range(bs):
			neg_idx = torch.multinomial(weights_i2t[b], 1).item()
			text_embeds_neg.append(text_embeds[neg_idx])
			text_atts_neg.append(text.attention_mask[neg_idx])
		text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
		text_atts_neg = torch.stack(text_atts_neg, dim=0)

		text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
		text_atts_all = torch.cat([text.attention_mask, text_atts_neg], dim=0)

		image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
		image_atts_all = torch.cat([image_atts, image_atts], dim=0)

		output_neg = self.text_encoder(encoder_embeds=text_embeds_all,
		                               attention_mask=text_atts_all,
		                               encoder_hidden_states=image_embeds_all,
		                               encoder_attention_mask=image_atts_all,
		                               return_dict=True,
		                               mode='fusion',
		                               output_hidden_states = output_hidden_states, output_attentions = output_attentions
		                               )

		vl_embeddings = torch.cat([output_pos.last_hidden_state[:, 0, :], output_neg.last_hidden_state[:, 0, :]], dim=0)
		vl_output = self.itm_head(vl_embeddings)

		itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
		                       dim=0).to(image.device)
		loss_itm = F.cross_entropy(vl_output, itm_labels)

		res = {}
		res["loss_ita"] = loss_ita
		res["loss_itm"] = loss_itm
		res["loss_logits"] = vl_output

		if self.emd:
			res["loss_emd"] = loss_emd

		if output_hidden_states:
			res.update( {'visual_attens': visual_attens, 'visual_hiddens': visual_hiddens,
			       'text_output.attentions': text_output.attentions, 'text_output.hidden_states': text_output.hidden_states,
			       'output_pos.attentions': output_pos.attentions, 'output_pos.hidden_states': output_pos.hidden_states,
			       'output_neg.attentions': output_pos.attentions, 'output_neg.hidden_states': output_pos.hidden_states} )
		return res


	def calcu_emd(self, image_embeds, text_embeds, solver, metric):
		# weight
		query, proto = image_embeds,text_embeds
		weight_1 = self.get_weight_vector(query, proto)
		weight_2 = self.get_weight_vector(proto, query)
		proto = self.normalize_feature(proto)
		query = self.normalize_feature(query)

		# similarity map
		similarity_map = self.get_similiarity_map(proto, query, metric)

		# EMD results
		if solver == 'opencv' or (not self.training):
			logits = self.get_emd_distance(similarity_map, weight_1, weight_2, solver='opencv')
		else:
			logits = self.get_emd_distance(similarity_map, weight_1, weight_2, solver='qpth')
		return logits


	def get_weight_vector(self, A, B):

		M = A.shape[0]
		N = B.shape[0]

		B = F.adaptive_avg_pool2d(B, [1, 1])
		B = B.repeat(1, A.shape[1], A.shape[2])

		A = A.unsqueeze(1)
		B = B.unsqueeze(0)

		A = A.repeat(1, N, 1, 1)
		B = B.repeat(M, 1, 1, 1)

		combination = (A * B).sum(3)
		combination = combination.view(M, N, -1)
		combination = F.relu(combination) + 1e-3
		return combination


	def get_similiarity_map(self, proto, query, metric="cosine"):

		way = proto.shape[0]
		num_query = query.shape[0]
		# query = query.view(query.shape[0], query.shape[1], -1)
		# proto = proto.view(proto.shape[0], proto.shape[1], -1)

		proto = proto.unsqueeze(0).repeat([num_query, 1, 1, 1])
		query = query.unsqueeze(1).repeat([1, way, 1, 1])
		# proto = proto.permute(0, 1, 3, 2)   # -> b * b * dim *L30
		# query = query.permute(0, 1, 3, 2)   # -> b * b * dim *L577
		# proto = proto.sum(-1).unsqueeze(-1)
		# query = query.sum(-1).unsqueeze(-1)

		feature_size = proto.shape[-2]  # 30

		if metric == 'cosine':
			proto = proto.unsqueeze(-3)
			query = query.unsqueeze(-2)
			query = query.repeat(1, 1, 1, feature_size, 1)
			similarity_map = F.cosine_similarity(proto, query, dim=-1)
		if metric == 'l2':
			proto = proto.unsqueeze(-3)
			query = query.unsqueeze(-2)
			query = query.repeat(1, 1, 1, feature_size, 1)
			similarity_map = (proto - query).pow(2).sum(-1)
			similarity_map = 1 - similarity_map
		return similarity_map


	def get_emd_distance(self, similarity_map, weight_1, weight_2, solver="opencv"):
		num_query = similarity_map.shape[0]
		num_proto = similarity_map.shape[1]
		num_node = weight_1.shape[-1]

		if solver == 'opencv':  # use openCV solver

			for i in range(num_query):
				for j in range(num_proto):
					_, flow = utils.emd_inference_opencv(1 - similarity_map[i, j, :, :], weight_1[i, j, :], weight_2[j, i, :])
					similarity_map[i, j, :, :] = (similarity_map[i, j, :, :]) * torch.from_numpy(flow).cuda()

			temperature = (self.temperature / num_node)
			# print("debug similary_map:", similarity_map.shape, similarity_map.sum(-1).sum(-1))
			logitis = similarity_map.sum(-1).sum(-1) * temperature
			return logitis


		elif solver == 'qpth':
			logitis = 0
			pass
			# weight_2 = weight_2.permute(1, 0, 2)
			# similarity_map = similarity_map.view(num_query * num_proto, similarity_map.shape[-2],
			#                                      similarity_map.shape[-1])
			# weight_1 = weight_1.view(num_query * num_proto, weight_1.shape[-1])
			# weight_2 = weight_2.reshape(num_query * num_proto, weight_2.shape[-1])
			#
			# _, flows = utils.emd_inference_qpth(1 - similarity_map, weight_1, weight_2, form=self.args.form,
			#                               l2_strength=self.args.l2_strength)
			#
			# logitis = (flows * similarity_map).view(num_query, num_proto, flows.shape[-2], flows.shape[-1])
			# temperature = (self.args.temperature / num_node)
			# logitis = logitis.sum(-1).sum(-1) * temperature
		else:
			raise ValueError('Unknown Solver')

		return logitis

	def normalize_feature(self, x, norm="center"):
		if norm == 'center':
			x = x - x.mean(1).unsqueeze(1)
			return x
		else:
			return x



	@torch.no_grad()
	def copy_params(self):
		for model_pair in self.model_pairs:
			for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
				param_m.data.copy_(param.data)  # initialize
				param_m.requires_grad = False  # not update by gradient

	@torch.no_grad()
	def _momentum_update(self):
		for model_pair in self.model_pairs:
			for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
				param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

	@torch.no_grad()
	def _dequeue_and_enqueue(self, image_feat, text_feat, idx):
		# gather keys before updating queue
		image_feats = concat_all_gather(image_feat)
		text_feats = concat_all_gather(text_feat)
		idxs = concat_all_gather(idx)

		batch_size = image_feats.shape[0]

		ptr = int(self.queue_ptr)
		assert self.queue_size % batch_size == 0  # for simplicity

		# replace the keys at ptr (dequeue and enqueue)
		# print(ptr,batch_size, self.image_queue.shape, image_feats.shape)
		self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
		self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
		self.idx_queue[:, ptr:ptr + batch_size] = idxs.T
		ptr = (ptr + batch_size) % self.queue_size  # move pointer

		self.queue_ptr[0] = ptr



@torch.no_grad()
def concat_all_gather(tensor):
	"""
	Performs all_gather operation on the provided tensors.
	*** Warning ***: torch.distributed.all_gather has no gradient.
	"""
	tensors_gather = [torch.ones_like(tensor)
	                  for _ in range(torch.distributed.get_world_size())]
	torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

	output = torch.cat(tensors_gather, dim=0)
	return output

