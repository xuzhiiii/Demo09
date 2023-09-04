import numpy as np
import io
import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist

# use for calculate EMD
import cv2
import torch
import torch.nn.functional as F
from qpth.qp import QPFunction


class SmoothedValue(object):
	"""Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

	def __init__(self, window_size=20, fmt=None):
		if fmt is None:
			fmt = "{median:.4f} ({global_avg:.4f})"
		self.deque = deque(maxlen=window_size)
		self.total = 0.0
		self.count = 0
		self.fmt = fmt

	def update(self, value, n=1):
		self.deque.append(value)
		self.count += n
		self.total += value * n

	def synchronize_between_processes(self):
		"""
        Warning: does not synchronize the deque!
        """
		if not is_dist_avail_and_initialized():
			return
		t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
		dist.barrier()
		dist.all_reduce(t)
		t = t.tolist()
		self.count = int(t[0])
		self.total = t[1]

	@property
	def median(self):
		d = torch.tensor(list(self.deque))
		return d.median().item()

	@property
	def avg(self):
		d = torch.tensor(list(self.deque), dtype=torch.float32)
		return d.mean().item()

	@property
	def global_avg(self):
		return self.total / self.count

	@property
	def max(self):
		return max(self.deque)

	@property
	def value(self):
		return self.deque[-1]

	def __str__(self):
		return self.fmt.format(
			median=self.median,
			avg=self.avg,
			global_avg=self.global_avg,
			max=self.max,
			value=self.value)


class MetricLogger(object):
	def __init__(self, delimiter="\t"):
		self.meters = defaultdict(SmoothedValue)
		self.delimiter = delimiter

	def update(self, **kwargs):
		for k, v in kwargs.items():
			if isinstance(v, torch.Tensor):
				v = v.item()
			assert isinstance(v, (float, int))
			self.meters[k].update(v)

	def __getattr__(self, attr):
		if attr in self.meters:
			return self.meters[attr]
		if attr in self.__dict__:
			return self.__dict__[attr]
		raise AttributeError("'{}' object has no attribute '{}'".format(
			type(self).__name__, attr))

	def __str__(self):
		loss_str = []
		for name, meter in self.meters.items():
			loss_str.append(
				"{}: {}".format(name, str(meter))
			)
		return self.delimiter.join(loss_str)

	def global_avg(self):
		loss_str = []
		for name, meter in self.meters.items():
			loss_str.append(
				"{}: {:.4f}".format(name, meter.global_avg)
			)
		return self.delimiter.join(loss_str)

	def synchronize_between_processes(self):
		for meter in self.meters.values():
			meter.synchronize_between_processes()

	def add_meter(self, name, meter):
		self.meters[name] = meter

	def log_every(self, iterable, print_freq, header=None):
		i = 0
		if not header:
			header = ''
		start_time = time.time()
		end = time.time()
		iter_time = SmoothedValue(fmt='{avg:.4f}')
		data_time = SmoothedValue(fmt='{avg:.4f}')
		space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
		log_msg = [
			header,
			'[{0' + space_fmt + '}/{1}]',
			'eta: {eta}',
			'{meters}',
			'time: {time}',
			'data: {data}'
		]
		if torch.cuda.is_available():
			log_msg.append('max mem: {memory:.0f}')
		log_msg = self.delimiter.join(log_msg)
		MB = 1024.0 * 1024.0
		for obj in iterable:
			data_time.update(time.time() - end)
			yield obj
			iter_time.update(time.time() - end)
			if i % print_freq == 0 or i == len(iterable) - 1:
				eta_seconds = iter_time.global_avg * (len(iterable) - i)
				eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
				if torch.cuda.is_available():
					print(log_msg.format(
						i, len(iterable), eta=eta_string,
						meters=str(self),
						time=str(iter_time), data=str(data_time),
						memory=torch.cuda.max_memory_allocated() / MB))
				else:
					print(log_msg.format(
						i, len(iterable), eta=eta_string,
						meters=str(self),
						time=str(iter_time), data=str(data_time)))
			i += 1
			end = time.time()
		total_time = time.time() - start_time
		total_time_str = str(datetime.timedelta(seconds=int(total_time)))
		print('{} Total time: {} ({:.4f} s / it)'.format(
			header, total_time_str, total_time / len(iterable)))


class AttrDict(dict):
	def __init__(self, *args, **kwargs):
		super(AttrDict, self).__init__(*args, **kwargs)
		self.__dict__ = self


def compute_acc(logits, label, reduction='mean'):
	ret = (torch.argmax(logits, dim=1) == label).float()
	if reduction == 'none':
		return ret.detach()
	elif reduction == 'mean':
		return ret.mean().item()


def compute_n_params(model, return_str=True):
	tot = 0
	for p in model.parameters():
		w = 1
		for x in p.shape:
			w *= x
		tot += w
	if return_str:
		if tot >= 1e6:
			return '{:.1f}M'.format(tot / 1e6)
		else:
			return '{:.1f}K'.format(tot / 1e3)
	else:
		return tot


def setup_for_distributed(is_master):
	"""
    This function disables printing when not in master process
    """
	import builtins as __builtin__
	builtin_print = __builtin__.print

	def print(*args, **kwargs):
		force = kwargs.pop('force', False)
		if is_master or force:
			builtin_print(*args, **kwargs)

	__builtin__.print = print


def is_dist_avail_and_initialized():
	if not dist.is_available():
		return False
	if not dist.is_initialized():
		return False
	return True


def get_world_size():
	if not is_dist_avail_and_initialized():
		return 1
	return dist.get_world_size()


def get_rank():
	if not is_dist_avail_and_initialized():
		return 0
	return dist.get_rank()


def is_main_process():
	return get_rank() == 0


def save_on_master(*args, **kwargs):
	if is_main_process():
		torch.save(*args, **kwargs)


def init_distributed_mode(args):
	if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
		args.rank = int(os.environ["RANK"])
		args.world_size = int(os.environ['WORLD_SIZE'])
		args.gpu = int(os.environ['LOCAL_RANK'])
	elif 'SLURM_PROCID' in os.environ:
		args.rank = int(os.environ['SLURM_PROCID'])
		args.gpu = args.rank % torch.cuda.device_count()
	else:
		print('Not using distributed mode')
		args.distributed = False
		return

	args.distributed = True

	torch.cuda.set_device(args.gpu)
	args.dist_backend = 'nccl'
	print('| distributed init (rank {}): {}'.format(
		args.rank, args.dist_url), flush=True)
	torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
	                                     world_size=args.world_size, rank=args.rank)
	torch.distributed.barrier()
	setup_for_distributed(args.rank == 0)


def emd_inference_qpth(distance_matrix, weight1, weight2, form='QP', l2_strength=0.0001):
	"""
    to use the QP solver QPTH to derive EMD (LP problem),
    one can transform the LP problem to QP,
    or omit the QP term by multiplying it with a small value,i.e. l2_strngth.
    :param distance_matrix: nbatch * element_number * element_number
    :param weight1: nbatch  * weight_number
    :param weight2: nbatch  * weight_number
    :return:
    emd distance: nbatch*1
    flow : nbatch * weight_number *weight_number

    """

	weight1 = (weight1 * weight1.shape[-1]) / weight1.sum(1).unsqueeze(1)
	weight2 = (weight2 * weight2.shape[-1]) / weight2.sum(1).unsqueeze(1)

	nbatch = distance_matrix.shape[0]
	nelement_distmatrix = distance_matrix.shape[1] * distance_matrix.shape[2]
	nelement_weight1 = weight1.shape[1]
	nelement_weight2 = weight2.shape[1]

	Q_1 = distance_matrix.view(-1, 1, nelement_distmatrix).double()

	if form == 'QP':
		# version: QTQ
		Q = torch.bmm(Q_1.transpose(2, 1), Q_1).double().cuda() + 1e-4 * torch.eye(
			nelement_distmatrix).double().cuda().unsqueeze(0).repeat(nbatch, 1, 1)  # 0.00001 *
		p = torch.zeros(nbatch, nelement_distmatrix).double().cuda()
	elif form == 'L2':
		# version: regularizer
		Q = (l2_strength * torch.eye(nelement_distmatrix).double()).cuda().unsqueeze(0).repeat(nbatch, 1, 1)
		p = distance_matrix.view(nbatch, nelement_distmatrix).double()
	else:
		raise ValueError('Unkown form')

	h_1 = torch.zeros(nbatch, nelement_distmatrix).double().cuda()
	h_2 = torch.cat([weight1, weight2], 1).double()
	h = torch.cat((h_1, h_2), 1)

	G_1 = -torch.eye(nelement_distmatrix).double().cuda().unsqueeze(0).repeat(nbatch, 1, 1)
	G_2 = torch.zeros([nbatch, nelement_weight1 + nelement_weight2, nelement_distmatrix]).double().cuda()
	# sum_j(xij) = si
	for i in range(nelement_weight1):
		G_2[:, i, nelement_weight2 * i:nelement_weight2 * (i + 1)] = 1
	# sum_i(xij) = dj
	for j in range(nelement_weight2):
		G_2[:, nelement_weight1 + j, j::nelement_weight2] = 1
	# xij>=0, sum_j(xij) <= si,sum_i(xij) <= dj, sum_ij(x_ij) = min(sum(si), sum(dj))
	G = torch.cat((G_1, G_2), 1)
	A = torch.ones(nbatch, 1, nelement_distmatrix).double().cuda()
	b = torch.min(torch.sum(weight1, 1), torch.sum(weight2, 1)).unsqueeze(1).double()
	flow = QPFunction(verbose=-1)(Q, p, G, h, A, b)

	emd_score = torch.sum((1 - Q_1).squeeze() * flow, 1)
	return emd_score, flow.view(-1, nelement_weight1, nelement_weight2)


def emd_inference_opencv(cost_matrix, weight1, weight2):
	# cost matrix is a tensor of shape [N,N]
	cost_matrix = cost_matrix.detach().cpu().numpy()

	weight1 = F.relu(weight1) + 1e-5
	weight2 = F.relu(weight2) + 1e-5

	weight1 = (weight1 * (weight1.shape[0] / weight1.sum().item())).view(-1, 1).detach().cpu().numpy()
	weight2 = (weight2 * (weight2.shape[0] / weight2.sum().item())).view(-1, 1).detach().cpu().numpy()

	cost, _, flow = cv2.EMD(weight1, weight2, cv2.DIST_USER, cost_matrix)
	return cost, flow


def emd_inference_opencv_test(distance_matrix, weight1, weight2):
	distance_list = []
	flow_list = []

	for i in range(distance_matrix.shape[0]):
		cost, flow = emd_inference_opencv(distance_matrix[i], weight1[i], weight2[i])
		distance_list.append(cost)
		flow_list.append(torch.from_numpy(flow))

	emd_distance = torch.Tensor(distance_list).cuda().double()
	flow = torch.stack(flow_list, dim=0).cuda().double()

	return emd_distance, flow


if __name__ == '__main__':
	random_seed = True
	if random_seed:
		pass
	else:

		seed = 1
		import random
		import numpy as np

		print('manual seed:', seed)
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

	batch_size = 50
	num_node = 25
	form = 'L2'  # in [ 'L2', 'QP' ]

	cosine_distance_matrix = torch.rand(batch_size, num_node, num_node).cuda()

	weight1 = torch.rand(batch_size, num_node).cuda()
	weight2 = torch.rand(batch_size, num_node).cuda()

	emd_distance_cv, cv_flow = emd_inference_opencv_test(cosine_distance_matrix, weight1, weight2)
	emd_distance_qpth, qpth_flow = emd_inference_qpth(cosine_distance_matrix, weight1, weight2, form=form)

	emd_score_cv = ((1 - cosine_distance_matrix) * cv_flow).sum(-1).sum(-1)
	emd_score_qpth = ((1 - cosine_distance_matrix) * qpth_flow).sum(-1).sum(-1)
	print('emd difference:', (emd_score_cv - emd_score_qpth).abs().max())
	pass
