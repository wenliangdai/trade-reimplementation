import os
import json
import pickle
import random
from tqdm import tqdm
import torch
from typing import List, Dict, Tuple
from collections import OrderedDict
from torch.utils.data import DataLoader
from embeddings import GloveEmbedding, KazumaCharEmbedding
from utils import iprint, fix_label_error
# from index import get_args
from vocabulary import Vocab
from dataset import Dataset
from config import SLOT_GATE_DICT, PAD_TOKEN

# domains used in our experiments
EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]
# data files
FILE_ONTOLOGY = 'data/multi-woz/MULTIWOZ2 2/ontology.json'
FILE_TRAIN = 'data/train_dials.json'
FILE_DEV = 'data/dev_dials.json'
FILE_TEST = 'data/test_dials.json'


# Get an array of domain-slot pair strings from the ontology
def get_slot_info(ontology: Dict) -> List[str]:
    slots = []
    for domain_slot in ontology.keys():
        domain = domain_slot.split("-")[0]
        if domain in EXPERIMENT_DOMAINS:
            if ('book' not in domain_slot):
                slots.append(domain_slot.replace(" ", "").lower())
            else:
                slots.append(domain_slot.lower())
    return slots


def dump_pretrained_emb(word2index, index2word, dump_path):
    iprint('Dumping pretrained embeddings...')
    embeddings = [GloveEmbedding(), KazumaCharEmbedding()]
    E = []
    for i in tqdm(range(len(word2index.keys()))):
        w = index2word[i]
        e = []
        for emb in embeddings:
            e += emb.emb(w, default='zero')
        E.append(e)
    with open(dump_path, 'wt') as f:
        json.dump(E, f)


def get_data(args, file, slots, dataset, vocab, mem_vocab, training, batch_size,
             shuffle=True) -> (List[Dict], int, List[Dict], DataLoader):
    pairs, max_len, this_slots = read_langs(
        args=args,
        file=file,
        slots=slots,
        dataset=dataset,
        vocab=vocab,
        mem_vocab=mem_vocab,
        training=training
    )
    dataloader = get_dataloader(
        args=args,
        pairs=pairs,
        vocab=vocab,
        mem_vocab=mem_vocab,
        batch_size=batch_size,
        shuffle=shuffle
    )
    return pairs, max_len, this_slots, dataloader

def get_all_data(args, training=True, batch_size=100) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # evaluation batch size
    eval_batch = args["eval_batch"] if args["eval_batch"] else batch_size

    # pickle file path
    if args['path']:
        saving_folder_path = args['path']
    else:
        saving_folder_path = 'save/{}-{}-{}-{}/'.format(args["decoder"], args["addName"], args['dataset'], args['task'])
    iprint('Path to save data: ' + saving_folder_path)

    if not os.path.exists(saving_folder_path):
        os.makedirs(saving_folder_path)

    # read domain-slot pairs
    ontology = json.load(open(FILE_ONTOLOGY, 'r'))
    all_slots = get_slot_info(ontology)

    # vocab
    vocab_name = 'vocab-all.pkl' if args["all_vocab"] else 'vocab-train.pkl'
    mem_vocab_name = 'mem-vocab-all.pkl' if args["all_vocab"] else 'mem-vocab-train.pkl'
    # if vocab files exist, read them in, otherwise we create new ones
    if os.path.exists(saving_folder_path + vocab_name) and os.path.exists(saving_folder_path + mem_vocab_name):
        iprint('Loading saved vocab files...')
        with open(saving_folder_path + vocab_name, 'rb') as handle:
            vocab = pickle.load(handle)
        with open(saving_folder_path + mem_vocab_name, 'rb') as handle:
            mem_vocab = pickle.load(handle)
    else:
        vocab = Vocab()
        vocab.index_words(all_slots, 'slot')
        mem_vocab = Vocab()
        mem_vocab.index_words(all_slots, 'slot')

    if training:
        pair_train, train_max_len, slot_train, train_dataloader = get_data(
            args=args,
            file=FILE_TRAIN,
            slots=all_slots,
            dataset='train',
            vocab=vocab,
            mem_vocab=mem_vocab,
            training=training,
            batch_size=batch_size,
            shuffle=True
        )

        nb_train_vocab = vocab.n_words
    else:
        pair_train, train_max_len, slot_train, train_dataloader, nb_train_vocab = [], 0, {}, [], 0

    pair_dev, dev_max_len, slot_dev, dev_dataloader = get_data(
        args=args,
        file=FILE_DEV,
        slots=all_slots,
        dataset='dev',
        vocab=vocab,
        mem_vocab=mem_vocab,
        training=training,
        batch_size=eval_batch,
        shuffle=False
    )

    pair_test, test_max_len, slot_test, test_dataloader = get_data(
        args=args,
        file=FILE_TEST,
        slots=all_slots,
        dataset='test',
        vocab=vocab,
        mem_vocab=mem_vocab,
        training=training,
        batch_size=eval_batch,
        shuffle=False
    )

    iprint('Dumping vocab files...')
    with open(saving_folder_path + vocab_name, 'wb') as handle:
        pickle.dump(vocab, handle)
    with open(saving_folder_path + mem_vocab_name, 'wb') as handle:
        pickle.dump(mem_vocab, handle)
    embedding_dump_path = 'data/embedding{}.json'.format(len(vocab.index2word))
    if not os.path.exists(embedding_dump_path) and args["load_embedding"]:
        dump_pretrained_emb(vocab.word2index, vocab.index2word, embedding_dump_path)

    test_4d = []
    if args['except_domain'] != '':
        pair_test_4d, _, _, test_4d = get_data(
            file=FILE_TEST,
            slots=all_slots,
            dataset='dev',
            vocab=vocab,
            mem_vocab=mem_vocab,
            training=training,
            batch_size=eval_batch,
            shuffle=False
        )

    max_word = max(train_max_len, dev_max_len, test_max_len) + 1

    iprint('Read %s pairs train' % len(pair_train))
    iprint('Read %s pairs dev' % len(pair_dev))
    iprint('Read %s pairs test' % len(pair_test))
    iprint('Vocab_size: %s' % vocab.n_words)
    iprint('Vocab_size Training %s' % nb_train_vocab)
    iprint('Vocab_size Belief %s' % mem_vocab.n_words)
    iprint('Max. length of dialog words for RNN: %s' % max_word)
    # iprint('USE_CUDA={}'.format(USE_CUDA))

    # slots_list = [all_slots, slot_train, slot_dev, slot_test]
    slots_dict = {
        'all': all_slots,
        'train': slot_train,
        'val': slot_dev,
        'test': slot_test
    }
    iprint('[Train Set & Dev Set Slots]: Number is {} in total'.format(len(slots_dict['val'])))
    iprint(slots_dict['val'])
    iprint('[Test Set Slots]: Number is {} in total'.format(len(slots_dict['test'])))
    iprint(slots_dict['test'])
    vocabs = [vocab, mem_vocab]
    return train_dataloader, dev_dataloader, test_dataloader, test_4d, vocabs, slots_dict, nb_train_vocab


def read_langs(args, file, slots, dataset, vocab, mem_vocab, training,
               max_line=None, update_vocab=False) -> (List[Dict], int, List[str]):
    iprint('Reading from {}'.format(file))

    data = []
    max_resp_len = 0
    max_value_len = 0
    domain_counter = {}
    with open(file) as f:
        dialogues = json.load(f)

        # integrate user utterance and system response into vocab
        for dialogue in dialogues:
            if (args['all_vocab'] or dataset == 'train') and training:
                for turn in dialogue['dialogue']:
                    vocab.index_words(turn['system_transcript'], 'utter')
                    vocab.index_words(turn['transcript'], 'utter')

        # determine training data ratio, default is 100%
        if training and dataset == 'train' and args['data_ratio'] != 100:
            random.Random(10).shuffle(dialogues)
            dialogues = dialogues[:int(len(dialogues) * 0.01 * args['data_ratio'])]

        cnt_lin = 1
        for dialogue in dialogues:
            dialogue_history = ''

            # Filtering and counting domains
            for domain in dialogue['domains']:
                if domain not in EXPERIMENT_DOMAINS:
                    continue
                if domain not in domain_counter.keys():
                    domain_counter[domain] = 0
                domain_counter[domain] += 1

            # Unseen domain setting
            if args['only_domain'] != '' and args['only_domain'] not in dialogue['domains']:
                continue
            if (
                args['except_domain'] != '' and
                dataset == 'test' and
                args['except_domain'] not in dialogue['domains']
            ) or (
                args['except_domain'] != '' and
                dataset != 'test' and
                [args['except_domain']] == dialogue['domains']
            ):
                continue

            # Reading data
            for turn in dialogue['dialogue']:
                turn_domain = turn['domain']
                turn_id = turn['turn_idx']
                turn_uttr = turn['system_transcript'] + ' ; ' + turn['transcript']
                turn_uttr_strip = turn_uttr.strip()
                dialogue_history += turn['system_transcript'] + ' ; ' + turn['transcript'] + ' ; '
                source_text = dialogue_history.strip()
                turn_belief_dict = fix_label_error(turn['belief_state'], False, slots)

                # Generate domain-dependent slot list
                slot_temp = slots
                if dataset == 'train' or dataset == 'dev':
                    if args['except_domain'] != '':
                        slot_temp = [k for k in slots if args['except_domain'] not in k]
                        turn_belief_dict = OrderedDict([(k, v) for k, v in turn_belief_dict.items() if args['except_domain'] not in k])
                    elif args['only_domain'] != '':
                        slot_temp = [k for k in slots if args['only_domain'] in k]
                        turn_belief_dict = OrderedDict([(k, v) for k, v in turn_belief_dict.items() if args['only_domain'] in k])
                else:
                    if args['except_domain'] != '':
                        slot_temp = [k for k in slots if args['except_domain'] in k]
                        turn_belief_dict = OrderedDict([(k, v) for k, v in turn_belief_dict.items() if args['except_domain'] in k])
                    elif args['only_domain'] != '':
                        slot_temp = [k for k in slots if args['only_domain'] in k]
                        turn_belief_dict = OrderedDict([(k, v) for k, v in turn_belief_dict.items() if args['only_domain'] in k])

                turn_belief_list = [str(k) + '-' + str(v) for k, v in turn_belief_dict.items()]

                if (args['all_vocab'] or dataset == 'train') and training:
                    mem_vocab.index_words(turn_belief_dict, 'belief')

                '''
                generate_y 是每个 slot 的 value 的 array (dontcare/none/真实值)
                gating_label 是每个 slot 的 value 的类型 (0/1/2), dontcare/none/ptr
                '''
                generate_y, gating_label = [], []
                # class_label, generate_y, slot_mask, gating_label = [], [], [], []
                # start_ptr_label, end_ptr_label = [], []
                for slot in slot_temp:
                    if slot in turn_belief_dict.keys():
                        generate_y.append(turn_belief_dict[slot])

                        if turn_belief_dict[slot] == 'dontcare':
                            gating_label.append(SLOT_GATE_DICT['dontcare'])
                        elif turn_belief_dict[slot] == 'none':
                            gating_label.append(SLOT_GATE_DICT['none'])
                        else:
                            gating_label.append(SLOT_GATE_DICT['ptr'])

                        if max_value_len < len(turn_belief_dict[slot]):
                            max_value_len = len(turn_belief_dict[slot])
                    else:
                        generate_y.append('none')
                        gating_label.append(SLOT_GATE_DICT['none'])

                data_detail = {
                    'ID': dialogue['dialogue_idx'],
                    'domains': dialogue['domains'],
                    'turn_domain': turn_domain,
                    'turn_id': turn_id,
                    'dialog_history': source_text,
                    'turn_belief': turn_belief_list,
                    'gating_label': gating_label,
                    'turn_uttr': turn_uttr_strip,
                    'generate_y': generate_y
                }
                data.append(data_detail)

                if max_resp_len < len(source_text.split()):
                    max_resp_len = len(source_text.split())

            cnt_lin += 1
            if max_line and cnt_lin >= max_line:
                break

    # add t{} to the lang file
    if "t{}".format(max_value_len - 1) not in mem_vocab.word2index.keys() and training:
        for time_i in range(max_value_len):
            mem_vocab.index_words("t{}".format(time_i), 'utter')

    iprint('domain_counter' + str(domain_counter))

    '''
    data -> an array of 每个 turn 的 data_detail 结构的数据
    max_resp_len -> 每个 turn，最长的 utterance + system_response 的长度 (word count)
    slot_temp -> 过滤后的 SLOTS, an array of strings of slot names
    '''
    return data, max_resp_len, slot_temp

def get_dataloader(args, pairs, vocab, mem_vocab, batch_size, shuffle) -> DataLoader:
    if shuffle and args['fisher_sample'] > 0:
        shuffle(pairs)
        pairs = pairs[:args['fisher_sample']]

    data_info = {}
    data_keys = pairs[0].keys()
    for k in data_keys:
        data_info[k] = []

    for pair in pairs:
        for k in data_keys:
            data_info[k].append(pair[k])

    dataset = Dataset(data_info, vocab.word2index, vocab.word2index, mem_vocab.word2index)

    if args["imbalance_sampler"] and shuffle:
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            sampler=ImbalancedDatasetSampler(dataset)
        )
    else:
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn
        )
    return data_loader


def collate_fn(data):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths) == 0 else max(lengths)
        padded_seqs = torch.ones(len(sequences), max_len).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        padded_seqs = padded_seqs.detach() # torch.tensor(padded_seqs)
        return padded_seqs, lengths

    def merge_multi_response(sequences):
        '''
        merge from batch * nb_slot * slot_len to batch * nb_slot * max_slot_len
        '''
        lengths = []
        for bsz_seq in sequences:
            length = [len(v) for v in bsz_seq]
            lengths.append(length)
        max_len = max([max(l) for l in lengths])
        padded_seqs = []
        for bsz_seq in sequences:
            pad_seq = []
            for v in bsz_seq:
                v = v + [PAD_TOKEN] * (max_len - len(v))
                pad_seq.append(v)
            padded_seqs.append(pad_seq)
        padded_seqs = torch.tensor(padded_seqs)
        lengths = torch.tensor(lengths)
        return padded_seqs, lengths

    def merge_memory(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths) == 0 else max(lengths) # avoid the empty belief state issue
        padded_seqs = torch.ones(len(sequences), max_len, 4).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            if len(seq) != 0:
                padded_seqs[i, :end, :] = seq[:end]
        return padded_seqs, lengths

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x['context']), reverse=True)
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    # merge sequences
    src_seqs, src_lengths = merge(item_info['context'])
    y_seqs, y_lengths = merge_multi_response(item_info["generate_y"])
    gating_label = torch.tensor(item_info["gating_label"])
    turn_domain = torch.tensor(item_info["turn_domain"])

    # if USE_CUDA:
    #     src_seqs = src_seqs.cuda()
    #     gating_label = gating_label.cuda()
    #     turn_domain = turn_domain.cuda()
    #     y_seqs = y_seqs.cuda()
    #     y_lengths = y_lengths.cuda()

    item_info["context"] = src_seqs
    item_info["context_len"] = src_lengths
    item_info["gating_label"] = gating_label
    item_info["turn_domain"] = turn_domain
    item_info["generate_y"] = y_seqs
    item_info["y_lengths"] = y_lengths
    return item_info


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset.turn_domain[idx]

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
