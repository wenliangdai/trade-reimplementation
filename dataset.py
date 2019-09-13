import torch
from config import EOS_TOKEN, UNK_TOKEN

class Dataset(torch.utils.data.Dataset):
    # Custom data.Dataset compatible with data.DataLoader
    def __init__(self, data_info, src_word2id, trg_word2id, mem_word2id):
        # Reads source and target sequences from txt files
        self.ID = data_info['ID']
        self.turn_domain = data_info['turn_domain']
        self.turn_id = data_info['turn_id']
        self.dialog_history = data_info['dialog_history']
        self.turn_belief = data_info['turn_belief']
        self.gating_label = data_info['gating_label']
        self.turn_uttr = data_info['turn_uttr']
        self.generate_y = data_info['generate_y']
        self.num_total_seqs = len(self.dialog_history)
        self.src_word2id = src_word2id
        self.trg_word2id = trg_word2id
        self.mem_word2id = mem_word2id

    def __getitem__(self, index):
        # Returns one data pair (source and target)
        ID = self.ID[index]
        turn_id = self.turn_id[index]
        turn_belief = self.turn_belief[index]
        gating_label = self.gating_label[index]
        turn_uttr = self.turn_uttr[index]
        turn_domain = self.preprocess_domain(self.turn_domain[index])
        generate_y = self.generate_y[index]
        generate_y = self.preprocess_slot(generate_y, self.trg_word2id)
        context = self.dialog_history[index]
        context = self.preprocess(context, self.src_word2id)
        context_plain = self.dialog_history[index]

        item_info = {
            'ID': ID,
            'turn_id': turn_id,
            'turn_belief': turn_belief,
            'gating_label': gating_label,
            'context': context,
            'context_plain': context_plain,
            'turn_uttr_pltorchain': turn_uttr,
            'turn_domain': turn_domain,
            'generate_y': generate_y
        }
        return item_info

    def __len__(self):
        return self.num_total_seqs

    def preprocess(self, sequence, word2idx):
        # Converts words to ids.
        story = [word2idx[word] if word in word2idx else UNK_TOKEN for word in sequence.split()]
        story = torch.Tensor(story)
        return story

    def preprocess_slot(self, sequence, word2idx):
        # Converts words to ids.
        story = []
        for value in sequence:
            v = [word2idx[word] if word in word2idx else UNK_TOKEN for word in value.split()] + [EOS_TOKEN]
            story.append(v)
        # story = torch.Tensor(story)
        return story

    def preprocess_memory(self, sequence, word2idx):
        # Converts words to ids.
        story = []
        for value in sequence:
            d, s, v = value
            s = s.replace('book', '').strip()
            # separate each word in value to different memory slot
            for wi, vw in enumerate(v.split()):
                idx = [word2idx[word] if word in word2idx else UNK_TOKEN for word in [d, s, 't{}'.format(wi), vw]]
                story.append(idx)
        story = torch.Tensor(story)
        return story

    def preprocess_domain(self, turn_domain):
        domains = {
            'attraction': 0,
            'restaurant': 1,
            'taxi': 2,
            'train': 3,
            'hotel': 4,
            'hospital': 5,
            'bus': 6,
            'police': 7
        }
        return domains[turn_domain]
