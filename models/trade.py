# import os
import json
import random
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from config import SLOT_GATE_DICT, PAD_TOKEN
from utils import iprint

class Trade(nn.Module):
    def __init__(self, args, device, vocabs, slots_dict):
        super(Trade, self).__init__()
        self.args = args
        self.device = device
        self.vocab = vocabs[0]
        self.mem_vocab = vocabs[1]
        self.slots_dict = slots_dict
        self.batch_size = args['batch']
        self.path = args['path']
        self.task = args['task']
        self.lr = args['learning_rate']
        self.dropout = args['dropout']
        self.hidden_size = args['hidden_size']

        self.encoder = Encoder(
            args=args,
            vocab_size=self.vocab.n_words,
            embed_size=self.hidden_size,
            dropout=self.dropout
        )
        self.decoder = Generator(
            vocab=self.vocab,
            shared_embedding=self.encoder.embedding,
            encoder_hidden_size=self.hidden_size,
            dropout=self.dropout,
            slots=self.slots_dict['all'],
            device=device
        )

    def forward(self, data, slots_type):
        slots = self.slots_dict[slots_type]
        # Encode and Decode
        use_teacher_forcing = random.random() < self.args["teacher_forcing_ratio"] and self.training
        # Build unknown mask for memory to encourage generalization
        if self.args['unk_mask'] and self.decoder.training:
            story_size = data['context'].size()
            rand_mask = np.random.binomial(
                n=np.ones((story_size[0], story_size[1]), dtype=np.int64),
                p=1 - self.dropout
            )
            rand_mask = torch.Tensor(data=rand_mask).long()
            rand_mask = rand_mask.to(device=self.device)
            # if USE_CUDA:
            #     rand_mask = rand_mask.cuda()
            story = data['context'] * rand_mask.long()
        else:
            story = data['context']

        # Encode dialog history
        encoded_outputs, encoded_hidden = self.encoder(story.transpose(0, 1), data['context_len'])

        # Get the words that can be copy from the memory
        # self.copy_list = data['context_plain']
        max_res_len = data['generate_y'].size(2) if self.encoder.training else 10

        all_point_outputs, all_gate_outputs, words_point_out = self.decoder.forward(
            batch_size=self.batch_size,
            encoded_hidden=encoded_hidden,
            encoded_outputs=encoded_outputs,
            encoded_lens=data['context_len'],
            story=story,
            max_res_len=max_res_len,
            target_batches=data['generate_y'],
            use_teacher_forcing=use_teacher_forcing,
            slots=slots
        )
        return all_point_outputs, all_gate_outputs, words_point_out


class Encoder(nn.Module):
    def __init__(self, args, vocab_size, embed_size, dropout, n_layers=1):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.n_layers = n_layers

        self.dropout = nn.Dropout(dropout)

        # input of shape (batch, seq_len)
        # output (batch, seq_len, embedding_dim)
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_size,
            padding_idx=PAD_TOKEN
        )
        self.embedding.weight.data.normal_(0, 0.1)

        # input of shape (seq_len, batch, input_size)
        # output of shape (seq_len, batch, num_directions * hidden_size)
        # h_n of shape (num_layers * num_directions, batch, hidden_size)
        self.gru = nn.GRU(
            input_size=embed_size,
            hidden_size=embed_size,
            num_layers=n_layers,
            bidirectional=True
        )

        if args["load_embedding"]:
            with open('data/embedding{}.json'.format(vocab_size)) as f:
                E = json.load(f)
            new = self.embedding.weight.data.new
            self.embedding.weight.data.copy_(new(E))
            self.embedding.weight.requires_grad = True

        if args["fix_embedding"]:
            self.embedding.weight.requires_grad = False

        iprint('Encoder embedding requires_grad {}'.format(self.embedding.weight.requires_grad))

    def forward(self, input_seqs, input_lengths):
        '''
        input_seqs is of shape (max_sentence_len, batch_size)
        input_lengths contains the actual lengths of all sentences
        '''
        embeddings = self.embedding(input_seqs)
        embeddings = self.dropout(embeddings)

        packed_embedded = nn.utils.rnn.pack_padded_sequence(input=embeddings, lengths=input_lengths)
        outputs, hidden = self.gru(packed_embedded)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(sequence=outputs)

        # (batch, hidden_size)
        # The original paper adds the forward and backward hidden states, so the dimension does not change
        # I think concatenation would make more sense, let's try
        # TODO: 对比 add / concat, 看哪个效果好
        # hidden = torch.cat((hidden[0], hidden[1]), dim=1) # concat
        hidden = hidden[0] + hidden[1] # add
        outputs = outputs[:, :, :self.embed_size] + outputs[:, :, self.embed_size:]

        # (batch, seq_len, num_directions * hidden_size), (1, batch, hidden_size)
        return outputs.transpose(0, 1), hidden.unsqueeze(0)


class Generator(nn.Module):
    def __init__(self, vocab, shared_embedding, encoder_hidden_size, dropout, slots, device):
        super(Generator, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.n_words
        self.embedding = shared_embedding
        self.hidden_size = encoder_hidden_size
        self.n_gates = len(SLOT_GATE_DICT)
        self.slots = slots
        self.device = device

        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size
        )

        self.W_ratio = nn.Linear(3 * self.hidden_size, 1)
        self.W_gate = nn.Linear(self.hidden_size, self.n_gates)

        # Create independent slot embeddings
        self.slot_w2i = {}
        for slot in self.slots:
            if slot.split('-')[0] not in self.slot_w2i.keys():
                self.slot_w2i[slot.split('-')[0]] = len(self.slot_w2i)
            if slot.split('-')[1] not in self.slot_w2i.keys():
                self.slot_w2i[slot.split('-')[1]] = len(self.slot_w2i)
        self.Slot_emb = nn.Embedding(len(self.slot_w2i), self.hidden_size)
        self.Slot_emb.weight.data.normal_(0, 0.1)

    def forward(self, batch_size, encoded_hidden, encoded_outputs, encoded_lens, story, max_res_len, target_batches, use_teacher_forcing, slots):
        all_point_outputs = torch.zeros(len(slots), batch_size, max_res_len, self.vocab_size, device=self.device)
        all_gate_outputs = torch.zeros(len(slots), batch_size, self.n_gates, device=self.device)

        # Create the domain+slot query embeddings
        slot_emb_dict = {}
        for domain_slot in slots:
            # Domain embbeding
            this_domain = domain_slot.split("-")[0]
            this_slot = domain_slot.split("-")[1]
            if this_domain in self.slot_w2i.keys():
                domain_w2idx = self.slot_w2i[this_domain]
                domain_w2idx = torch.tensor([domain_w2idx], device=self.device)
                # if USE_CUDA: domain_w2idx = domain_w2idx.cuda()
                domain_emb = self.Slot_emb(domain_w2idx)
            # Slot embbeding
            if this_slot in self.slot_w2i.keys():
                slot_w2idx = self.slot_w2i[this_slot]
                slot_w2idx = torch.tensor([slot_w2idx], device=self.device)
                # if USE_CUDA: slot_w2idx = slot_w2idx.cuda()
                slot_emb = self.Slot_emb(slot_w2idx)

            # Combine two embeddings as one query
            slot_emb_dict[domain_slot] = domain_emb + slot_emb

        # Compute pointer-generator output
        words_point_out = []
        for i, domain_slot in enumerate(slots): # TODO: Parallel this part to make it train faster
            hidden = encoded_hidden
            words = []
            domain_slot_emb = slot_emb_dict[domain_slot]
            # embedding size is the same as hidden state size
            # decoder_input = self.dropout(domain_slot_emb).expand(batch_size, self.hidden_size)
            decoder_input = self.dropout(domain_slot_emb).expand(batch_size, -1)
            for wi in range(max_res_len):
                dec_state, hidden = self.gru(decoder_input.unsqueeze(0), hidden)

                # context_vec, logits, prob = self.attend(encoded_outputs, hidden, encoded_lens)
                context_vectors, scores = self.attend(encoded_outputs, hidden.squeeze(0), encoded_lens)

                # Calculate slot-gate output for each domain-slot pair
                if wi == 0:
                    all_gate_outputs[i] = self.W_gate(context_vectors) # (batch, 3)

                p_vocab = self.attend_vocab(self.embedding.weight, hidden.squeeze(0)) # (batch, vocab_size)
                p_gen_vectors = torch.cat(
                    tensors=[dec_state.squeeze(0), context_vectors, decoder_input],
                    dim=-1
                ) # (batch, hidden_size * 3)
                p_gen = torch.sigmoid(self.W_ratio(p_gen_vectors)) # (batch, 1)

                p_context_ptr = torch.zeros(p_vocab.size(), device=self.device)
                p_gen = p_gen.expand_as(p_context_ptr)
                # if USE_CUDA: p_context_ptr = p_context_ptr.cuda()
                p_context_ptr.scatter_add_(1, story, scores)
                p_vocab_final = (1 - p_gen) * p_context_ptr + p_gen * p_vocab
                pred_word = torch.argmax(p_vocab_final, dim=1)
                words.append([self.vocab.index2word[word_i.item()] for word_i in pred_word])
                all_point_outputs[i, :, wi, :] = p_vocab_final
                if use_teacher_forcing:
                    decoder_input = self.embedding(target_batches[:, i, wi]) # Chosen word is next input
                else:
                    decoder_input = self.embedding(pred_word)
                # if USE_CUDA: decoder_input = decoder_input.cuda()
            words_point_out.append(words)

        return all_point_outputs, all_gate_outputs, words_point_out

    def attend(self, seq, cond, lens):
        """
        attend over the sequences `seq` using the condition `cond`.
        seq (batch, seq_len, num_directions * hidden_size)
        cond (batch, hidden_size)
        lens need to be ordered in descending order
        """
        scores = torch.bmm(seq, cond.unsqueeze(dim=-1)).squeeze(dim=-1) # (batch, seq_len)
        max_len = max(lens)
        for i, l in enumerate(lens):
            # mask padded positions to negative infinite
            scores[i:, l:max_len] = -float('inf')
        scores = F.softmax(scores, dim=1)
        context_vectors = torch.bmm(scores.unsqueeze(dim=1), seq).squeeze(dim=1) # (batch, hidden_size)
        return context_vectors, scores

    def attend_vocab(self, seq, cond):
        scores = cond.mm(seq.t())
        scores = F.softmax(scores, dim=1)
        return scores
