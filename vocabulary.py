from config import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN

class Vocab:
    def __init__(self):
        self.word2index = {}
        self.index2word = {
            PAD_TOKEN: "PAD",
            SOS_TOKEN: "SOS",
            EOS_TOKEN: "EOS",
            UNK_TOKEN: 'UNK'
        }
        # number of words
        self.n_words = len(self.index2word)
        self.word2index = dict([(v, k) for k, v in self.index2word.items()])

    def index_words(self, sentence, type):
        if type == 'utter':
            for word in sentence.split(' '):
                self.index_word(word)
        elif type == 'slot':
            for domain_slot in sentence:
                domain, slot = domain_slot.split('-')
                self.index_word(domain)
                for slot_word in slot.split(' '):
                    self.index_word(slot_word)
        elif type == 'belief':
            for domain_slot, value in sentence.items():
                domain, slot = domain_slot.split('-')
                self.index_word(domain)
                for slot_word in slot.split(' '):
                    self.index_word(slot_word)
                for v in value.split(' '):
                    self.index_word(v)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1
