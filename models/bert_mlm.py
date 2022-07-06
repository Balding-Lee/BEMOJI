"""
微调 Transformer 模型
:author: Qizhi Li
"""
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModelForMaskedLM


class FineTuneConfig:
    def __init__(self):
        self.hidden_size = 768

        # self.bert_chinese_path = '../static_data/bert-base-chinese'
        # self.bert_english_path = '../static_data/bert-base-uncased'

        self.fine_tune_chinese_texts_max_length = 64
        self.fine_tune_english_texts_max_length = 64

    def mlm_config(self, mlm_probability=0.15, special_tokens_mask=None,
                   prob_replace_mask=0.8, prob_replace_rand=0.1, prob_keep_ori=0.1):
        """
        MLM 的设置
        :param lm_probability:
        :param mlm_probability: 被mask的token总数
        :param special_tokens_mask: 特殊token
        :param prob_replace_mask: 被替换成[MASK]的token比率
        :param prob_replace_rand: 被随机替换成其他token比率
        :param prob_keep_ori: 保留原token的比率
        """
        assert sum([prob_replace_mask, prob_replace_rand, prob_keep_ori]) == 1, ValueError(
            "Sum of the probs must equal to 1.")
        self.mlm_probability = mlm_probability
        self.special_tokens_mask = special_tokens_mask
        self.prob_replace_mask = prob_replace_mask
        self.prob_replace_rand = prob_replace_rand
        self.prob_keep_ori = prob_keep_ori

    def fine_tune_config(self, batch_size, epochs, learning_rate, weight_decay,
                         from_path, device):
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device
        self.from_path = from_path


class TransformerMLM(nn.Module):
    def __init__(self, args, config, device, parameter_path=None):
        super().__init__()
        self.args = args
        self.config = config
        self.device = device

        tokenizer_class = AutoTokenizer

        # if args.dataset == 'english':
        #     pretrained_weight = self.config.bert_english_path
        # elif args.dataset == 'chinese':
        #     pretrained_weight = self.config.bert_chinese_path

        mlm_class = AutoModelForMaskedLM
        pretrained_weight = config.from_path

        self.bert_config = AutoConfig.from_pretrained(pretrained_weight)
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weight)
        self.context_bert = mlm_class.from_pretrained(pretrained_weight, config=self.bert_config).to(self.device)
        if parameter_path:
            self.context_bert.load_state_dict(torch.load(parameter_path))

    def mask_tokens(self, inputs):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.config.mlm_probability)
        if self.config.special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = self.config.special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(
            torch.full(labels.shape, self.config.prob_replace_mask)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        current_prob = self.config.prob_replace_rand / (1 - self.config.prob_replace_mask)
        indices_random = torch.bernoulli(
            torch.full(labels.shape, current_prob)).bool() & masked_indices & ~indices_replaced
        indices_random = indices_random.to(self.device)
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long).to(self.device)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def forward(self, input_seqs):
        """
        :param input_seqs: list
                不带有 prompt 的序列
        """
        context_tokens = self.tokenizer.batch_encode_plus(input_seqs,
                                                          add_special_tokens=True,
                                                          max_length=self.config.fine_tune_chinese_texts_max_length,
                                                          padding='max_length',
                                                          truncation='longest_first',
                                                          return_tensors='pt')

        context_input_ids = context_tokens['input_ids'].to(self.device)

        context_inputs, context_labels = self.mask_tokens(context_input_ids)
        # emoji_inputs, emoji_labels = self.mask_tokens(emoji_tokens['input_ids'])

        context_features = self.context_bert(input_ids=context_inputs,
                                             labels=context_labels)

        context_mask_loss = context_features.loss

        return context_mask_loss