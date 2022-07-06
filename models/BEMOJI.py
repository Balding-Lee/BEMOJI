"""
BEMOJI for pre-training
:author: Qizhi Li
"""
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertTokenizer, BertForMaskedLM


class FineTuneConfig:
    def __init__(self):
        self.hidden_size = 768

        self.bert_chinese_path = '../static_data/bert-base-chinese'
        self.bert_english_path = '../static_data/bert-base-uncased'

        self.fine_tune_chinese_texts_max_length = 64
        self.fine_tune_english_texts_max_length = 96
        self.fine_tune_chinese_emoji_max_length = 32
        self.fine_tune_english_emoji_max_length = 90

    def mlm_config(self, mlm_probability=0.15, special_tokens_mask=None,
                   prob_replace_mask=0.8, prob_replace_rand=0.1, prob_keep_ori=0.1):
        """
        MLM settings
        :param lm_probability:
        :param mlm_probability: the total number of masked tokens
        :param special_tokens_mask: specital token
        :param prob_replace_mask: the ratio of token being replaced by [MASK]
        :param prob_replace_rand: the ratio of token being replaced by other token
        :param prob_keep_ori: the ratio of keeping the original token
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


class BEMOJI(nn.Module):
    def __init__(self, args, config, device):
        super().__init__()
        self.args = args
        self.config = config
        self.mse_loss = nn.MSELoss()
        self.num_losses = 2
        self.device = device

        pretrained_weight, mlm_class, emoji_class = None, None, None
        tokenizer_class = BertTokenizer

        # if args.dataset == 'english':
        #     pretrained_weight = self.config.bert_english_path
        # elif args.dataset == 'chinese':
        #     pretrained_weight = self.config.bert_chinese_path

        mlm_class, emoji_class = BertForMaskedLM, BertModel

        if args.mode == 'fine_tune_english':
            # bert_model = BertModel.from_pretrained(pretrained_weight)
            # self.bert_pooler = bert_model.pooler.dense
            pretrained_weight = self.config.bert_english_path

        elif args.mode == 'fine_tune_chinese':
            # bert_model = BertModel.from_pretrained(pretrained_weight)
            # self.bert_pooler = bert_model.pooler.dense
            pretrained_weight = self.config.bert_chinese_path

        # elif args.mode == 'CLS':
        #     model_class = BertModel

        assert pretrained_weight is not None

        self.bert_config = BertConfig.from_pretrained(pretrained_weight)
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weight)
        self.emoji_bert = emoji_class.from_pretrained(pretrained_weight, config=self.bert_config).to(self.device)
        self.context_bert = mlm_class.from_pretrained(pretrained_weight, config=self.bert_config).to(self.device)

        # CovWeightingLoss Config
        # How to compute the mean statistics: Full mean or decaying mean.
        self.mean_decay = True if args.mean_sort == 'decay' else False
        # self.mean_decay = False
        self.mean_decay_param = args.mean_decay_param

        self.current_iter = -1
        self.alphas = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(self.device)

        # Initialize all running statistics at 0.
        self.running_mean_L = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(
            self.device)
        self.running_mean_l = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(
            self.device)
        self.running_S_l = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(
            self.device)
        self.running_std_l = None

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

    def cov_weighting_loss(self, losses):
        """
        https://github.com/rickgroen/cov-weighting/blob/main/losses/covweighting_loss.py
        :param losses: list
        :return:
        """
        L = torch.tensor(losses, requires_grad=False).to(self.device)

        # print(L)
        # print(self.current_iter)

        # If we are doing validation, we would like to return an unweighted loss be able
        # to see if we do not overfit on the training set.
        if not self.train:
            return torch.sum(L)

        # Increase the current iteration parameter.
        self.current_iter += 1
        # If we are at the zero-th iteration, set L0 to L. Else use the running mean.
        L0 = L.clone() if self.current_iter == 0 else self.running_mean_L
        # Compute the loss ratios for the current iteration given the current loss L.
        l = L / L0

        # If we are in the first iteration set alphas to all 1/32
        if self.current_iter <= 1:
            self.alphas = torch.ones((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(
                self.device) / self.num_losses
        # Else, apply the loss weighting method.
        else:
            ls = self.running_std_l / self.running_mean_l
            self.alphas = ls / torch.sum(ls)

        # Apply Welford's algorithm to keep running means, variances of L,l. But only do this throughout
        # training the model.
        # 1. Compute the decay parameter the computing the mean.
        if self.current_iter == 0:
            mean_param = 0.0
        # elif self.current_iter > 0 and self.mean_decay:
        #     mean_param = self.mean_decay_param
        else:
            mean_param = (1. - 1 / (self.current_iter + 1))

        # 2. Update the statistics for l
        x_l = l.clone().detach()
        new_mean_l = mean_param * self.running_mean_l + (1 - mean_param) * x_l
        self.running_S_l += (x_l - self.running_mean_l) * (x_l - new_mean_l)
        self.running_mean_l = new_mean_l

        # The variance is S / (t - 1), but we have current_iter = t - 1
        running_variance_l = self.running_S_l / (self.current_iter + 1)
        self.running_std_l = torch.sqrt(running_variance_l + 1e-8)

        # 3. Update the statistics for L
        x_L = L.clone().detach()
        self.running_mean_L = mean_param * self.running_mean_L + (1 - mean_param) * x_L

        # Get the weighted losses and perform a standard back-pass.
        weighted_losses = [self.alphas[i] * losses[i] for i in range(len(losses))]
        # print('alphas:', self.alphas)
        loss = sum(weighted_losses)
        return loss

    def forward(self, input_seqs, input_emoji_defs, input_prompt):
        """
        :param input_seqs: list
                The sequence without prompt
        :param input_emoji_defs: list
                emoji description sequence
        :param input_prompt: list
                The sequence with prompt
        """
        if self.args.mode == 'fine_tune_chinese':
            context_tokens = self.tokenizer.batch_encode_plus(input_seqs,
                                                              add_special_tokens=True,
                                                              max_length=self.config.fine_tune_chinese_texts_max_length,
                                                              padding='max_length',
                                                              truncation='longest_first',
                                                              return_tensors='pt')
            emoji_tokens = self.tokenizer.batch_encode_plus(input_emoji_defs,
                                                            add_special_tokens=True,
                                                            max_length=self.config.fine_tune_chinese_emoji_max_length,
                                                            padding='max_length',
                                                            truncation='longest_first',
                                                            return_tensors='pt')
            prompt_tokens = self.tokenizer.batch_encode_plus(input_prompt,
                                                             max_length=self.config.fine_tune_chinese_texts_max_length,
                                                             padding='max_length',
                                                             truncation='longest_first',
                                                             return_tensors='pt')
        elif self.args.mode == 'fine_tune_english':
            context_tokens = self.tokenizer.batch_encode_plus(input_seqs,
                                                              add_special_tokens=True,
                                                              max_length=self.config.fine_tune_english_texts_max_length,
                                                              padding='max_length',
                                                              truncation='longest_first',
                                                              return_tensors='pt')
            emoji_tokens = self.tokenizer.batch_encode_plus(input_emoji_defs,
                                                            add_special_tokens=True,
                                                            max_length=self.config.fine_tune_english_emoji_max_length,
                                                            padding='max_length',
                                                            truncation='longest_first',
                                                            return_tensors='pt')
            prompt_tokens = self.tokenizer.batch_encode_plus(input_prompt,
                                                             max_length=self.config.fine_tune_english_texts_max_length,
                                                             padding='max_length',
                                                             truncation='longest_first',
                                                             return_tensors='pt')

        context_input_ids = context_tokens['input_ids'].to(self.device)
        emoji_input_ids = emoji_tokens['input_ids'].to(self.device)
        emoji_att_mask = emoji_tokens['attention_mask'].to(self.device)

        prompt_input_ids = prompt_tokens['input_ids'].to(self.device)
        prompt_att_mask = prompt_tokens['attention_mask'].to(self.device)
        prompt_token_type_ids = prompt_tokens['token_type_ids'].to(self.device)

        context_inputs, context_labels = self.mask_tokens(context_input_ids)
        # emoji_inputs, emoji_labels = self.mask_tokens(emoji_tokens['input_ids'])

        context_features = self.context_bert(input_ids=context_inputs,
                                             labels=context_labels)
        emoji_features = self.emoji_bert(input_ids=emoji_input_ids,
                                         attention_mask=emoji_att_mask)['pooler_output']

        context_mask_loss = context_features.loss
        # emoji_mask_loss = emoji_features.loss

        # context_hidden = context_features.hidden_states[12]
        # emoji_hidden = emoji_features.hidden_states[12]

        prompt_features = self.context_bert(input_ids=prompt_input_ids,
                                            attention_mask=prompt_att_mask,
                                            token_type_ids=prompt_token_type_ids,
                                            output_hidden_states=True).hidden_states

        # 0: CLS, 1: [MASK]
        prompt_mask_hidden_states = prompt_features[12][:, 1, :]

        prompt_loss = self.mse_loss(prompt_mask_hidden_states, emoji_features)

        # print(self.current_iter)

        loss = self.cov_weighting_loss([context_mask_loss, prompt_loss])

        # print(prompt_loss, context_mask_loss)
        # print([context_mask_loss, prompt_loss])
        # print(torch.tensor([context_mask_loss, prompt_loss]))
        # print([context_mask_loss, prompt_loss][0].grad_fn)

        return loss
