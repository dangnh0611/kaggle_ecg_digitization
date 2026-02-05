import torch

__all__ = ["MLMMasker"]


class MLMMasker:
    """
    Ref: https://github.com/huggingface/transformers/blob/0dd65a03198424a41ec6948e445c313e9f292939/src/transformers/data/data_collator.py#L827
    """

    def __init__(
        self,
        tokenizer,
        mlm_prob=0.15,
        mask_prob=0.8,
        random_prob=0.1,
    ):
        self.tokenizer = tokenizer
        self.mlm_prob = mlm_prob
        self.mask_prob = mask_prob
        self.random_prob = random_prob
        self._random_prob = random_prob / (1.0 - mask_prob)
        self.mask_token_id = tokenizer.mask_token_id

    def __repr__(self):
        return f"MLMMasker(tokenizer={self.tokenizer.__class__}, mlm_prob = {self.mlm_prob}, mask_prob = {self.mask_prob}, random_prob = {self.random_prob}, mask_token_id = {self.mask_token_id})"

    def __call__(self, inputs, special_tokens_mask=None):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_prob)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -1  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, self.mask_prob)).bool()
            & masked_indices
        )
        inputs[indices_replaced] = self.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, self._random_prob)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
