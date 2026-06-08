import torch


class BaseCollator:
    """Turn a list of dataset samples into one batch.

    Broad idea:
    - dataset.py returns one variable-length sample at a time
    - the collator pads token tensors to a shared length
    - images stay as Python lists because each sample can have a different
      number of image tiles
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _empty_batch(self):
        return {"input_ids": [], "labels": [], "attention_mask": [], "images": []}

    def _pad_batch(self, batch, max_length):
        batch["input_ids"] = [
            torch.nn.functional.pad(
                ids,
                (max_length - len(ids), 0),
                value=self.tokenizer.pad_token_id,
            )
            for ids in batch["input_ids"]
        ]
        batch["labels"] = [
            torch.nn.functional.pad(
                labels,
                (max_length - len(labels), 0),
                value=self.tokenizer.pad_token_id,
            )
            for labels in batch["labels"]
        ]
        batch["attention_mask"] = [
            torch.nn.functional.pad(
                attention_mask,
                (max_length - len(attention_mask), 0),
                value=0,
            )
            for attention_mask in batch["attention_mask"]
        ]

    def _discard_samples_that_are_too_long(self, batch, max_length):
        filtered = [
            (input_ids, labels, attention_mask, images)
            for input_ids, labels, attention_mask, images in zip(
                batch["input_ids"],
                batch["labels"],
                batch["attention_mask"],
                batch["images"],
            )
            if len(input_ids) <= max_length
        ]

        if not filtered:
            return self._empty_batch()

        input_ids, labels, attention_masks, images = zip(*filtered)
        return {
            "input_ids": list(input_ids),
            "labels": list(labels),
            "attention_mask": list(attention_masks),
            "images": list(images),
        }

    def prepare_batch(self, samples, max_length=None):
        """Pad and stack one DataLoader batch.

        Raw input:
            samples: list[dict], each from VQADataset

        Expected output:
            input_ids: [B, T] torch.long
            labels: [B, T] torch.long
            attention_mask: [B, T]
            images: list length B
        """

        # handle empty batch
        if not samples:
            return self._empty_batch()

        # drop None rows
        samples = [sample for sample in samples if sample is not None]
        if not samples:
            return self._empty_batch()

        # batch is a list of dicts, each containing "input_ids",
        # "attention_mask", "labels", "images"
        # convert it to a dict of lists of tensors
        batch = {key: [sample[key] for sample in samples] for key in samples[0]}

        if max_length is not None:
            batch = self._discard_samples_that_are_too_long(batch, max_length)

        if len(batch["input_ids"]) == 0:
            return batch

        # pad samples to max length
        if max_length is None:
            max_length = max(len(input_ids) for input_ids in batch["input_ids"])
        self._pad_batch(batch, max_length)

        return {
            "input_ids": torch.stack(batch["input_ids"]),
            "attention_mask": torch.stack(batch["attention_mask"]),
            "images": batch["images"],
            "labels": torch.stack(batch["labels"]),
        }


class VQACollator(BaseCollator):
    """Collator for VQA-style causal LM training.

    Important invariant:
    label padding must be -100, because cross entropy ignores -100.
    Padding labels with the tokenizer pad ID would train the model to predict
    padding tokens.
    """

    def __init__(self, tokenizer, max_length):
        super().__init__(tokenizer)
        self.max_length = max_length

    def _pad_batch(self, batch, max_length):
        """
        Reimplementing to use -100 as the pad value for labels, so that it's ignored by the loss
        """

        batch["input_ids"] = [
            torch.nn.functional.pad(
                ids,
                (max_length - len(ids), 0),
                value=self.tokenizer.pad_token_id,
            )
            for ids in batch["input_ids"]
        ]
        batch["labels"] = [
            torch.nn.functional.pad(
                labels,
                (max_length - len(labels), 0),
                value=-100,
            )
            for labels in batch["labels"]
        ]
        batch["attention_mask"] = [
            torch.nn.functional.pad(
                attention_mask,
                (max_length - len(attention_mask), 0),
                value=0,
            )
            for attention_mask in batch["attention_mask"]
        ]

    def __call__(self, samples):
        return self.prepare_batch(samples, max_length=self.max_length)
