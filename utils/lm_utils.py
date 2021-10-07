import torch
from transformers import AutoModelForMaskedLM, BertJapaneseTokenizer
import numpy as np


def subset_mlm(sequence, candidates, model, tokenizer):

  sequence += tokenizer.mask_token
  inputs = tokenizer(sequence, return_tensors="pt")
  mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

  token_logits = model(**inputs).logits
  mask_token_probs = token_logits[0, mask_token_index, :].softmax(1)

  token_subset = tokenizer(candidates, return_tensors="pt")['input_ids'][0].tolist()[1:-1]
  probs_subset = mask_token_probs[:,token_subset][0].tolist()

  sorted_out = [(tokenizer.decode([t]), p) for t, p in sorted(zip(token_subset, probs_subset), key=lambda x: x[1], reverse=True)]
  sorted_out_no_unk = [x for x in sorted_out if x[0] != tokenizer.unk_token]

  assert len(sorted_out_no_unk) > 0, "All token candidates tokenized as UNK!"

  return sorted_out_no_unk


def beam_search_from_marginal_mlm(candidate_chars, model, tokenizer, beams=3):

  def unique_ordered(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

  candidate_chars = [unique_ordered(c) for c in candidate_chars]
  sequences = [(c, 0.0) for c in candidate_chars[0]]
  candidate_chars = candidate_chars[1:]

  for i in range(len(candidate_chars)):
    all_candidates = []

    for j in range(len(sequences)):
      seq, score = sequences[j]
      try:
        preds = subset_mlm(seq, candidate_chars[i], model, tokenizer)
      except AssertionError:
        return None
      log_preds = [(t, np.log(p)) for t, p in preds]

      for k in range(len(log_preds)):
        candidate = [seq + log_preds[k][0], score - log_preds[k][1]]
        all_candidates.append(candidate)

    ordered = sorted(all_candidates, key=lambda x: x[1])
    sequences = ordered[:beams]

  return sequences


if __name__ == '__main__':

    model = AutoModelForMaskedLM.from_pretrained('cl-tohoku/bert-base-japanese-char-v2')
    tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-char-v2")
    beam_search_from_marginal_mlm(["人入天", "事亊庸", "課註許", "長畏艮果"], model, tokenizer, beams=2)