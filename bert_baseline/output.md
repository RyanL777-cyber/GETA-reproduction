(test) C:\Users\iwant\Downloads\GETA\bert_baseline>python baseline_bert_squad.py
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|█████████████████████████████████████████████| 197/197 [00:00<00:00, 4057.10it/s]
BertForQuestionAnswering LOAD REPORT from: bert-base-uncased
Key                                        | Status     | 
-------------------------------------------+------------+-
cls.predictions.transform.LayerNorm.bias   | UNEXPECTED | 
bert.pooler.dense.weight                   | UNEXPECTED | 
cls.seq_relationship.weight                | UNEXPECTED | 
cls.predictions.transform.dense.weight     | UNEXPECTED | 
cls.predictions.transform.LayerNorm.weight | UNEXPECTED | 
bert.pooler.dense.bias                     | UNEXPECTED | 
cls.predictions.transform.dense.bias       | UNEXPECTED | 
cls.predictions.bias                       | UNEXPECTED | 
cls.seq_relationship.bias                  | UNEXPECTED | 
qa_outputs.weight                          | MISSING    | 
qa_outputs.bias                            | MISSING    | 

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
- MISSING:      those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
{'train_runtime': '3.579', 'train_samples_per_second': '17.88', 'train_steps_per_second': '2.235', 'train_loss': '5.853', 'epoch': '1'}
100%|████████████████████████████████████████████████████████████████████| 8/8 [00:03<00:00,  2.23it/s]
100%|████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 14.00it/s]

Final Evaluation Results:
{'exact_match': 0.0, 'f1': 7.553800366300367}
