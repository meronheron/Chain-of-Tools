from rich.progress import track

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from WillMindS.utils.io import read_JSON

from corpus import judge_Dataset, judge_Collater, \
                    retriever_Dataset, retriever_Collater

def tool_judge_train(config, logger, model, dataset_dir_dict, mode="train+test"):
    logger.info("============ Training tool judger ============")

    optimizer = optim.AdamW(model.tool_judge.parameters(), lr=config.lr)
    collater = judge_Collater(model.tokenizer)

    if mode != "test":
        raw_dataset = []
        for dataset_name in dataset_dir_dict:
            raw_dataset.extend(read_JSON(dataset_dir_dict[dataset_name]+"train.jsonl"))
        train_dataset = judge_Dataset(raw_dataset)
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collater)

    for epoch in range(1, config.train_epoch+1):
        if mode !="test":
            model.train()
            counter_dict = {"correct": 0,
                            "predict": 0,
                            "gold": 0}
            logger.info("------ Training Epoch {} ------".format(epoch))
            for step, data in track(enumerate(train_dataloader), description='Training epoch {} ...'.format(epoch)):
                all_steps = len(train_dataloader)
                for key,_ in data.items():
                    data[key] = data[key].cuda()
                with torch.no_grad():
                    foundation_output = model.foundation_model(data["input_ids"], output_hidden_states=True)
                judge_logits = model.tool_judging(foundation_output.hidden_states[-1][0])
                judge_logits = judge_logits.to(dtype=torch.float16)
                judge_logits = torch.sigmoid(judge_logits)

                label = data["judge_labels"][0].to(dtype=judge_logits.dtype, device=judge_logits.device)
                loss = F.binary_cross_entropy(judge_logits, label)
                loss.backward()

                if step % config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.tool_judge.parameters(), config.max_grad_norm)
                    optimizer.step()
                    model.zero_grad()

                correct_num, predict_num, gold_num = tool_judge_eval_calculate(judge_logits.round().int(), data["judge_labels"][0])
                counter_dict["correct"] += correct_num
                counter_dict["predict"] += predict_num
                counter_dict["gold"] += gold_num

                if step % 100 == 0 and step > 1:
                    f1 = 2 * counter_dict["correct"] / (counter_dict["predict"] + counter_dict["gold"] + 1e-10)
                    logger.info('step:{}/{}   '.format(step+1, all_steps)+'loss:'+str(loss.item())+' F1:'+str(f1))
                    counter_dict["correct"], counter_dict["predict"], counter_dict["gold"] = 0, 0, 0

        if mode != "train":
            for dataset_name in dataset_dir_dict:
                dev_dataset = judge_Dataset(read_JSON(dataset_dir_dict[dataset_name]+'dev.jsonl'))
                eval_counter_dict = {"correct": 0,
                                     "predict": 0,
                                     "gold": 0}
                model.eval()
                dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=True, collate_fn=collater)
                for step, data in track(enumerate(dev_dataloader), description='Evaling ...'):
                    for key,_ in data.items():
                        data[key] = data[key].cuda()
                    with torch.no_grad():
                        foundation_output = model.foundation_model(data["input_ids"], output_hidden_states=True)
                        judge_logits = model.tool_judging(foundation_output.hidden_states[-1][0])
                        judge_logits = torch.sigmoid(judge_logits)
                    correct_num, predict_num, gold_num = tool_judge_eval_calculate(judge_logits.round().int(), data["judge_labels"][0])
                    eval_counter_dict["correct"] += correct_num
                    eval_counter_dict["predict"] += predict_num
                    eval_counter_dict["gold"] += gold_num

                p = eval_counter_dict["correct"] / (eval_counter_dict["predict"] + 1e-10)
                r = eval_counter_dict["correct"] / (eval_counter_dict["gold"] + 1e-10)
                f1 = 2 * eval_counter_dict["correct"] / (eval_counter_dict["predict"] + eval_counter_dict["gold"] + 1e-10)
                logger.info('Eval: P: {}  R: {}  F1: {} '.format(str(p), str(r), str(f1)))

            if mode == "test":
                break

        if epoch >= 1:
            model.sl_judge(config.model_dir, "save", "epoch_"+str(epoch))

def tool_judge_eval_calculate(predict_labels, gold_labels):
    correct_num = (predict_labels * gold_labels).eq(1).sum()
    predict_num = predict_labels.eq(1).sum()
    gold_num = gold_labels.eq(1).sum()
    return correct_num, predict_num, gold_num

def return_retriever_train():
    pass
