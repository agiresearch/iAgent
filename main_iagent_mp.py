# This is a main script that tests the functionality of specific agents.
# It requires no user input.
import json
import pandas as pd
import os
import warnings
import argparse

from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import re
from openai import OpenAI
import openai
import logging
from pathlib import Path
import time
import random
from model import iAgent,i2Agent

class AverageMeter(object):
    def __init__(self, *keys: str):
        self.totals = {key: 0.0 for key in keys}
        self.counts = {key: 0 for key in keys}

    def update(self, **kwargs: float) -> None:
        for key, value in kwargs.items():
            self._check_attr(key)
            self.totals[key] += value
            self.counts[key] += 1

    def __getattr__(self, attr: str) -> float:
        self._check_attr(attr)
        total = self.totals[attr]
        count = self.counts[attr]
        return total / count if count else 0.0

    def _check_attr(self, attr: str) -> None:
        assert attr in self.totals and attr in self.counts

def return_title_ranking_list(ranked_list,title_dict,descript_dict):
    string_ranked_list = ""
    for i in ranked_list:
        title = title_dict[i] if isinstance(title_dict[i], str) else str(title_dict[i])
        description = descript_dict[i] if isinstance(descript_dict[i], str) else str(descript_dict[i])

        cleaned_description = re.sub(u"\\<.*?\\>", "", description)[:20]

        string_ranked_list += "item id:{}, corresponding title:{}, description:{} ;".format(i, title[:50], cleaned_description)
        # string_ranked_list += "item id:{},corresponding title:{}, description:{} ;".format(i,title_dict[i][:-20],re.sub(u"\\<.*?\\>", "",descript_dict[i][:-20]))
    return string_ranked_list

def parse_response_last(text):
    # Extract ranking list using regex
    ranking_list_pattern = re.compile(r'\[([\d,\s]+)\]')
    ranking_list_match = ranking_list_pattern.search(text)

    # Extract explanations using regex
    explanation_pattern = re.compile(r'(\d+)\.\s\*\*(.*?)\*\*\s-\s(.*?)\n', re.DOTALL)
    explanation_matches = explanation_pattern.findall(text)

    # Process the ranking list
    if ranking_list_match:
        ranking_list_str = ranking_list_match.group(1)
        ranking_list = [int(x) for x in ranking_list_str.split(',')]
    else:
        ranking_list = []

    # Process the explanations into a dictionary
    explanations = {
        int(match[0]): {
            'title': match[1].strip(),
            'description': match[2].strip()
        }
        for match in explanation_matches
    }
    return ranking_list,explanation_matches

def init_logger(log_dir: str, log_file: str) -> None:
    logger = logging.getLogger()
    format_str = r'[%(asctime)s] %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        datefmt=r'%Y/%m/%d %H:%M:%S',
        format=format_str
    )
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(str(log_dir / log_file))
    fh.setFormatter(logging.Formatter(format_str))
    logger.addHandler(fh)
    return logger


from concurrent.futures import ThreadPoolExecutor, as_completed

def main(chunk_num, df_data, logger, args, title_id_dict, descript_id_dict):
    instruction = df_data["instruction"].tolist()
    title = df_data['title'].tolist()
    description = df_data['description'].tolist()
    asin = df_data['asin'].tolist()
    ranked_lists = df_data['ranked_lists'].tolist()
    reviewText = df_data['reviewText'].tolist()

    all_item_set = set(title_id_dict.keys())

    # parse arguments and set configuration for this run accordingly
    warnings.filterwarnings("ignore")
    # run agents concurrently for maximum efficiency using a scheduler
    stats = AverageMeter('hit1', 'hit3', 'hit5', 'ndcg3', 'ndcg5', 'mrr', 'agent_turnaround_time')

    futures = []
    error_num = 0
    right_num = 0
    with ThreadPoolExecutor(max_workers=1) as executor:
        for i in range(len(instruction)):
            try:
                rank_str_tmp = return_title_ranking_list(ranked_lists[i], title_id_dict, descript_id_dict)
            except:
                logger.info("merge rank list error in {}-th data ".format(i))
                continue

            positive_samples = set(asin[i])

            negative_samples = all_item_set - positive_samples

            negative_sample = random.choice(list(negative_samples))

            neg_sample_title = title_id_dict[negative_sample]
            neg_sample_descript = descript_id_dict[negative_sample]

            task_input = {
                "instruction": instruction[i],
                "title": title[i][:-1],
                "description": description[i][:-1],
                "asin": asin[i][:-1],
                "answer": asin[i][-1],
                "ranked_list_str": rank_str_tmp,
                "reviewText": reviewText[i][:-1],
                "neg_sample_title": neg_sample_title,
                "neg_sample_descript": neg_sample_descript,
                "pure_ranked_list": ranked_lists[i]
            }

            if args.agent_type == "static":
                rec_agent = iAgent(task_input, logger)
            elif args.agent_type == "dynamic":
                rec_agent = i2Agent(task_input, logger)
            futures.append(executor.submit(rec_agent.run))
            if i > 1:
                break
        for future in as_completed(futures):
            try:
                result = future.result()
                HIT_1, HIT_3, HIT_5 = result['HIT']
                NDCG_1, NDCG_3, NDCG_5 = result['NDCG']
                MRR = result['MRR']
                # agent_turnaround_time = result['agent_turnaround_time']
                if HIT_1 != -1:
                    stats.update(hit1=HIT_1, hit3=HIT_3, hit5=HIT_5, ndcg3=NDCG_3, ndcg5=NDCG_5, mrr=MRR)
                    right_num += 1
                else:
                    error_num += 1
            except Exception as e:
                logger.error(f"Error in processing future: {e}")
                error_num += 1

    logger.info("right_num:{}   error number :{}".format(right_num,error_num))
    logger.info("chunk number:{},len of data:{},hit1:{},hit3:{},hit5:{},ndcg3:{},ndcg5:{},mrr:{}".format(chunk_num,len(df_data),stats.hit1,stats.hit3,stats.hit5,stats.ndcg3,stats.ndcg5,stats.mrr)) 
    with open('result/{}_{}/results_ours_{}.txt'.format(args.dataset,args.domain,args.agent_type),'a+') as f:  
        f.write("chunk number:{},len of data:{},hit1:{},hit3:{},hit5:{},ndcg3:{},ndcg5:{},mrr:{}".format(chunk_num,len(df_data),stats.hit1,stats.hit3,stats.hit5,stats.ndcg3,stats.ndcg5,stats.mrr))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='iAgent')
    parser.add_argument('--dataset', type=str, default="amazon", help='type of dataset')
    parser.add_argument('--domain', type=str, default="books", help='type of dataset')
    parser.add_argument('--agent_type', type=str, default="static", help='type of runagent')

    args = parser.parse_args()


    df_data = pd.read_pickle("data/{}All_recagent.pkl".format(args.domain))
    logger = init_logger("result/{}_{}/".format(args.dataset,args.domain), "resultsteps_{}".format(time.strftime('%m_%d_%H_%M_%S', time.localtime()))+"_{}.log".format(args.agent_type))
    logger.info(vars(args))
    split_size = 1000
    num_chunks = len(df_data) // split_size + (1 if len(df_data) % split_size != 0 else 0)
    df_dict = pd.read_csv("data/combined_{}_asin_mapping.csv".format(args.domain))
    title_id_dict, descript_id_dict = {}, {}
    for i in range(len(df_dict)):
        title_id_dict[df_dict['index'].iloc[i]] = df_dict['title'].iloc[i]
        descript_id_dict[df_dict['index'].iloc[i]] = df_dict['description'].iloc[i]
    for i in range(num_chunks):
        start_index = i * split_size
        end_index = min(start_index + split_size,len(df_data))
        df_chunk = df_data.iloc[start_index:end_index]
        print("len:{}".format(len(df_chunk)))
        main(i,df_chunk,logger,args,title_id_dict,descript_id_dict)