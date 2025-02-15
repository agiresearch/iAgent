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

def cal_ndcg_hr_single(answer,ranking_list,topk=10):
    try:
        rank = ranking_list.index(answer)
        # print(rank)
        HIT = 0
        NDCG = 0
        MRR = 1.0 / (rank + 1.0)
        if rank < topk:
            NDCG = 1.0 / np.log2(rank + 2.0)
            HIT = 1.0
    except ValueError:
        HIT = -1 
        NDCG = -1
        MRR = -1
    return HIT , NDCG , MRR 

class iAgent():
    def __init__(self, task_input, logger):
        self.task_input = task_input
        self.messages = []
        self.client = OpenAI(
            # This is the default and can be omitted
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

        self.workflow = [
            {
                "message": "Based on the following instruction, assist me in generating relevant knowledge. Please specify the types of descriptions that the recommended items should include. Do not directly recommend specific items. ",
                "tool_use": None
            },
            {
                "message": "Based on the information, give recommendations for the user based on the constraints. ",
                "tool_use": None
            }
        ]
        self.logger = logger

    def run(self):
        task_input = self.task_input
        instruction,title,description, asin,answer,candidate_ranked_list,pure_ranked_list = task_input['instruction'],task_input['title'],task_input['description'],task_input['asin'],task_input['answer'],task_input['ranked_list_str'],task_input['pure_ranked_list']
        reviewText = task_input["reviewText"]
        
        user_memory = ""
        for j in range(len(asin)):
            # Ensure the description is properly handled as a string
            description_str = description[j][-200:] if isinstance(description[j][-200:], str) else str(description[j][-200:])
            user_memory += "user historical information, item title:{},item description:{} ;".format(
                title[j], re.sub(u"\\<.*?\\>", "", description_str)
            )

        user_memory_previous = ""
        for j in range(len(asin)-1):
            # Ensure the description and reviewText are properly handled as strings
            description_str = description[j][-200:] if isinstance(description[j][-200:], str) else str(description[j][-200:])
            review_str = reviewText[j][-200:] if isinstance(reviewText[j][-200:], str) else str(reviewText[j][-200:])
            user_memory_previous += "title:{},description:{},review:{} \t ".format(
                title[j], re.sub(u"\\<.*?\\>", "", description_str),
                re.sub(u"\\<.*?\\>", "", review_str)
            )

        workflow = self.workflow

        try:
            if workflow:
                MRR = None
                for i, step in enumerate(workflow):
                    message = step["message"]
                    if i == 0:
                        self.messages.append({
                            "role": "assistant",
                            "content": "{} \n. Don’t use numerical numbering for the generated content; you can use bullet points instead. \n Instruction:{}".format(message,instruction)
                            })
                        retries = 0
                        while retries < 3:
                            try:
                                completion = self.client.chat.completions.create(
                                                messages=self.messages,
                                                model= "gpt-4o-mini",
                                                response_format={
                                                    "type": "json_schema",
                                                    "json_schema": {
                                                        "name": "custom_response",
                                                        "schema": {
                                                            "type": "object",
                                                            "properties": {
                                                                "knowledge": {
                                                                    "type": "string",
                                                                },
                                                            },
                                                            "required": ["knowledge"],
                                                            "additionalProperties": False
                                                        },
                                                        "strict": True
                                                    }
                                                }
                                            )
                                try:
                                    knowledge_tool_str = json.loads(completion.choices[0].message.content)["knowledge"]
                                    break
                                except Exception as e:
                                    self.logger.info(f"An unexpected error occurred: {e}")
                                    knowledge_tool_str = "Extract Error!"
                                    retries += 1

                            except Exception as e:
                                self.logger.info(f"An unexpected error occurred: {e}")
                                knowledge_tool_str = "Extract Error!"
                                retries += 1
                                self.logger.info(f"Request timed out. Attempt {retries} of {3}. Retrying in {5} seconds...")
                                time.sleep(5)  


                        str_instruction_print = "{} \n. Don’t use numerical numbering for the generated content; you can use bullet points instead. \n Instruction:{}".format(message,instruction)
                        self.logger.info(f"based on the message:{str_instruction_print}, LLM generate knowledge is: {knowledge_tool_str}\n")

                    if i == len(workflow) - 1:
                        self.messages.append({
                            "role": "assistant",
                            "content": "{}.\n Don’t use numerical numbering for the generated content; you can use bullet points instead. \n Candidate ranking list:{},Knowledge:{},Static Interest:{}, Pure Ranking List:{}".format(message,candidate_ranked_list,knowledge_tool_str,user_memory,pure_ranked_list)
                            })
                        retries = 0
                        ranker_str_logger = "{}.\n Don’t use numerical numbering for the generated content; you can use bullet points instead. \n Candidate ranking list:{},Knowledge:{},Static Interest:{}, Pure Ranking List:{}".format(message,candidate_ranked_list,knowledge_tool_str,user_memory,pure_ranked_list)
                        while retries < 3:
                            try:
                                completion = self.client.chat.completions.create(
                                    messages=self.messages,
                                    model= "gpt-4o-mini",
                                    response_format={
                                        "type": "json_schema",
                                        "json_schema": {
                                            "name": "custom_response",
                                            "schema": {
                                                "type": "object",
                                                "properties": {
                                                    "rerank_list": {
                                                        "type": "array",
                                                        "items": {
                                                            "type": "integer"
                                                        }
                                                    },
                                                    "explanation": {
                                                        "type": "array",
                                                        "items": {
                                                            "type": "string"
                                                        }
                                                    }
                                                },
                                                "required": ["rerank_list", "explanation"],
                                                "additionalProperties": False
                                            },
                                            "strict": True
                                        }
                                    }
                                )
                                response = completion.choices[0].message.content
                                try:
                                    response_dict = json.loads(response)
                                    rerank_list,explanation = response_dict["rerank_list"],response_dict["explanation"]
                                    break
                                except Exception as e:
                                    self.logger.info(f"An unexpected error occurred: {e}")
                                    rerank_list = "Extract Error!"
                                    explanation = "Extract Error!"
                                    retries += 1
                            
                            except Exception as e:
                                self.logger.info(f"An unexpected error occurred: {e}")
                                rerank_list = "Extract Error!"
                                explanation = "Extract Error!"
                                retries += 1
                                self.logger.info(f"Request timed out. Attempt {retries} of {3}. Retrying in {5} seconds...")
                                time.sleep(5)
                                # break


                        self.logger.info("message:{},explanation:{},pure_ranking_list:{}, answer, {} llm_ranking_list, {}\n".format(ranker_str_logger,explanation,pure_ranked_list,answer,rerank_list))
                        HIT_1 , NDCG_1 , MRR  = cal_ndcg_hr_single(answer,rerank_list,1)
                        HIT_3 , NDCG_3 , MRR  = cal_ndcg_hr_single(answer,rerank_list,3)
                        HIT_5 , NDCG_5 , MRR  = cal_ndcg_hr_single(answer,rerank_list,5)

                retry_mrr_times = 0
                while MRR == -1 and retry_mrr_times<3:
                    retry_mrr_times += 1
                    self.logger.info(f"Generate error ranking list: {rerank_list}\n")
                    retry_message_str = "Rerank list is out of the order, you should rerank the item from the pure ranking list. The previous list:{}. Therefore, try it again according the following information.".format(rerank_list)
                    self.messages.append({
                            "role": "assistant",
                            "content": "{}. \n Don’t use numerical numbering for the generated content; you can use bullet points instead. \n Candidate ranking list:{},Knowledge:{},,Static Inster:{},  Please generate the reranked list from Pure Ranking List:{}. The length of the reranked list should be 10.".format(retry_message_str,candidate_ranked_list,knowledge_tool_str,user_memory,pure_ranked_list)
                            })
                    retries = 0
                    while retries < 3:
                        try:
                            completion = self.client.chat.completions.create(
                                messages=self.messages,
                                model= "gpt-4o-mini",
                                response_format={
                                    "type": "json_schema",
                                    "json_schema": {
                                        "name": "custom_response",
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "rerank_list": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "integer"
                                                    }
                                                },
                                                "explanation": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "string"
                                                    }
                                                }
                                            },
                                            "required": ["rerank_list", "explanation"],
                                            "additionalProperties": False
                                        },
                                        "strict": True
                                    }
                                }
                            )
                            response = completion.choices[0].message.content
                            try:
                                response_dict = json.loads(response)
                                rerank_list,explanation = response_dict["rerank_list"],response_dict["explanation"]
                                break
                            except Exception as e:
                                self.logger.info(f"An unexpected error occurred: {e}")
                                rerank_list = "Extract Error!"
                                explanation = "Extract Error!"
                                retries += 1
                        
                        except Exception as e:
                            self.logger.info(f"An unexpected error occurred: {e}")
                            rerank_list = "Extract Error!"
                            retries += 1
                            self.logger.info(f"Request timed out. Attempt {retries} of {3}. Retrying in {5} seconds...")
                            time.sleep(5)
                            # break
                    
                    self.logger.info("one more time, pure_ranking_list:{}, answer, {} llm_ranking_list, {}\n".format(pure_ranked_list,answer,rerank_list))
                    HIT_1 , NDCG_1 , MRR  = cal_ndcg_hr_single(answer,rerank_list,1)
                    HIT_3 , NDCG_3 , MRR  = cal_ndcg_hr_single(answer,rerank_list,3)
                    HIT_5 , NDCG_5 , MRR  = cal_ndcg_hr_single(answer,rerank_list,5)

                return {
                    "HIT":(HIT_1,HIT_3,HIT_5),
                    "NDCG":(NDCG_1,NDCG_3,NDCG_5),
                    "MRR":MRR,
                }

            else:
                return {
                    "HIT":(-1,-1,-1),
                    "NDCG":(-1,-1,-1),
                    "MRR":-1,
                }
                    
        except Exception as e:
            self.logger.error(e)
            return {
                    "HIT":(-1,-1,-1),
                    "NDCG":(-1,-1,-1),
                    "MRR":-1,
                }        

class i2Agent():
    def __init__(self,task_input,logger):
        self.task_input = task_input
        self.messages = []
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        self.workflow = [
            {
                "message": "Here is the background of one user. ",
                "tool_use": None
            },
            {
                "message": "",
                "tool_use": None
            },
            {
                "message": "Based on the following instruction, assist me in generating relevant knowledge. Please specify the types of descriptions that the recommended items should include. Do not directly recommend specific items. ",
                "tool_use": None
            },
            {
                "message": "Based on the generated knowledge and the instruction, extract some dynamic interest information from the static memory. Moreover, based on the profile and the instruction, extract some dynamic profile information. ",
                "tool_use": None
            },
            {
                "message": "Based on the information, give recommendations for the user based on the constrains. ",
                "tool_use": None
            }
        ]
        self.logger = logger

    def run(self):
        max_length = 15
        task_input = self.task_input
        instruction,title,description, asin,answer,candidate_ranked_list,pure_ranked_list = task_input['instruction'],task_input['title'],task_input['description'],task_input['asin'],task_input['answer'],task_input['ranked_list_str'],task_input['pure_ranked_list']
        reviewText,neg_sample_title,neg_sample_descript = task_input["reviewText"],task_input["neg_sample_title"],task_input["neg_sample_descript"]
        title,description, asin = title[-max_length:],description[-max_length:], asin[-max_length:]

        user_memory = ""
        for j in range(len(asin)):
            # Ensure the description is properly handled as a string
            description_str = description[j][-200:] if isinstance(description[j][-200:], str) else str(description[j][-200:])
            user_memory += "user historical information, item title:{},item description:{} ;".format(
                title[j], re.sub(u"\\<.*?\\>", "", description_str)
            )

        user_memory_previous = ""
        for j in range(len(asin)-1):
            # Ensure the description and reviewText are properly handled as strings
            description_str = description[j][-200:] if isinstance(description[j][-200:], str) else str(description[j][-200:])
            review_str = reviewText[j][-200:] if isinstance(reviewText[j][-200:], str) else str(reviewText[j][-200:])
            user_memory_previous += "title:{},description:{},review:{} \t ".format(
                title[j], re.sub(u"\\<.*?\\>", "", description_str),
                re.sub(u"\\<.*?\\>", "", review_str)
            )
        workflow = self.workflow

        try:
            self.messages_initial = []
            self.messages_neighbor = []
            if workflow:
                MRR = None
                for i, step in enumerate(workflow):
                    message = step["message"]
                    tool_use = step["tool_use"]

                    if i == 0:
                        step_one_message_str = "{} \n. Please recommend one item for her. The first one title:{}, descrition:{}. The second one title:{}, description:{}. ".format(message,title[-2],re.sub(u"\\<.*?\\>", "",str(description[-2][-200:])),neg_sample_title,re.sub(u"\\<.*?\\>", "",str(neg_sample_descript[-200:])))
                        self.messages_initial.append({
                            "role": "assistant",
                            "content": "{} \n. Please recommend one item for her. The first one title:{}, descrition:{}. The second one title:{}, description:{}. ".format(message,title[-2],re.sub(u"\\<.*?\\>", "",str(description[-2][-200:])),neg_sample_title,re.sub(u"\\<.*?\\>", "",str(neg_sample_descript[-200:])))
                            })
                        retries = 0
                        while retries < 3:
                            try:
                                completion = self.client.chat.completions.create(
                                                messages=self.messages_initial,
                                                model= "gpt-4o-mini",
                                                response_format={
                                                    "type": "json_schema",
                                                    "json_schema": {
                                                        "name": "custom_response",
                                                        "schema": {
                                                            "type": "object",
                                                            "properties": {
                                                                "recommend_content": {
                                                                    "type": "string",
                                                                },
                                                            },
                                                            "required": ["recommend_content"],
                                                            "additionalProperties": False
                                                        },
                                                        "strict": True
                                                    }
                                                }
                                            )
                                try:
                                    recommend_content = json.loads(completion.choices[0].message.content)["recommend_content"]
                                    self.messages_initial.append({
                                        "role": "assistant",
                                        "content": "The recommend content: {} \n. ".format(recommend_content)
                                        })
                                    break
                                except Exception as e:
                                    self.logger.info(f"An unexpected error occurred: {e}")
                                    recommend_content = "Extract Error!"
                                    retries += 1    
                            
                            except Exception as e:
                                self.logger.info(f"An unexpected error occurred: {e}")
                                recommend_content = "Extract Error!"
                                retries += 1
                                self.logger.info(f"Request timed out. Attempt {retries} of {3}. Retrying in {5} seconds...")
                                time.sleep(5)
                                # break

                        self.logger.info("The first step message:{} \n".format(step_one_message_str))
                        self.logger.info(f"generate recommend_content is: {recommend_content}\n")

                    if i == 1:
                        second_message_str = "{} \n. Great! Actually, this user choose the item with title:{} and review:{}. Can you generate the profile of this user background? Please make a detailed profile. Don’t use numerical numbering for the generated content; you can use bullet points instead.".format(message,title[-2],reviewText[-2][-200:])
                        self.messages_initial.append({
                            "role": "assistant",
                            "content": "{} \n. Great! Actually, this user choose the item with title:{} and review:{}. Can you generate the profile of this user background? Please make a detailed profile. Don’t use numerical numbering for the generated content; you can use bullet points instead.".format(message,title[-2],reviewText[-2][-200:])
                            })
                        retries = 0
                        while retries < 3:
                            try:
                                completion = self.client.chat.completions.create(
                                                messages=self.messages_initial,
                                                model= "gpt-4o-mini",
                                                response_format={
                                                    "type": "json_schema",
                                                    "json_schema": {
                                                        "name": "custom_response",
                                                        "schema": {
                                                            "type": "object",
                                                            "properties": {
                                                                "profile": {
                                                                    "type": "string",
                                                                },
                                                            },
                                                            "required": ["profile"],
                                                            "additionalProperties": False
                                                        },
                                                        "strict": True
                                                    }
                                                }
                                            )
                                try:
                                    profile = json.loads(completion.choices[0].message.content)["profile"]
                                    break
                                except Exception as e:
                                    self.logger.info(f"An unexpected error occurred: {e}")
                                    profile = "Extract Error!"
                                    retries += 1
                            
                            except Exception as e:
                                self.logger.info(f"An unexpected error occurred: {e}")
                                profile = "Extract Error!"
                                retries += 1
                                self.logger.info(f"Request timed out. Attempt {retries} of {3}. Retrying in {5} seconds...")
                                time.sleep(5) 
                                # break

                        self.logger.info("second message :{}\n".format(second_message_str))
                        self.logger.info(f"generate profile is: {profile}\n")

                    if i == 2:
                        self.messages.append({
                            "role": "assistant",
                            "content": "{} \n. Don’t use numerical numbering for the generated content; you can use bullet points instead. \n Instruction:{}".format(message,instruction)
                            })
                        retries = 0
                        while retries < 3:
                            try:
                                completion = self.client.chat.completions.create(
                                                messages=self.messages,
                                                model= "gpt-4o-mini",
                                                response_format={
                                                    "type": "json_schema",
                                                    "json_schema": {
                                                        "name": "custom_response",
                                                        "schema": {
                                                            "type": "object",
                                                            "properties": {
                                                                "knowledge": {
                                                                    "type": "string",
                                                                },
                                                            },
                                                            "required": ["knowledge"],
                                                            "additionalProperties": False
                                                        },
                                                        "strict": True
                                                    }
                                                }
                                            )
                                try:
                                    knowledge_tool_str = json.loads(completion.choices[0].message.content)["knowledge"]
                                    break
                                except Exception as e:
                                    self.logger.info(f"An unexpected error occurred: {e}")
                                    knowledge_tool_str = "Extract Error!"
                                    retries += 1
                            
                            except Exception as e:
                                self.logger.info(f"An unexpected error occurred: {e}")
                                knowledge_tool_str = "Extract Error!"
                                retries += 1
                                self.logger.info(f"Request timed out. Attempt {retries} of {3}. Retrying in {5} seconds...")
                                time.sleep(5)  
                                # break

                        
                        self.logger.info(f"generate knowledge is: {knowledge_tool_str}\n")

                    if i == 3:
                        forth_message_str = "{}. Don’t use numerical numbering for the generated content; you can use bullet points instead. \n Generated Knowledge:{} Instruction:{} Historical Information:{} Profile:{}".format(message,knowledge_tool_str,instruction,user_memory,profile)
                        self.messages.append({
                            "role": "assistant",
                            "content": "{}. Don’t use numerical numbering for the generated content; you can use bullet points instead. \n Generated Knowledge:{} Instruction:{} Historical Information:{} Profile:{}".format(message,knowledge_tool_str,instruction,user_memory,profile)
                            })
                        retries = 0
                        while retries < 3:
                            try:
                                completion = self.client.chat.completions.create(
                                    messages=self.messages,
                                    model= "gpt-4o-mini",
                                    response_format={
                                        "type": "json_schema",
                                        "json_schema": {
                                            "name": "custom_response",
                                            "schema": {
                                                "type": "object",
                                                "properties": {
                                                    "dynamic_interest": {
                                                        "type": "string",
                                                    },
                                                    "dynamic_profile": {
                                                        "type": "string",
                                                    },
                                                },
                                                "required": ["dynamic_interest","dynamic_profile"],
                                                "additionalProperties": False
                                            },
                                            "strict": True
                                        }
                                    }
                                )
                                try:
                                    dynamic_interest_str = json.loads(completion.choices[0].message.content)["dynamic_interest"]
                                    dynamic_profile_str = json.loads(completion.choices[0].message.content)["dynamic_profile"]
                                    break
                                except Exception as e:
                                    self.logger.info(f"An unexpected error occurred: {e}")
                                    dynamic_interest_str = "Extract Error!"
                                    dynamic_profile_str = "Extract Error!"
                                    retries += 1
                            
                            except Exception as e:
                                self.logger.info(f"An unexpected error occurred: {e}")
                                dynamic_interest_str = "Extract Error!"
                                retries += 1
                                self.logger.info(f"Request timed out. Attempt {retries} of {3}. Retrying in {5} seconds...")
                                time.sleep(5)  

                        self.logger.info("forth step message is :{}\n".format(forth_message_str))
                        self.logger.info(f"dynamic interest is: {dynamic_interest_str} dynamic_profile_str:{dynamic_profile_str}\n")


                    if i == len(workflow) - 1:
                        self.messages.append({
                            "role": "assistant",
                            "content": "{}.\n Don’t use numerical numbering for the generated content; you can use bullet points instead. \n Candidate ranking list:{},Knowledge:{},Dynamic Interest:{},Static Interest:{}, Static User Profile:{}, Dynamic User Profile:{}, Pure Ranking List:{}".format(message,candidate_ranked_list,knowledge_tool_str,dynamic_interest_str,user_memory,profile,dynamic_profile_str,pure_ranked_list)
                            })
                        retries = 0
                        while retries < 3:
                            try:
                                completion = self.client.chat.completions.create(
                                    messages=self.messages,
                                    model= "gpt-4o-mini",
                                    response_format={
                                        "type": "json_schema",
                                        "json_schema": {
                                            "name": "custom_response",
                                            "schema": {
                                                "type": "object",
                                                "properties": {
                                                    "rerank_list": {
                                                        "type": "array",
                                                        "items": {
                                                            "type": "integer"
                                                        }
                                                    },
                                                    "explanation": {
                                                        "type": "array",
                                                        "items": {
                                                            "type": "string"
                                                        }
                                                    }
                                                },
                                                "required": ["rerank_list", "explanation"],
                                                "additionalProperties": False
                                            },
                                            "strict": True
                                        }
                                    }
                                )
                                response = completion.choices[0].message.content
                                try:
                                    response_dict = json.loads(response)
                                    rerank_list,explanation = response_dict["rerank_list"],response_dict["explanation"]
                                    break
                                except Exception as e:
                                    self.logger.info(f"An unexpected error occurred: {e}")
                                    rerank_list = "Extract Error!"
                                    explanation = "Extract Error!"
                                    retries += 1
                            
                            except Exception as e:
                                self.logger.info(f"An unexpected error occurred: {e}")
                                rerank_list = "Extract Error!"
                                explanation = "Extract Error!"
                                retries += 1
                                self.logger.info(f"Request timed out. Attempt {retries} of {3}. Retrying in {5} seconds...")
                                time.sleep(5) 
                                # break

                        self.logger.info("pure_ranking_list:{}, answer, {} llm_ranking_list, {}\n".format(pure_ranked_list,answer,rerank_list))
                        HIT_1 , NDCG_1 , MRR  = cal_ndcg_hr_single(answer,rerank_list,1)
                        HIT_3 , NDCG_3 , MRR  = cal_ndcg_hr_single(answer,rerank_list,3)
                        HIT_5 , NDCG_5 , MRR  = cal_ndcg_hr_single(answer,rerank_list,5)

                retry_mrr_times = 0
                while MRR == -1 and retry_mrr_times<3:

                    retry_mrr_times += 1
                    self.logger.info(f"Generate error ranking list: {rerank_list}\n")
                    retry_message_str = "Rerank list is out of the order, you should rerank the item from the pure ranking list. The previous list:{}. Therefore, try it again according the following information.".format(rerank_list)
                    self.messages.append({
                            "role": "assistant",
                            "content": "{}. \n Don’t use numerical numbering for the generated content; you can use bullet points instead. \n Candidate ranking list:{},Knowledge:{},Dynamic Interest:{},Static Inster:{}, User Profile:{}, Please generate the reranked list from Pure Ranking List:{}. The length of the reranked list should be 10.".format(retry_message_str,candidate_ranked_list,knowledge_tool_str,dynamic_interest_str,user_memory,profile,pure_ranked_list)
                            })
                    retries = 0
                    while retries < 3:
                        try:
                            # 尝试发送请求
                            completion = self.client.chat.completions.create(
                                messages=self.messages,
                                model= "gpt-4o-mini",
                                response_format={
                                    "type": "json_schema",
                                    "json_schema": {
                                        "name": "custom_response",
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "rerank_list": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "integer"
                                                    }
                                                },
                                                "explanation": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "string"
                                                    }
                                                }
                                            },
                                            "required": ["rerank_list", "explanation"],
                                            "additionalProperties": False
                                        },
                                        "strict": True
                                    }
                                }
                            )
                            response = completion.choices[0].message.content
                            try:
                                response_dict = json.loads(response)
                                rerank_list,explanation = response_dict["rerank_list"],response_dict["explanation"]
                                break
                            except Exception as e:
                                self.logger.info(f"An unexpected error occurred: {e}")
                                rerank_list = "Extract Error!"
                                explanation = "Extract Error!"
                                retries += 1
                        
                        except Exception as e:
                            self.logger.info(f"An unexpected error occurred: {e}")
                            rerank_list = "Extract Error!"
                            retries += 1
                            self.logger.info(f"Request timed out. Attempt {retries} of {3}. Retrying in {5} seconds...")
                            time.sleep(5)
                            # break
                    
                    self.logger.info("one more time, pure_ranking_list:{}, answer, {} llm_ranking_list, {}\n".format(pure_ranked_list,answer,rerank_list))
                    HIT_1 , NDCG_1 , MRR  = cal_ndcg_hr_single(answer,rerank_list,1)
                    HIT_3 , NDCG_3 , MRR  = cal_ndcg_hr_single(answer,rerank_list,3)
                    HIT_5 , NDCG_5 , MRR  = cal_ndcg_hr_single(answer,rerank_list,5)

                return {
                    "HIT":(HIT_1,HIT_3,HIT_5),
                    "NDCG":(NDCG_1,NDCG_3,NDCG_5),
                    "MRR":MRR,
                }

            else:
                return {
                    "HIT":(-1,-1,-1),
                    "NDCG":(-1,-1,-1),
                    "MRR":-1,
                }
                    
        except Exception as e:
            self.logger.error(e)
            return {
                    "HIT":(-1,-1,-1),
                    "NDCG":(-1,-1,-1),
                    "MRR":-1,
                }