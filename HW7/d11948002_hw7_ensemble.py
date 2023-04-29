#%%

import json
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset 
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModelForQuestionAnswering
from tqdm.auto import tqdm

_exp = "ensemble"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

#%%

def same_seeds(seed):
	torch.manual_seed(seed)
	if torch.cuda.is_available():
			torch.cuda.manual_seed(seed)
			torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
same_seeds(2)


#%%

tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-large")

#%%

def read_data(file):
    with open(file, 'r', encoding="utf-8") as reader:
        data = json.load(reader)
    return data["questions"], data["paragraphs"]

data_dir = "/neodata/ML/hw7_dataset/"
train_questions, train_paragraphs = read_data(data_dir+"hw7_train.json")
dev_questions, dev_paragraphs = read_data(data_dir+"hw7_dev.json")
test_questions, test_paragraphs = read_data(data_dir+"hw7_test.json")


#%%

train_questions_tokenized = tokenizer([train_question["question_text"] for train_question in train_questions], add_special_tokens=False)
dev_questions_tokenized = tokenizer([dev_question["question_text"] for dev_question in dev_questions], add_special_tokens=False)
test_questions_tokenized = tokenizer([test_question["question_text"] for test_question in test_questions], add_special_tokens=False) 

train_paragraphs_tokenized = tokenizer(train_paragraphs, add_special_tokens=False)
dev_paragraphs_tokenized = tokenizer(dev_paragraphs, add_special_tokens=False, return_offsets_mapping=True)
test_paragraphs_tokenized = tokenizer(test_paragraphs, add_special_tokens=False, return_offsets_mapping=True)

# You can safely ignore the warning message as tokenized sequences 
# will be futher processed in datset __getitem__ before passing to model


class QA_Dataset(Dataset):
    def __init__(self, split, questions, tokenized_questions, tokenized_paragraphs):
        self.split = split
        self.questions = questions
        self.tokenized_questions = tokenized_questions
        self.tokenized_paragraphs = tokenized_paragraphs
        self.max_question_len = 200
        self.max_paragraph_len = 300
        
        ##### TODO: Change value of doc_stride #####
        self.doc_stride = 80

        # Input sequence length = [CLS] + question + [SEP] + paragraph + [SEP]
        self.max_seq_len = 1 + self.max_question_len + 1 + self.max_paragraph_len + 1

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        tokenized_question = self.tokenized_questions[idx]
        tokenized_paragraph = self.tokenized_paragraphs[question["paragraph_id"]]

        ##### TODO: Preprocessing #####
        # Hint: How to prevent model from learning something it should not learn
        if self.split == "train":
            # Convert answer's start/end positions in paragraph_text to start/end positions in tokenized_paragraph  
            answer_start_token = tokenized_paragraph.char_to_token(question["answer_start"])
            answer_end_token = tokenized_paragraph.char_to_token(question["answer_end"])

            # A single window is obtained by slicing the portion of paragraph containing the answer
            mid = (answer_start_token + answer_end_token) // 2

            #references: https://github.com/Singyuan/Machine-Learning-NTUEE-2022/blob/master/hw7/hw7.ipynb

            answer_length = answer_end_token - answer_start_token + 1
            if answer_length // 2 < self.max_paragraph_len - answer_length // 2:
                rnd = random.randint(answer_length // 2, self.max_paragraph_len - answer_length // 2)
            else:
                rnd = self.max_paragraph_len // 2
            paragraph_start = max(0, min(mid - rnd, len(tokenized_paragraph) - self.max_paragraph_len))
            #paragraph_start = max(0, min(mid - self.max_paragraph_len // 2, len(tokenized_paragraph) - self.max_paragraph_len))
            
            paragraph_end = paragraph_start + self.max_paragraph_len

            # Slice question/paragraph and add special tokens (101: CLS, 102: SEP)
            input_ids_question = [101] + tokenized_question.ids[:self.max_question_len] + [102] 
            input_ids_paragraph = tokenized_paragraph.ids[paragraph_start : paragraph_end] + [102]		
            
            # Convert answer's start/end positions in tokenized_paragraph to start/end positions in the window  
            answer_start_token += len(input_ids_question) - paragraph_start
            answer_end_token += len(input_ids_question) - paragraph_start
            
            # Pad sequence and obtain inputs to model 
            input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)
            return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), answer_start_token, answer_end_token

        # Validation/Testing
        else:
            input_ids_list, token_type_ids_list, attention_mask_list = [], [], []
            
            # Paragraph is split into several windows, each with start positions separated by step "doc_stride"
            for i in range(0, len(tokenized_paragraph), self.doc_stride):
                
                # Slice question/paragraph and add special tokens (101: CLS, 102: SEP)
                input_ids_question = [101] + tokenized_question.ids[:self.max_question_len] + [102]
                input_ids_paragraph = tokenized_paragraph.ids[i : i + self.max_paragraph_len] + [102]
                
                # Pad sequence and obtain inputs to model
                input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)
                
                input_ids_list.append(input_ids)
                token_type_ids_list.append(token_type_ids)
                attention_mask_list.append(attention_mask)
            
            return torch.tensor(input_ids_list), torch.tensor(token_type_ids_list), torch.tensor(attention_mask_list)

    def padding(self, input_ids_question, input_ids_paragraph):
        # Pad zeros if sequence length is shorter than max_seq_len
        padding_len = self.max_seq_len - len(input_ids_question) - len(input_ids_paragraph)
        # Indices of input sequence tokens in the vocabulary
        input_ids = input_ids_question + input_ids_paragraph + [0] * padding_len
        # Segment token indices to indicate first and second portions of the inputs. Indices are selected in [0, 1]
        token_type_ids = [0] * len(input_ids_question) + [1] * len(input_ids_paragraph) + [0] * padding_len
        # Mask to avoid performing attention on padding token indices. Mask values selected in [0, 1]
        attention_mask = [1] * (len(input_ids_question) + len(input_ids_paragraph)) + [0] * padding_len
        
        return input_ids, token_type_ids, attention_mask

train_set = QA_Dataset("train", train_questions, train_questions_tokenized, train_paragraphs_tokenized)
dev_set = QA_Dataset("dev", dev_questions, dev_questions_tokenized, dev_paragraphs_tokenized)
test_set = QA_Dataset("test", test_questions, test_questions_tokenized, test_paragraphs_tokenized)



#references: 
# https://github.com/Singyuan/Machine-Learning-NTUEE-2022/blob/master/hw7/hw7.ipynb

train_batch_size = 4
doc_stride = train_set.doc_stride

# dataloader
# Although batch size=1, it is actually a batch consisting of several windows from the same QA pair
train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, pin_memory=True)
dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)

#%%

#references: https://github.com/Singyuan/Machine-Learning-NTUEE-2022/blob/master/hw7/hw7.ipynb

dev_paragraphs_offsets = dev_paragraphs_tokenized["offset_mapping"]
test_paragraphs_offsets = test_paragraphs_tokenized["offset_mapping"]


def index_before_tokenize_dev(idx, start_index, end_index, k):
    start = start_index - len(dev_questions[idx]['question_text']) - 2 + k * doc_stride
    end = end_index - len(dev_questions[idx]['question_text']) - 2 + k * doc_stride
    
    new_start = dev_paragraphs_offsets[dev_questions[idx]['paragraph_id']][start]
    new_end = dev_paragraphs_offsets[dev_questions[idx]['paragraph_id']][end]
    return new_start[0], new_end[1]
     
def index_before_tokenize_test(idx, start_index, end_index, k):
    start = start_index - len(test_questions[idx]['question_text']) - 2 + k * doc_stride
    end = end_index - len(test_questions[idx]['question_text']) - 2 + k * doc_stride
    
    new_start = test_paragraphs_offsets[test_questions[idx]['paragraph_id']][start]
    new_end = test_paragraphs_offsets[test_questions[idx]['paragraph_id']][end]
    return new_start[0], new_end[1]


#%%

#references: https://github.com/pai4451/ML2021/blob/main/hw7/ensemble_fusion859.ipynb

def evaluate(data, output1, output2, output3, idx, split='test'):
    ##### TODO: Postprocessing #####
    # There is a bug and room for improvement in postprocessing 
    # Hint: Open your prediction file to see what is wrong 
    
    answer = ''
    max_prob = float('-inf')
    num_of_windows = data[0].shape[1]
    start_logits = (output1.start_logits + output2.start_logits + output3.start_logits) / 3
    end_logits = (output1.end_logits + output2.end_logits + output3.end_logits) / 3
    
    for k in range(num_of_windows):
        # Obtain answer by choosing the most probable start position / end position
        start_prob, start_index = torch.max(start_logits[k], dim=0)
        end_prob, end_index = torch.max(end_logits[k], dim=0)
        
        # Probability of answer is calculated as sum of start_prob and end_prob
        prob = start_prob + end_prob
        
        # Replace answer if calculated probability is larger than previous windows
        if prob > max_prob and (end_index > start_index):
            max_prob = prob
            # Convert tokens to chars (e.g. [1920, 7032] --> "大 金")
            answer = tokenizer.decode(data[0][0][k][start_index : end_index + 1])
            record_k = k

        if answer.find('[UNK]') != -1:
            if split == 'dev':
                new_start, new_end = index_before_tokenize_dev(idx, start_index, end_index, record_k)
                answer = dev_paragraphs[dev_questions[idx]['paragraph_id']][new_start:new_end]

            if split == 'test':
                new_start, new_end = index_before_tokenize_test(idx, start_index, end_index, record_k)
                answer = test_paragraphs[test_questions[idx]['paragraph_id']][new_start:new_end]
    
    # Remove spaces in answer (e.g. "大 金" --> "大金")
    answer = answer.replace('[CLS]','')
    answer = answer[answer.find('[SEP]')+1:]
    return answer.replace(' ','')


#%%

model1 = AutoModelForQuestionAnswering.from_pretrained("saved_model/model1").to(device)
model2 = AutoModelForQuestionAnswering.from_pretrained("saved_model/model2").to(device)
model3 = AutoModelForQuestionAnswering.from_pretrained("saved_model/model3").to(device)

#%%

result = []

model1.eval()
model2.eval()
model3.eval()

with torch.no_grad():
    for i, data in enumerate(tqdm(test_loader)):
        output1 = model1(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device),
                       attention_mask=data[2].squeeze(dim=0).to(device))
        output2 = model2(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device),
                       attention_mask=data[2].squeeze(dim=0).to(device))
        output3 = model3(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device),
                       attention_mask=data[2].squeeze(dim=0).to(device))
        
        result.append(evaluate(data, output1, output2, output3, i))
        
result_file = f"results/d11948002_hw7_{_exp}.csv"
with open(result_file, 'w') as f:	
    f.write("ID,Answer\n")
    for i, test_question in enumerate(test_questions):
    # Replace commas in answers with empty strings (since csv is separated by comma)
    # Answers in kaggle are processed in the same way
        f.write(f"{test_question['id']},{result[i].replace(',','')}\n")

print(f"Completed! Result is in {result_file}")

#%%




#%%




#%%




#%%