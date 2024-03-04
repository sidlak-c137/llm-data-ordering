# Little script to compute hardness for gsm8k
import json
from datasets import load_dataset
from tqdm import tqdm

def get_steps_and_answer(a):
  steps = a.count('\n')
  marker_index = a.find("####")
  assert marker_index != -1, f'unable to find answer for the answer string {a}'
  final_answer = a[marker_index + 4:].strip()

  return steps, final_answer

def output_list(questions, answers):
  final_ret=[]
  for i in tqdm(range(len(questions))):
    steps, final_answer = get_steps_and_answer(answers[i])
    entry = {'question':questions[i], 'answer': answers[i], 'step': steps, 'answer': final_answer}
    final_ret.append(entry)
  return final_ret  

def output_jsonl(out_name, questions, answers):
  final_rest = output_list(questions, answers)
  with open(out_name, 'w') as outfile:
    for entry in final_rest:
        json.dump(entry, outfile)
        outfile.write('\n')



def main():
  dataset = load_dataset("gsm8k", "main")
  questions = dataset['train']['question']
  answers = dataset['train']['answer']
  
  output_jsonl("gsm8k-steps-answers.jsonl", questions, answers)


if __name__ == "__main__":
    main()