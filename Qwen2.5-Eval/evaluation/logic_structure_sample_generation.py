# rm -rf math_eval.py; vim math_eval.py
import random
import os
import argparse
import time
from vllm import LLM, SamplingParams
from datetime import datetime
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from evaluate import evaluate
from utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from parser import *
from trajectory import *
from data_loader import load_data
from python_executor import PythonExecutor
from model_utils import load_hf_lm_and_tokenizer, generate_completions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="tool-integrated", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=8192, type=int)
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--vllm_batch_size",default=0, type=int)
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply chat template to prompt.",
    )
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument(
        "--adapt_few_shot",
        action="store_true",
        help="Few shot for multiple-choice questions, zero shot for others.",
    )
    args = parser.parse_args()
    args.top_p = (
        1 if args.temperature == 0 else args.top_p
    )  # top_p must be 1 when using greedy sampling (vllm)
    return args

def setup(args):
    # load model
    available_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    print(f"Using GPUs: {available_gpus}")
    if args.use_vllm:
        llm = LLM(
            model=args.model_name_or_path,
            tensor_parallel_size=len(available_gpus) // args.pipeline_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            trust_remote_code=True,
        )
        tokenizer = None
        if args.apply_chat_template:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path, trust_remote_code=True
            )
    else:
        llm, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            load_in_half=True,
            use_fast_tokenizer=True,
            use_safetensors=args.use_safetensors,
        )

    
    main(llm, tokenizer, args)


def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True


def main(llm, tokenizer, args):
    print("=" * 50)

    # init python executor
    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr="solution()")
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    samples = []
    idx = 0
    example = {}
    # example["question"] = "Compute the area of a triangle with side lengths $3$, $4$, $5$."
    example["question"] = "Find the roots of the cubic equation $x^3 - 2x^2 - 5x + 6 = 0$."
    example["gt_ans"] = "1, -2, 3"
    full_prompt = construct_prompt(example, "math500", args)

    if idx == args.start:
        print(full_prompt)

    sample = {
        "idx": idx,
        "question": example["question"],
        "gt_cot": None,
        "gt": example["gt_ans"],
        "prompt": full_prompt,
    }

    # add remain fields
    for key in [
        "level",
        "type",
        "unit",
        "solution_type",
        "choices",
        "solution",
        "ques_type",
        "ans_type",
        "answer_type",
        "dataset",
        "subfield",
        "filed",
        "theorem",
        "answer",
        "difficulty"
    ]:
        if key in example:
            sample[key] = example[key]
    sample_1 = sample.copy()
    # sample_1["prompt"] = sample_1["prompt"] + "The side lengths satisfy $5^2 = 3^2 + 4^2$. By the converse of the Pythagorean theorem, the triangle is right-angled with the side of length $5$ as the hypotenuse."
    sample_1["prompt"] = sample_1["prompt"] + "Test possible rational roots by plugging in factors of the constant term ($\pm 1$, $\pm 2$, $\pm 3$, $\pm 6$) and find that $x = 1$ is a root."
    
    sample_2 = sample.copy()
    sample_2["prompt"] = sample_2["prompt"] + "Check values among the divisors of the constant term and determine that substituting $x = 1$ zeroes the polynomial."
    
    sample_3 = sample.copy()
    # sample_2["prompt"] = sample_2["prompt"] + "Since $3^2 + 4^2 = 5^2$, the angle opposite the side of length $5$ measures $90^\circ$ (i.e. $\pi/2$ radians), so the triangle is right-angled with legs $3$ and $4$."
    sample_3["prompt"] = sample_3["prompt"] + "Confirm  $x = 1$  is a root by synthetic division of the polynomial by $(x - 1)$, yielding a quadratic quotient."
    
    sample_4 = sample.copy()
    # sample_3["prompt"] = sample_3["prompt"] + "Compute the semiperimeter $s = \frac{3+4+5}{2} = 6$, then apply Heron’s formula $A = \sqrt{s(s-3)(s-4)(s-5)}$."
    sample_4["prompt"] = sample_4["prompt"] + "We can use Cardano’s cubic formula here."

    samples.append(sample_1)
    samples.append(sample_2)
    samples.append(sample_3)
    samples.append(sample_4)

    # repeat n times
    # You can add some extra text to the prompt to continue the reasoning process, but remember to save the extra text in the output.
    input_prompts = [
        # sample["prompt"] + "blablabla" for sample in samples for _ in range(args.n_sampling)
        sample["prompt"] for sample in samples for _ in range(args.n_sampling)
    ]
    if args.apply_chat_template:
        input_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt.strip()}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in input_prompts
        ]
    remain_prompts = input_prompts
    remain_prompts = [(i, prompt) for i, prompt in enumerate(remain_prompts)]
    end_prompts = []

    max_func_call = 1 if args.prompt_type in ["cot", "pal"] else 4

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>", "."]

    if args.prompt_type in ["cot"]:
        stop_words.append("\n\nQuestion:")
    if args.prompt_type in ["pal", "tool-integrated", "jiuzhang_tora"]:
        stop_words.extend(["\n\n---", "```output"])
    elif args.prompt_type in ["wizard_zs", "platypus_fs"]:
        stop_words.extend(["Instruction", "Response"])
    elif "jiuzhang" in args.prompt_type:
        stop_words.append("\n\n## Question")
    elif "numina" in args.prompt_type:
        stop_words.append("\n### Problem")
    elif "pure" in args.prompt_type:
        stop_words.append("\n\n\n")

    # start inference
    # measure time use
    start_time = time.time()
    result_prompts = []
    for epoch in range(max_func_call):
        print("-" * 20, "Epoch", epoch)
        current_prompts = remain_prompts
        if len(current_prompts) == 0:
            break

        # get all outputs
        prompts = [item[1] for item in current_prompts]
        result_prompts.extend(prompts)
        if args.use_vllm:
            if args.vllm_batch_size:
                outputs = []
                for i in range(0, len(prompts), args.vllm_batch_size):
                    batch_prompts = prompts[i:i+args.vllm_batch_size]
                    
                    batch_outputs = llm.generate(
                        batch_prompts,
                        SamplingParams(
                            temperature=args.temperature,
                            max_tokens=args.max_tokens_per_call,
                            stop=stop_words,
                            stop_token_ids=(
                                [151645, 151643]
                                if "qwen2" in args.model_name_or_path.lower()
                                else None
                            ),
                        ),
                    )
                    
                    batch_outputs = sorted(
                        batch_outputs, key=lambda x: int(x.request_id)
                    )
                    batch_outputs = [output.outputs[0].text for output in batch_outputs]
                    outputs.extend(batch_outputs)
            else:
                outputs = llm.generate(
                    prompts,
                    SamplingParams(
                        temperature=args.temperature,
                        # top_p=args.top_p,
                        max_tokens=args.max_tokens_per_call,
                        # n=1,
                        stop=stop_words,
                        stop_token_ids=(
                            [151645, 151643]
                            if "qwen2" in args.model_name_or_path.lower()
                            else None
                        ),
                    ),
                )

                outputs = sorted(
                    outputs, key=lambda x: int(x.request_id)
                )  # sort outputs by request_id
                outputs = [output.outputs[0].text for output in outputs]
        else:
            outputs = generate_completions(
                model=llm,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=args.max_tokens_per_call,
                batch_size=16,
                stop_id_sequences=stop_words,
            )

        assert len(outputs) == len(current_prompts)

        # process all outputs
        remain_prompts = []
        remain_codes = []
        for (i, query), output in zip(current_prompts, outputs):
            output = output.rstrip()
            query += output
            if args.prompt_type == "pal":
                remain_prompts.append((i, query))
                if "```python" in output:
                    output = extract_program(query)
                remain_codes.append(output)
            elif args.prompt_type == "cot":
                end_prompts.append((i, query))
            elif "boxed" not in output and output.endswith("```"):
                program = extract_program(query)
                remain_prompts.append((i, query))
                remain_codes.append(program)
            else:
                end_prompts.append((i, query))

        # execute the remain prompts
        remain_results = executor.batch_apply(remain_codes)
        for k in range(len(remain_prompts)):
            i, query = remain_prompts[k]
            res, report = remain_results[k]
            exec_result = res if res else report
            if "pal" in args.prompt_type:
                exec_result = "\\boxed{" + exec_result + "}"
            exec_result = f"\n```output\n{exec_result}\n```\n"
            query += exec_result
            # not end
            if epoch == max_func_call - 1:
                query += "\nReach max function call limit."
            remain_prompts[k] = (i, query)

    # unsolved samples
    print("Unsolved samples:", len(remain_prompts))
    end_prompts.extend(remain_prompts)
    # sort by idx
    end_prompts = sorted(end_prompts, key=lambda x: x[0])

    # remove input_prompt from end_prompt
    codes = []
    assert len(input_prompts) == len(end_prompts)
    for i in range(len(input_prompts)):
        _, end_prompt = end_prompts[i]
        code = end_prompt.split(input_prompts[i])[-1].strip()
        for stop_word in stop_words:
            if stop_word in code and stop_word != ".":
                code = code.split(stop_word)[0].strip()
        codes.append(code)

    # put results back to examples
    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i * args.n_sampling : (i + 1) * args.n_sampling]
        sample.pop("prompt")
        sample.update({"code": code})
        all_samples.append(sample)

    out_file_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
    output_dir = args.output_dir
    # if not os.path.exists(output_dir):
    #     output_dir = f"outputs/{output_dir}" # don't add this
    out_file = f"{output_dir}/{out_file_prefix}_s{args.start}_e{args.end}_logic_sentences.jsonl"
    os.makedirs(f"{output_dir}", exist_ok=True)

    # save outputs
    if args.save_outputs:
        save_jsonl(all_samples, out_file)


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)
