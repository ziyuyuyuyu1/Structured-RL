import re
import json

def split_sentences_keep_blocks(text):
    # Protect math/code blocks
    patterns = [
        (r'```.*?```', re.DOTALL),   # fenced code blocks
        (r'\\\[.*?\\\]', re.DOTALL), # display math
        (r'\\\(.*?\\\)', re.DOTALL), # inline math
        (r'\$.*?\$', re.DOTALL),     # inline math $...$
    ]
    placeholders = []
    protected = text
    for i, (pat, flags) in enumerate(patterns):
        matches = list(re.finditer(pat, protected, flags))
        for m in reversed(matches):
            token = f"__BLOCK_{i}_{len(placeholders)}__"
            placeholders.append((token, m.group(0)))
            start, end = m.span()
            protected = protected[:start] + token + protected[end:]
    # Split sentences
    split_regex = re.compile(r'(?<=[\.\?\!])\s+(?=[A-Z\\`])')
    parts = [p.strip() for p in split_regex.split(protected) if p.strip()]
    # Restore placeholders
    for i, sentence in enumerate(parts):
        for token, original in placeholders:
            sentence = sentence.replace(token, original)
        parts[i] = sentence
    return parts

def split_keep_dot(text):
    # This pattern splits at a '.' followed by whitespace or newline, but keeps the '.' attached to the sentence.
    pattern = r'(?<=\.)[ \n]+'
    parts = re.split(pattern, text)
    # Strip each part to clean extra spaces/newlines
    parts = [p.strip() for p in parts if p.strip()]
    return parts

input_path = "verl_few_shot/Qwen2.5-Math-1.5B-dsr_sub/eval/global_step_0/math500/test_qwen25-math-cot_-1_seed0_t0.7_s0_e-1.jsonl"         # path to your existing results file
output_path = "eval_results_sentences.jsonl"

with open(input_path, "r", encoding="utf-8") as fin, \
    open(output_path, "w", encoding="utf-8") as fout:
    for line in fin:
        sample = json.loads(line)
        code_sentences = []
        for answer in sample.get("code", []):
            answer_text = str(answer)
            code_sentences.append(split_keep_dot(answer_text))
        sample["code_sentences"] = code_sentences
        fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
        break


print(f"Processed results saved to {output_path}")