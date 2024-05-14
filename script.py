from time import time
import openai
from tiktoken import encoding_for_model
import threading
from datetime import datetime
from sklearn.linear_model import LinearRegression

PROMPT = "Let's count! 1 2 3"

GPT5_tokenizer = encoding_for_model("gpt-4o")
GPT4_tokenizer = encoding_for_model("gpt-4")
GPT3_tokenizer = encoding_for_model("davinci")


TINY_MODEL_MAX_TOKS = [10, 50, 100, 150, 200, 250]
SMALL_MODEL_MAX_TOKS = [10, 50, 100, 150]
BIG_MODEL_MAX_TOKS = [10, 50, 100]

CHAT = True

model_infos = {
    # "ada": (TINY_MODEL_MAX_TOKS, GPT3_tokenizer, not CHAT),
    # "babbage": (TINY_MODEL_MAX_TOKS, GPT3_tokenizer, not CHAT),
    # "curie": (TINY_MODEL_MAX_TOKS, GPT3_tokenizer, not CHAT),
    # "davinci": (BIG_MODEL_MAX_TOKS, GPT3_tokenizer, not CHAT),
    "gpt-3.5-turbo": (TINY_MODEL_MAX_TOKS, GPT4_tokenizer, CHAT),
    "gpt-3.5-turbo-instruct": (TINY_MODEL_MAX_TOKS, GPT4_tokenizer, not CHAT),
    "gpt-4": (BIG_MODEL_MAX_TOKS, GPT4_tokenizer, CHAT),
    "gpt-4-1106-preview": (BIG_MODEL_MAX_TOKS, GPT4_tokenizer, CHAT),
    "gpt-3.5-turbo-1106": (TINY_MODEL_MAX_TOKS, GPT4_tokenizer, CHAT),
    "gpt-4-turbo-preview": (SMALL_MODEL_MAX_TOKS, GPT4_tokenizer, CHAT),
    "gpt-4o": (SMALL_MODEL_MAX_TOKS, GPT5_tokenizer, CHAT),
    "davinci-002": (SMALL_MODEL_MAX_TOKS, GPT4_tokenizer, not CHAT),
    "babbage-002": (TINY_MODEL_MAX_TOKS, GPT4_tokenizer, not CHAT),
}
models = list(model_infos.keys())


def complete(engine: str, prompt: str, max_tokens: int = 100):
    tokenizer = model_infos[engine][1]

    # to avoid early eos
    logit_bias = {t: 100 for i in range(1, 280 + 1) for t in tokenizer.encode(" " + str(i))}

    st = time()
    if model_infos[engine][2]:
        completion = openai.ChatCompletion.create(
            model=engine,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0,
            logit_bias=logit_bias,
        ).choices[0]["message"]["content"]
    else:
        completion = (
            openai.Completion.create(
                engine=engine, prompt=prompt, max_tokens=max_tokens, temperature=0, logit_bias=logit_bias
            )
            .choices[0]
            .text
        )
    taken = time() - st
    tokens_in_answer = len(tokenizer.encode(completion))

    return tokens_in_answer, taken


def measure():
    lock = threading.Lock()

    results = {model: [] for model in models}

    def target(model, max_tokens):
        tokens_in_answer, taken = complete(model, PROMPT, max_tokens)
        with lock:
            results[model].append((taken, tokens_in_answer))

    threads = []
    for model in models:
        for max_tokens in model_infos[model][0]:
            t = threading.Thread(target=target, args=(model, max_tokens))
            threads.append(t)
            t.start()
    for t in threads:
        t.join()

    def fit_classifier(results: list[tuple[float, float]]):
        x = [r[1] for r in results]
        y = [r[0] for r in results]
        lr = LinearRegression()
        lr.fit([[z] for z in x], y)
        return lr

    classifiers = {
        model: fit_classifier(results[model]) for model in models if len(results[model]) == len(model_infos[model][0])
    }
    kept_models = list(classifiers.keys())
    slopes = {model: classifiers[model].coef_[0] for model in kept_models}
    intercepts = {model: classifiers[model].intercept_ for model in kept_models}
    for model in kept_models:
        second_per_tok = slopes[model]
        tok_per_second = 1 / second_per_tok
        print(f"{model}: {tok_per_second:.2f} tok/s")

    return results, classifiers, slopes, intercepts, kept_models


results, classifiers, slopes, intercepts, kept_models = measure()

time_str = str(datetime.now())
with open("time_mes.csv", "a") as f:
    for model in kept_models:
        line = ",".join([time_str, model, str(1 / slopes[model]), str(intercepts[model])])
        f.write(line + "\n")
