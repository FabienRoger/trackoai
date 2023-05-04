from time import time
import openai
from tiktoken import encoding_for_model
import threading
from datetime import datetime
from sklearn.linear_model import LinearRegression

PROMPT = "Let's count to 1.000.000! 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16"

GPT4_tokenizer = encoding_for_model("gpt-4")
GPT3_tokenizer = encoding_for_model("davinci")


def complete(engine: str, prompt: str, max_tokens: int = 100):

    if engine in ["gpt-3.5-turbo", "gpt-4"]:
        completion = openai.ChatCompletion.create(
            model=engine, messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens, temperature=0
        ).choices[0]["message"]["content"]
        tokens_in_answer = len(GPT4_tokenizer.encode(completion))
    else:
        completion = (
            openai.Completion.create(engine=engine, prompt=prompt, max_tokens=max_tokens, temperature=0).choices[0].text
        )
        tokens_in_answer = len(GPT3_tokenizer.encode(completion))
    return tokens_in_answer


models = ["ada", "babbage", "curie", "davinci", "text-davinci-003", "gpt-3.5-turbo", "gpt-4"]


def measure():
    lock = threading.Lock()

    results = {model: [] for model in models}

    def target(model, max_tokens):
        st = time()
        tokens_in_answer = complete(model, PROMPT, max_tokens)
        taken = time() - st
        with lock:
            results[model].append((taken, tokens_in_answer))

    threads = []
    for model in models:
        for max_tokens in [10, 50, 90]:
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

    classifiers = {model: fit_classifier(results[model]) for model in models}
    slopes = {model: classifiers[model].coef_[0] for model in models}
    intercepts = {model: classifiers[model].intercept_ for model in models}
    for model in models:
        second_per_tok = slopes[model]
        tok_per_second = 1 / second_per_tok
        print(f"{model}: {tok_per_second:.2f} tok/s")

    return results, classifiers, slopes, intercepts


results, classifiers, slopes, intercepts = measure()

time_str = str(datetime.now())
with open("time_mes.csv", "a") as f:
    for model in models:
        line = ",".join([time_str, model, str(1 / slopes[model]), str(intercepts[model])])
        f.write(line + "\n")
