import ast
import re
import time

import tiktoken


def _compute_tokens_from_payload(payload, encoding):
    """
    Estimate the number of tokens required in the request.
    See: https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken

    :param payload: an array of dicts or an array of strings
    :param encoding: a tiktoken encoder instance
    :return: a count of tokens
    """
    num_tokens = 0
    for message in payload:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n

        # if we pass a wellformed payload
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))

            # if there's a name, the role is not processed and so doesn't count to api usage
            if key == "name":
                num_tokens += -1

                # pad the estimate with the structure of response; note: does not include actual response
    # every reply is primed with <im_start>assistant
    num_tokens += 2

    return num_tokens


def _compute_tokens_from_list(labels, encoding):
    """
    Estimate the number of tokens required in the request.
    See: https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken

    :param labels: a list of string labels
    :param encoding: a tiktoken encoder instance
    :return: a count of tokens
    """
    return [len(encoding.encode(l)) for l in labels]


def _create_payload(labels, prompt, role):
    return [
        {"role": "system",
         "content": role},
        {"role": "user", "content": f"{prompt} Here is the list of labels: {labels}"},
    ]


def _call_gpt(client, messages, model):
    return client.chat.completions.create(model=model, messages=messages)
    # return openai.ChatCompletion.create(model=model, messages=messages)


def _chunk_labels(labels, encoding, threshold, debug=False):
    """
    segment the data into sublists to not exceed api limits. Splitting text strings into tokens is useful because GPT
    models see text in the form of tokens. Knowing how many tokens are in a text string can tell you (a) whether the
    string is too long for a text model to process and (b) how much an OpenAI API call costs (as usage is priced by
    token).

    :param labels: feature inputs as a list of strings
    :param encoding: a tiktoken encoder instance
    :param threshold: a maximum token size to chunk the inputs into
    :return:
    """

    current_chunk, chunks = [], []
    count_tokens = 0
    label_tokens = _compute_tokens_from_list(labels, encoding)

    # step throw all labels
    for label, tokens in zip(labels, label_tokens):
        if debug:
            print(label)
            print(tokens)
        # append labels to the current chunk and update our count
        current_chunk.append(label)
        count_tokens += tokens

        # if the array exceeds the token threshold then....
        if count_tokens + 2 > threshold:

            # remove the last label
            hold = current_chunk.pop()

            # complain if the label itself is so big that it's as big as the threshold
            if tokens > threshold:
                raise Exception(f"Label {label} is too big: *{tokens} tokens* for this threshold: *{threshold} tokens*")

            # since we removed the offending label, the current chunk should be the right size
            chunks.append(current_chunk)

            # start a new arr with the one we popped out and reset our counter
            current_chunk = [hold]
            count_tokens = 0

    # add the final set of labels
    chunks.append(current_chunk)

    return chunks


def _process_result(result):
    """
    Filter out patterns that look like [' and ']
    TODO: why don't we just look at those patterns directly instead of regex?
    :param result: raw openAI API response
    :return: processed answers
    """
    answer = result.choices[0].message.content
    apostrophe_pattern = r"(?<=\w)'(?=[^,\]])|'(?=\w+?'\s)"
    answer = re.sub(apostrophe_pattern, '', answer)

    idx_open = answer.find("[")
    idx_close = answer.find(']') + 1  # +1 since indexing ignores current spot

    try:
        results = ast.literal_eval(answer[idx_open:idx_close])
    except Exception as e:
        print(e)
        results = answer

    return results


def get_gpt_alt_labels(labels, client, prompt, role, model='gpt-4o-mini', threshold=4000, num_retries=3, debug=True):
    """
    Call GPT to generate alternate labels.
    :param labels: list of human-annotated labels
    :param num_retries: how many times do we try again?
    :return: (a list of alternate labels, api call response status)
    """

    start = time.time()

    # it's probably clk100k_base (can pass to get_encoding()), but let's not assume
    encoding = tiktoken.encoding_for_model(model)
    prompt_size = _compute_tokens_from_payload(_create_payload([], prompt, role), encoding)

    # chunk the prompt to the right size; if we pass in everything, it's gonna timeout/fail
    threshold -= prompt_size
    chunks = _chunk_labels(labels, encoding, threshold=threshold)

    if debug:
        print(f'Prompt size: {prompt_size}')
        print(f"split time = {time.time() - start}")
        print("Number of loops: ", len(chunks))

    alt_labels = []

    for i, chunk in enumerate(chunks):
        tries = 0
        success = False
        start = time.time()

        if debug:
            print(f"Loop {i} running through array of size {len(labels)}")

        while not success and tries < num_retries:
            payload = _create_payload(chunk, prompt, role)
            result = _call_gpt(client, payload, model)
            alt_chunk = _process_result(result)

            n_labels = len(chunk)
            n_alts = len(alt_chunk)

            if n_labels == n_alts:
                success = True
            else:
                tries += 1

                print(f'FAILED. {n_labels} labels but {n_alts} alts!')
                print(f'attempting retry # {tries}')

        if success:
            alt_labels += alt_chunk
        else:
            print(f"failed completely after {num_retries} retries.")
            return

        if debug:
            print(f"api call time = {time.time() - start}")

    return alt_labels