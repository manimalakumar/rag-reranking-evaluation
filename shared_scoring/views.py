"""Notebook views of saved data; no scoring/model actions are invoked."""
from html import escape
from .io import read_json


def show_question(inputs, question_id, arms):
    """Display both saved answers and each arm's separate five-passage context."""
    from IPython.display import display, Markdown, HTML
    selected = [row for row in inputs if row['query_id'] == str(question_id)]
    if not selected:
        raise ValueError(f'Question {question_id} is not in these saved inputs.')
    display(Markdown('**Question:** ' + escape(selected[0]['question'])))
    display(Markdown('**Reference answers:** ' + '; '.join(escape(ref) for ref in selected[0]['reference_answers'])))
    for arm, label in arms.items():
        matches = [row for row in selected if row['arm'] == arm]
        if not matches:
            display(Markdown(f'**{label}:** saved answer unavailable'))
            continue
        row = matches[0]
        display(Markdown('#### ' + label))
        display(Markdown(f"Finish reason: `{row['finish_reason']}` · Truncated: **{row['truncated']}**"))
        display(HTML('<pre style="white-space:pre-wrap">' + escape(row['answer']) + '</pre>'))
        display(HTML('<details><summary>This arm’s exact saved passages</summary><pre style="white-space:pre-wrap">' +
                     escape(row['context']) + '</pre></details>'))


def show_saved_request(path, label):
    """Show a saved request and raw response without sending the request."""
    import json
    from IPython.display import display, HTML
    record = read_json(path)
    display(HTML('<details><summary>' + escape(label) + '</summary><pre style="white-space:pre-wrap">' +
                 escape(json.dumps(record, indent=2, ensure_ascii=False)) + '</pre></details>'))
