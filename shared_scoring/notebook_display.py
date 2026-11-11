"""Compact notebook presentation; all rows remain available without new inference."""
from html import escape
import json
import re
import pandas as pd


def table_html(frame, preview_rows=1, threshold=8):
    """Show a short preview and expandable remaining rows, with full cell text."""
    def render(part):
        with pd.option_context('display.max_colwidth',None):
            return part.to_html(index=False, escape=True, na_rep='Unavailable',
                                float_format=lambda value: f'{value:.4f}')
    if any(c in frame.columns for c in ['answer','squad_answer']):threshold=1
    if len(frame) <= threshold:
        return render(frame)
    return (render(frame.head(preview_rows)) + '<details><summary>Show remaining '
            + str(len(frame)-preview_rows) + ' rows (' + str(len(frame)) + ' total)</summary>'
            + render(frame.iloc[preview_rows:]) + '</details>')


def display_table(frame):
    from IPython.display import HTML, display
    display(HTML(table_html(frame)))


def presentation_value(value):
    """Omit integrity digests from teaching displays without changing source records."""
    if isinstance(value, dict):
        return {key:presentation_value(item) for key,item in value.items()
                if not re.search(r'(?:^|_)(?:sha(?:256|512)?|checksums?|hash(?:es)?)(?:$|_)',str(key),re.I)}
    if isinstance(value, (list,tuple)):
        return [presentation_value(item) for item in value]
    return value


def details_html(label, value):
    value=presentation_value(value)
    body = (table_html(value) if isinstance(value, pd.DataFrame) else
            '<pre style="white-space:pre-wrap;overflow-wrap:anywhere">'
            + escape(value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False, default=str)) + '</pre>')
    return '<details><summary>'+escape(label)+'</summary>'+body+'</details>'


from contextlib import contextmanager


def preview_panel_html(body, label, preview_rows=1, prefer_logs=False):
    """Leave the first table rows visible and collapse all remaining output."""
    match = re.search(r'(<table\b[^>]*>.*?<tbody\b[^>]*>)(.*?)(</tbody>\s*</table>)', body, re.S)
    if prefer_logs:match=None
    if not match:
        log = re.search(r'(<pre\b[^>]*>)(.*?)(</pre>)',body,re.S)
        if log:
            lines=log.group(2).splitlines(keepends=True)
            preview=body[:log.start()]+log.group(1)+''.join(lines[:preview_rows])+log.group(3)
            remaining=log.group(1)+''.join(lines[preview_rows:])+log.group(3)+body[log.end():]
            return preview+'<details><summary>'+escape(label)+'</summary>'+remaining+'</details>'
        return '<details><summary>' + escape(label) + '</summary>' + body + '</details>'
    rows = re.findall(r'<tr\b[^>]*>.*?</tr>', match.group(2), re.S)
    if not rows:
        return '<details><summary>' + escape(label) + '</summary>' + body + '</details>'
    preview = body[:match.start()] + match.group(1) + ''.join(rows[:preview_rows]) + match.group(3)
    remaining = (match.group(1) + ''.join(rows[preview_rows:]) + match.group(3)) if len(rows)>preview_rows else ''
    remaining += body[match.end():]
    return preview + '<details><summary>' + escape(label) + '</summary>' + remaining + '</details>'


@contextmanager
def collapsible_output(label, preview_rows=0, preview_logs=False):
    """Retain stdout, stderr and rich displays in one closed expandable panel."""
    from IPython.display import HTML, display
    from IPython.utils.capture import capture_output
    captured = None
    try:
        with capture_output() as captured:
            yield
    finally:
        if captured is not None:
            body = ''
            for text in [captured.stdout, captured.stderr]:
                if text:
                    body += '<pre style="white-space:pre-wrap;overflow-wrap:anywhere">' + escape(text) + '</pre>'
            for output in captured.outputs:
                data = output.data
                if 'text/html' in data:
                    body += data['text/html']
                else:
                    text = data.get('text/markdown', data.get('text/plain', ''))
                    body += '<pre style="white-space:pre-wrap;overflow-wrap:anywhere">' + escape(text) + '</pre>'
            html = preview_panel_html(body,label,preview_rows,prefer_logs=preview_logs) if preview_rows else '<details><summary>' + escape(label) + '</summary>' + body + '</details>'
            display(HTML(html))
