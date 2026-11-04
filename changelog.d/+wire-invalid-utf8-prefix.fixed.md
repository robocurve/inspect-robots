**Wire capture:** keep all valid JSONL rows before an invalid UTF-8 entry when
reading saved captures. The CLI and HTML viewer no longer hide earlier calls
because the text decoder encountered corruption in a later line.
