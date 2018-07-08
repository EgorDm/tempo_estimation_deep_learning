def make_safe_filename(s):
    def safe_char(c):
        if c.isalnum():
            return c
        if c in ' -':
            return c
        else:
            return "_"
    return "".join(safe_char(c) for c in s).rstrip("_")