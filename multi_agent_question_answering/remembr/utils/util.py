def file_to_string(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return file.read().strip()
