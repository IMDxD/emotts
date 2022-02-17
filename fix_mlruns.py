from pathlib import Path


mlpath = Path("mlruns")
new_path = str(Path(__file__).parent)
if not new_path.startswith("/"):
    new_path = "/" + new_path.replace("\\", "/")
old_path = "/home/uadmin/Projects/made-emotts-2021"
for file in mlpath.rglob("*.yaml"):
    with open(file) as f:
        text = f.read()
    text = text.replace(old_path, new_path)
    with open(file, "w") as f:
        f.write(text)
