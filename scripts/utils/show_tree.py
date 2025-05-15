import os

EXCLUDE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.zip'}
MAX_DEPTH = 5  # 조정 가능

def show_tree(root_path, prefix="", depth=0):
    if depth > MAX_DEPTH:
        return

    items = sorted(os.listdir(root_path))
    items = [item for item in items if not any(item.lower().endswith(ext) for ext in EXCLUDE_EXTS)]

    for i, item in enumerate(items):
        path = os.path.join(root_path, item)
        connector = "└── " if i == len(items) - 1 else "├── "

        print(prefix + connector + item)

        if os.path.isdir(path):
            extension = "    " if i == len(items) - 1 else "│   "
            show_tree(path, prefix + extension, depth + 1)

if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    print(f"📂 Directory Tree (excluding images): {root}\n")
    show_tree(root)
