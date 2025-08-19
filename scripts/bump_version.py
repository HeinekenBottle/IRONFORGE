import re
import sys
import pathlib
import subprocess


def main():
    if len(sys.argv) != 2:
        print("usage: bump_version.py <new-version>")
        sys.exit(1)
    new = sys.argv[1]
    vf = pathlib.Path("ironforge/__version__.py")
    txt = vf.read_text(encoding="utf-8")
    new_txt = re.sub(r'__version__\s*=\s*".*?"', f'__version__ = "{new}"', txt)
    if txt == new_txt:
        print("version unchanged")
        return
    vf.write_text(new_txt, encoding="utf-8")
    subprocess.check_call(["git", "add", str(vf)])
    subprocess.check_call(["git", "commit", "-m", f"chore: bump version to {new}"])
    subprocess.check_call(["git", "tag", f"v{new}"])
    print("Bumped to", new)


if __name__ == "__main__":
    main()

