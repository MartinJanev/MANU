"""
Check that a new output file has exactly the key structure of a reference file.

Usage:  python3 check_schema.py results/radfm_ms_processed.jsonl reference/radfm_ms_processed.jsonl
Prints every key path that is missing or extra (e.g. output.parsed_response.ms_type) and
exits with 1 if there is any difference. Records with status "error" are skipped
(their parsed_response is null by design).
"""
import json
import sys


def key_paths(d, prefix=""):
    paths = set()
    for k, v in d.items():
        paths.add(prefix + k)
        if isinstance(v, dict):
            paths |= key_paths(v, prefix + k + ".")
    return paths


def schema(path):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    ok = [r for r in rows if r.get("output", {}).get("status") != "error"]
    print(f"{path}: {len(rows)} rows, {len(rows) - len(ok)} errors")
    return set().union(*(key_paths(r) for r in ok))


if __name__ == "__main__":
    new, ref = schema(sys.argv[1]), schema(sys.argv[2])
    for p in sorted(ref - new):
        print("missing:", p)
    for p in sorted(new - ref):
        print("extra:  ", p)
    print("same structure" if new == ref else "DIFFERENT structure")
    sys.exit(new != ref)
