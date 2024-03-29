import os
import json
import shutil
import subprocess
from pathlib import Path
import distill_coverage_results

env = dict(os.environ)
env['RUSTFLAGS'] = "-C instrument-coverage"
env['LLVM_PROFILE_FILE'] = "your_name-%p-%m.profraw"

coverage_file = 'coverage.json'

def get_profile_files(the_dir):
    return list(filter(lambda x: x.endswith('.profraw'), os.listdir(the_dir)))

def delete_coverage_files():
    for the_file in get_profile_files('.'):
        if the_file.endswith('.profraw'):
            os.unlink(the_file)

    try:
        os.unlink(coverage_file)
    except:
        pass

def run_coverage_test():
    subprocess.check_call(['cargo','test'],env=env)

def is_my_code(desc):
    for path in ['.cargo','library/std','src/py','src/classic/bins','/rustc']:
        if path in desc['name']:
            return False

    return True

def collect_coverage():
    profile_files = get_profile_files('.')
    args = ['grcov','--binary-path','target/debug','--output-type','covdir','--output-path',coverage_file] + profile_files
    subprocess.check_call(args)
    result = distill_coverage_results.CoverageResult()

    infile = json.loads(open(coverage_file).read())
    for root in infile['children'].keys():
        distill_coverage_results.distill_coverage(result, Path(root), infile['children'][root])
    return list(filter(is_my_code, result.show_result()))

if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('--require-percent', help="required percentage to succeed", default=None)
    args = parser.parse_args()

    delete_coverage_files()
    run_coverage_test()
    result = collect_coverage()
    print(json.dumps(result, indent=4))

    if args.require_percent is not None:
        required_percentage = int(args.require_percent)
        has_required_pct = True

        for file in filter(lambda f: '/tests/' not in f['name'], result):
            have_pct = int(file['coveragePercent'])
            if have_pct < required_percentage:
                print(f"{file['name']} lacks required coverage have {have_pct} want {required_percentage}")
                has_required_pct = False

        if not has_required_pct:
            sys.exit(1)

