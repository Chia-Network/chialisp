import io
import os
import shlex
import sys
import unittest
from chialisp import call_tool, launch_tool


# If the REPAIR environment variable is set, any tests failing due to
# wrong output will be corrected. Be sure to do a "git diff" to validate that
# you're getting changes you expect.

REPAIR = os.getenv("REPAIR", 0)


def get_test_cases(path):
    PREFIX = os.path.dirname(__file__)
    TESTS_PATH = os.path.join(PREFIX, path)
    paths = []
    for dirpath, dirnames, filenames in os.walk(TESTS_PATH):
        for fn in filenames:
            if fn.endswith(".txt") and fn[0] != ".":
                paths.append(os.path.join(dirpath, fn))
    paths.sort()
    test_cases = []
    with open("test_cases.txt", "w") as t:
        t.write("test_name, cmd_lines, expected_output[0], expected_stderr[0]\n")
    for p in paths:
        with open(p) as f:
            # allow "#" comments at the beginning of the file
            cmd_lines = []
            comments = []
            while 1:
                line = f.readline().rstrip()
                if len(line) < 1 or line[0] != "#":
                    if line[-1:] == "\\":
                        cmd_lines.append(line[:-1])
                        continue
                    cmd_lines.append(line)
                    break
                comments.append(line + "\n")
            lines = f.readlines()
            expected_outputs = []
            expected_stderrs = []
            for line in lines:
                line = line.strip()
                if line.startswith("stderr:"):
                    expected_stderrs.append(line[7:])
                else:
                    expected_outputs.append(line)
            test_name = os.path.relpath(p, PREFIX).replace(".", "_").replace("/", "_")
            expected_output = "\n".join(expected_outputs)
            expected_stderr = "\n".join(expected_stderrs)
            test_cases.append((test_name, cmd_lines, expected_output, expected_stderr, comments, p))
            with open("test_cases.txt", "a") as t:
                expected_out = expected_output.replace("\n", "\\n")
                expected_err = expected_stderr.replace("\n", "\\n")
                t.write(f'{test_name}, {cmd_lines}, {expected_out}, {expected_err}\n')
    return test_cases


class TestCmds(unittest.TestCase):
    def invoke_tool(self, cmd_line):

        args = shlex.split(cmd_line)

        default_stage = 0
        if args[0] == 'run':
            default_stage = 2

        if args[0] == 'run' or args[0] == 'brun':
            r = launch_tool(
                args[0],
                args,
                default_stage
            )
        else:
            r = call_tool(
                args[0],
                args
            )

        print("r", r)
        exit_code, stdout, stderr = r
        return exit_code, bytes(stdout).decode('utf8'), bytes(stderr).decode('utf8')


def make_f(cmd_lines, expected_stdout_param, expected_stderr_param, comments, path):
    def f(self):
        cmd = "".join(cmd_lines)
        for c in cmd.split(";"):
            r, actual_output, actual_stderr = self.invoke_tool(c)
        actual_output = actual_output.strip()
        expected_stdout = expected_stdout_param#.strip()
        expected_stderr = expected_stderr_param#.strip()
        if actual_stderr != expected_stderr:
            print("--------------------------------")
            print("path={path}")
            print("cmd={cmd}")
            print("comments={comments}")
            stdout_msg = f"expected_stdout={expected_stdout} actual_output={actual_output}"
            stderr_msg = f"expected_stderr={expected_stderr} actual_stderr={actual_stderr}"
            self.assertEqual(expected_stdout, actual_output, stdout_msg)
            self.assertEqual(expected_stderr, actual_stderr, stderr_msg)
            print("--------------------------------")
    return f

def inject(*paths):
    for path in paths:
        for idx, (name, i, o, s, comments, path) in enumerate(get_test_cases(path)):
            print(idx, (name, i, o, s, comments, path))
            name_of_f = "test_%s" % name
            print("name_of_f",name_of_f)
            setattr(TestCmds, name_of_f, make_f(i, o, s, comments, path))


inject("opc")

inject("opd")

inject("stage_1")

inject("stage_2")

inject("clvm_runtime")

inject("cmd")

# inject("v0_0_2")


def main():
    unittest.main()


if __name__ == "__main__":
    main()


"""
Copyright 2018 Chia Network Inc

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
