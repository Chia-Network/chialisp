import sys

from chialisp.clvm_tools_rs import cmds


# this dynamic bit of programming may look super-complicated,
# but it's just me being lazy and automating with a macro to
# produce code of the form:
#  brun = lambda: cmds.brun_main(sys.argv)
#  run = lambda: cmds.run_main(sys.argv)
# etc.


for cmd in "brun run opc opd cldb shrink clisp_to_json repl".split():
    code = f"{cmd} = lambda: cmds.{cmd}_main(sys.argv)"
    exec(code)
