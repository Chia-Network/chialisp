from chialisp import binutils
from clvm_tools.binutils import assemble, disassemble
from clvm_rs.program import Program
from random import randint

test_cases = [
    "(q)",
    "()",
    "(1 . 2)",
    "(() . ())",
    "(1 2 3 . ())",
    "(1 2 3)",
    "((4 5 6) (7 8 9))",
    "(i (q . 1) (q . 2) (q . 3))",
    "((0x33 . i))",
    "(g . (138935604050210 ()))",
    "0x0001",
    "0x00",
    "0x5c4859"
]

def extend_atom():
    n_choice = randint(0,14)
    if n_choice > 10:
        return ""
    return f"{chr(n_choice+97)}@"

def pad(n,v,s):
    if len(s) < n:
        return "".join(([v]*(n - len(s)))) + s
    else:
        return s

def extend_hex():
    n_choice = randint(0,400)
    if n_choice > 255:
        return ""
    return f"{pad(2,'0',hex(n_choice)[2:])}^"

def any_item():
    keys = list(filter(lambda k: k not in "&^", production_rules.keys()))
    key_choice = randint(0,len(keys) - 1)
    return keys[key_choice]

production_rules = {
    "!": lambda: "()",
    "~": lambda: "(*&",
    "&": lambda: [")", " *&", " . *)"][randint(0,2)],
    "#": lambda: str(randint(0,2**48) - (2**47)),
    "@": lambda: f"{chr(randint(0,10)+97)}{extend_atom()}",
    "$": lambda: f"0x{pad(2,'0',hex(randint(0,255))[2:])}^",
    "^": extend_hex,
    "*": any_item
}
production_keys = "".join(production_rules.keys())

def do_replacement(x):
    if x in production_rules:
        return production_rules[x]()
    else:
        return x

def generate_random_sexp(s):
    def generate_round(s):
        return "".join([do_replacement(x) for x in s])

    s = generate_round(s)
    while any([x in production_keys for x in s]):
        s = generate_round(s)

    return s

for i in range(20):
    test_cases.append(generate_random_sexp("*"))

for t in test_cases:
    print(t)
    expected = assemble(t)
    assembled = binutils.assemble_generic(lambda x,y: Program.to((x,y)), Program.to, t)
    print(expected, assembled)

    assert expected == assembled

    disassembled = binutils.disassemble_generic(bytes(assembled))
    expected_dis = disassemble(expected)
    assert expected_dis == disassembled
