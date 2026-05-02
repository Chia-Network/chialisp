import struct

def byte_of(w,n):
    shift = n * 8
    return (w >> shift) & 0xff

def bytes_of(w):
    return bytes(map(lambda n: byte_of(w,n), range(4)))

word_t = gdb.lookup_type('word').pointer()

class SExp:
    def __init__(self,ptr,**kwargs):
        self.ptr = ptr
        if 'atom' in kwargs:
            self.kind = 0
            self.atom = kwargs['atom']
        else:
            self.kind = 1
            self.cons = kwargs['cons']

    def is_cons(self):
        return self.kind

    def to_string(self):
        if self.kind == 0:
            if len(self.atom) == 0:
                return '()'
            else:
                return repr(self.atom)
        else:
            res = '('
            sep = ''
            examine = self
            proper_list = []

            while examine.is_cons():
                res += sep
                sep = ' '
                res += examine.cons[0].to_string()
                proper_list.append(examine.cons[0])
                examine = examine.cons[1]

            if len(examine.atom) != 0:
                res += ' . '
                res += examine.to_string()

            res += ')'

            return res

def read_inf_word(address):
    res = struct.unpack('<I', gdb.selected_inferior().read_memory(address, 4))[0]
    return res

def value(address):
    if address == 0:
        return SExp(0, atom=[])

    fval = read_inf_word(address)
    if fval & 1:
        the_len = fval >> 1
        num_words = (the_len + 3) >> 2
        words = []
        for v in range(num_words):
            words.append(read_inf_word(address + (v+1) * 4))
        final_bytes=b''.join(map(bytes_of,words))[:the_len]
        return SExp(address, atom=final_bytes)
    else:
        v1 = value(read_inf_word(address))
        v2 = value(read_inf_word(address+4))
        return SExp(address, cons=[v1,v2])

class WordPtrPrinter:
    def __init__(self, val):
        self.val = val

    def to_string(self):
        got_value = value(self.val)
        return got_value.to_string()

def lookup_type(val):
    if str(val.type) == 'word *' or str(val.type) == 'word' or str(val.type) == 'sexp':
        if val.address is not None:
            addr = int(val.address)
        else:
            addr = struct.unpack('<I', val.bytes)[0]
        return WordPtrPrinter(addr)
    return None

gdb.pretty_printers.append(lookup_type)
