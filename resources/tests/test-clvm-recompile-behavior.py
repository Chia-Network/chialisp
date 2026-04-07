import os
import sys
import shutil
import stat
import time
import hashlib
from chialisp import compile_clvm

TESTDIR = 'testdir'
COMPILED_HASH = '6956b7c3af37e220819dc6e4919e7a1058167192db8eec81714218c4c3d8286d'
SOURCE_FILE = './resources/tests/did_innerpuz.clsp'
TARGET_FILE = './testdir/did_innerpuz.clvm.hex'
SEARCH_PATHS = ['./resources/tests']

def hash_file(path):
    hasher = hashlib.sha256()
    hasher.update(open(path).read().encode('utf8'))
    return hasher.hexdigest()

def test_compile_clvm(source, output, paths, want_sha256):
    compile_clvm(source, output, paths)
    if want_sha256 is not None:
        have_digest = hash_file(output)
        assert want_sha256 == have_digest

def update_file(fname):
    source_content = open(fname).read()
    time.sleep(5.0)
    # Rewrite file.
    with open(fname,'w') as f:
        f.write(source_content)

def try_delete_dir(path):
    try:
        os.chmod(path, stat.S_IREAD | stat.S_IWRITE | stat.S_IXUSR)
        shutil.rmtree(path)
    except:
        pass

def lock_file(testdir, path):
    os.chmod(path, stat.S_IREAD)
    os.chmod(testdir, stat.S_IREAD | stat.S_IXUSR)

def unlock_file(testdir, path):
    os.chmod(testdir, stat.S_IREAD | stat.S_IWRITE | stat.S_IXUSR)
    os.chmod(path, stat.S_IREAD | stat.S_IWRITE)

try_delete_dir(TESTDIR)
os.mkdir(TESTDIR)

# Get baseline date
source_file_date = os.stat(SOURCE_FILE).st_mtime

# Verify that we recompile when the target file isn't present.
test_compile_clvm(SOURCE_FILE, TARGET_FILE, SEARCH_PATHS, COMPILED_HASH)

hex_mod_date = os.stat(TARGET_FILE).st_mtime
assert hex_mod_date > source_file_date

# Update the source file.
update_file(SOURCE_FILE)
source_file_updated_date = os.stat(SOURCE_FILE).st_mtime
assert source_file_updated_date > hex_mod_date
assert source_file_updated_date > source_file_date

# Ensure that we updated the date.
test_compile_clvm(SOURCE_FILE, TARGET_FILE, SEARCH_PATHS, COMPILED_HASH)

updated_hex_mod_date = os.stat(TARGET_FILE).st_mtime
assert updated_hex_mod_date > hex_mod_date
assert updated_hex_mod_date > source_file_updated_date

time.sleep(5.0)
# Mess with the target file, ensure it gets recompiled.
with open(TARGET_FILE,'a') as f:
    f.write('0a0a0a0a0a')

update_file(SOURCE_FILE)
test_compile_clvm(SOURCE_FILE, TARGET_FILE, SEARCH_PATHS, COMPILED_HASH)
# Coming out of here means the new hash matched.
second_updated_hex_mod_date = os.stat(TARGET_FILE).st_mtime

# Update the hex file again and protect it.
update_file(TARGET_FILE)
lock_file(TESTDIR, TARGET_FILE)
hex_date_when_protected = os.stat(TARGET_FILE).st_mtime

update_file(SOURCE_FILE)
test_compile_clvm(SOURCE_FILE, TARGET_FILE, SEARCH_PATHS, COMPILED_HASH)
third_updated_hex_mod_date = os.stat(TARGET_FILE).st_mtime
assert third_updated_hex_mod_date == hex_date_when_protected

# Unprotect and modify.  Compilation should throw because the content won't be
# what's expected.
unlock_file(TESTDIR, TARGET_FILE)
time.sleep(5.0)
# Mess with the target file, ensure it gets recompiled.
with open(TARGET_FILE,'a') as f:
    f.write('0a0a0a0a0a')
lock_file(TESTDIR, TARGET_FILE)

# Update the time on the source file so we'll try to recompile
update_file(SOURCE_FILE)

# Check that we got the wrong hash because the file was modified.
try:
    test_compile_clvm(SOURCE_FILE, TARGET_FILE, SEARCH_PATHS, None)
    have_digest = hash_file(TARGET_FILE)
    assert False
except:
    # We threw because the file wasn't right but we couldn't overwrite it.
    assert COMPILED_HASH != hash_file(TARGET_FILE)

# Set it writable and recompile.
unlock_file(TESTDIR, TARGET_FILE)
test_compile_clvm(SOURCE_FILE, TARGET_FILE, SEARCH_PATHS, COMPILED_HASH)

last_source_mtime = os.stat(SOURCE_FILE).st_mtime
last_hex_mtime = os.stat(TARGET_FILE).st_mtime
assert last_hex_mtime > last_source_mtime
