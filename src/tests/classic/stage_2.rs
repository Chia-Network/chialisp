use clvmr::allocator::Allocator;
use std::collections::HashMap;
use std::fs;
use std::rc::Rc;

use crate::classic::clvm::__type_compatibility__::Stream;
use crate::classic::clvm::sexp::rest;
use crate::classic::clvm_tools::binutils::{assemble, assemble_from_ir, disassemble};
use crate::classic::clvm_tools::clvmc::compile_clvm_text;
use crate::classic::clvm_tools::cmds::call_tool;
use crate::classic::clvm_tools::ir::reader::read_ir;
use crate::classic::clvm_tools::stages::stage_2::compile::{
    do_com_prog, get_compile_filename, get_last_path_component, try_expand_macro_for_atom,
};
use crate::classic::clvm_tools::stages::stage_2::helpers::{brun, evaluate, quote, run};
use crate::classic::clvm_tools::stages::stage_2::operators::run_program_for_search_paths;
use crate::classic::clvm_tools::stages::stage_2::reader::{process_embed_file, read_file};

use crate::compiler::compiler::DefaultCompilerOpts;
use crate::compiler::comptypes::{CompileErr, CompilerOpts, HasCompilerOptsDelegation};
use crate::compiler::sexp::decode_string;
use crate::compiler::srcloc::Srcloc;

fn test_expand_macro(
    allocator: &mut Allocator,
    macro_data: String,
    prog_rest: String,
    macros: String,
    symbols: String,
) -> String {
    let macro_ir = read_ir(&macro_data).unwrap();
    let macro_source = assemble_from_ir(allocator, Rc::new(macro_ir)).unwrap();
    let prog_ir = read_ir(&prog_rest).unwrap();
    let prog_source = assemble_from_ir(allocator, Rc::new(prog_ir)).unwrap();
    let macros_ir = read_ir(&macros).unwrap();
    let macros_source = assemble_from_ir(allocator, Rc::new(macros_ir)).unwrap();
    let symbols_ir = read_ir(&symbols).unwrap();
    let symbols_source = assemble_from_ir(allocator, Rc::new(symbols_ir)).unwrap();
    let exp_res = try_expand_macro_for_atom(
        allocator,
        macro_source,
        prog_source,
        macros_source,
        symbols_source,
    )
    .unwrap();
    disassemble(allocator, exp_res.1, Some(0))
}

fn test_inner_expansion(
    allocator: &mut Allocator,
    macro_code: String,
    prog_rest: String,
) -> String {
    let macro_ir = read_ir(&macro_code).unwrap();
    let macro_source = assemble_from_ir(allocator, Rc::new(macro_ir)).unwrap();
    let prog_ir = read_ir(&prog_rest).unwrap();
    let prog_source = assemble_from_ir(allocator, Rc::new(prog_ir)).unwrap();
    let exp_res = brun(allocator, macro_source, prog_source).unwrap();
    disassemble(allocator, exp_res, Some(0))
}

fn test_do_com_prog(
    allocator: &mut Allocator,
    program_src: String,
    macro_lookup_src: String,
    symbol_table_src: String,
) -> String {
    let runner = run_program_for_search_paths("*test*", &vec![".".to_string()], false);
    let prog_ir = read_ir(&program_src).unwrap();
    let program = assemble_from_ir(allocator, Rc::new(prog_ir)).unwrap();
    let macro_ir = read_ir(&macro_lookup_src).unwrap();
    let macro_lookup = assemble_from_ir(allocator, Rc::new(macro_ir)).unwrap();
    let sym_ir = read_ir(&symbol_table_src).unwrap();
    let symbol_table = assemble_from_ir(allocator, Rc::new(sym_ir)).unwrap();
    let result = do_com_prog(allocator, 849, program, macro_lookup, symbol_table, runner).unwrap();
    disassemble(allocator, result.1, Some(0))
}

#[test]
fn test_macro_expansion() {
    let mut allocator = Allocator::new();
    let res = test_expand_macro(
        &mut allocator,
        "(c (q . \"list\") (c (f 1) (c (c (q . \"mod\") (c (f (r 1)) (c (f (r (r 1))) (q)))) (q))))".to_string(),
        "(\"function\" (\"BODY\") (29041 (\"opt\" (\"com\" (q \"unquote\" \"BODY\") (29041 (\"unquote\" (\"macros\"))) (29041 (\"unquote\" (\"symbols\")))))))".to_string(),
        "((\"list\" (a (q 2 (q 2 2 (c 2 (c 3 (q)))) (c (q 2 (i 5 (q 4 (q . 4) (c 9 (c (a 2 (c 2 (c 13 (q)))) (q)))) (q 1)) 1) 1)) 1)) (\"defmacro\" (c (q . \"list\") (c (f 1) (c (c (q . \"mod\") (c (f (r 1)) (c (f (r (r 1))) (q)))) (q))))))".to_string(),
        "()".to_string()
    );
    assert_eq!(res, "(a (\"com\" (a (q 4 (q . \"list\") (c (f 1) (c (c (q . \"mod\") (c (f (r 1)) (c (f (r (r 1))) (q)))) (q)))) (q \"function\" (\"BODY\") (29041 (\"opt\" (\"com\" (q \"unquote\" \"BODY\") (29041 (\"unquote\" (\"macros\"))) (29041 (\"unquote\" (\"symbols\")))))))) (q (\"list\" (a (q 2 (q 2 2 (c 2 (c 3 (q)))) (c (q 2 (i 5 (q 4 (q . 4) (c 9 (c (a 2 (c 2 (c 13 (q)))) (q)))) (q 1)) 1) 1)) 1)) (\"defmacro\" (c (q . \"list\") (c (f 1) (c (c (q . \"mod\") (c (f (r 1)) (c (f (r (r 1))) (q)))) (q)))))) (q)) 1)".to_string());
}

#[test]
fn test_inner_macro_exp() {
    let mut allocator = Allocator::new();
    let res = test_inner_expansion(
        &mut allocator,
        "(c (q . \"list\") (c (f 1) (c (c (q . \"mod\") (c (f (r 1)) (c (f (r (r 1))) (q)))) (q))))".to_string(),
        "(\"function\" (\"BODY\") (29041 (\"opt\" (\"com\" (q \"unquote\" \"BODY\") (29041 (\"unquote\" (\"macros\"))) (29041 (\"unquote\" (\"symbols\")))))))".to_string()
    );
    assert_eq!(res, "(a (q 4 (q . \"list\") (c (f 1) (c (c (q . \"mod\") (c (f (r 1)) (c (f (r (r 1))) (q)))) (q)))) (q \"function\" (\"BODY\") (29041 (\"opt\" (\"com\" (q \"unquote\" \"BODY\") (29041 (\"unquote\" (\"macros\"))) (29041 (\"unquote\" (\"symbols\"))))))))".to_string());
}

#[test]
fn test_compile_during_assert_1() {
    let mut allocator = Allocator::new();
    let res = test_do_com_prog(
        &mut allocator,
        "(\"assert\" 1)".to_string(),
        "((\"assert\" (a (i 3 (q 4 (q . 26982) (c 2 (c (c (q . \"assert\") 3) (q (x))))) (q . 2)) 1)) (26982 (c (q . 2) (c (c (q . 3) (c 2 (c (c (q . \"function\") (c 5 ())) (c (c (q . \"function\") (c 11 ())) ())))) (q 64)))) (\"function\" (c (q . \"opt\") (c (c (q . \"com\") (c (c (q . 1) 2) (q (29041 (\"unquote\" (\"macros\"))) (29041 (\"unquote\" (\"symbols\")))))) ()))) (\"list\" (a (q 2 (q 2 2 (c 2 (c 3 (q)))) (c (q 2 (i 5 (q 4 (q . 4) (c 9 (c (a 2 (c 2 (c 13 (q)))) (q)))) (q 1)) 1) 1)) 1)) (\"defmacro\" (c (q . \"list\") (c (f 1) (c (c (q . \"mod\") (c (f (r 1)) (c (f (r (r 1))) (q)))) (q))))))".to_string(),
        "()".to_string()
    );

    assert_eq!(res, "(a (\"com\" (a (q 2 (i 3 (q 4 (q . 26982) (c 2 (c (c (q . \"assert\") 3) (q (x))))) (q . 2)) 1) (q 1)) (q (\"assert\" (a (i 3 (q 4 (q . 26982) (c 2 (c (c (q . \"assert\") 3) (q (x))))) (q . 2)) 1)) (26982 (c (q . 2) (c (c (q . 3) (c 2 (c (c (q . \"function\") (c 5 ())) (c (c (q . \"function\") (c 11 ())) ())))) (q 64)))) (\"function\" (c (q . \"opt\") (c (c (q . \"com\") (c (c (q . 1) 2) (q (29041 (\"unquote\" (\"macros\"))) (29041 (\"unquote\" (\"symbols\")))))) ()))) (\"list\" (a (q 2 (q 2 2 (c 2 (c 3 (q)))) (c (q 2 (i 5 (q 4 (q . 4) (c 9 (c (a 2 (c 2 (c 13 (q)))) (q)))) (q 1)) 1) 1)) 1)) (\"defmacro\" (c (q . \"list\") (c (f 1) (c (c (q . \"mod\") (c (f (r 1)) (c (f (r (r 1))) (q)))) (q)))))) (q)) 1)".to_string());
}

#[test]
fn test_compile_check_output_diag_assert() {
    let mut allocator = Allocator::new();
    let res = test_do_com_prog(
        &mut allocator,
        "(\"mod\" () (\"defmacro\" \"assert\" \"items\" (26982 (r \"items\") (\"list\" 26982 (f \"items\") (c \"assert\" (r \"items\")) (q 8)) (f \"items\"))) (\"assert\" 1))".to_string(),
        "((26982 (c (q . 2) (c (c (q . 3) (c 2 (c (c (q . \"function\") (c 5 ())) (c (c (q . \"function\") (c 11 ())) ())))) (q 64)))) (\"function\" (c (q . \"opt\") (c (c (q . \"com\") (c (c (q . 1) 2) (q (29041 (\"unquote\" (\"macros\"))) (29041 (\"unquote\" (\"symbols\")))))) ()))) (\"list\" (a (q 2 (q 2 2 (c 2 (c 3 (q)))) (c (q 2 (i 5 (q 4 (q . 4) (c 9 (c (a 2 (c 2 (c 13 (q)))) (q)))) (q 1)) 1) 1)) 1)) (\"defmacro\" (c (q . \"list\") (c (f 1) (c (c (q . \"mod\") (c (f (r 1)) (c (f (r (r 1))) (q)))) (q))))))".to_string(),
        "()".to_string()
    );

    assert_eq!(
        res,
        "(a (q \"opt\" (q 2 (\"opt\" (\"com\" (q \"assert\" 1) (q (\"assert\" (a (i 3 (q 4 (q . 26982) (c 2 (c (c (q . \"assert\") 3) (q (x))))) (q . 2)) 1)) (26982 (c (q . 2) (c (c (q . 3) (c 2 (c (c (q . \"function\") (c 5 ())) (c (c (q . \"function\") (c 11 ())) ())))) (q 64)))) (\"function\" (c (q . \"opt\") (c (c (q . \"com\") (c (c (q . 1) 2) (q (29041 (\"unquote\" (\"macros\"))) (29041 (\"unquote\" (\"symbols\")))))) ()))) (\"list\" (a (q 2 (q 2 2 (c 2 (c 3 (q)))) (c (q 2 (i 5 (q 4 (q . 4) (c 9 (c (a 2 (c 2 (c 13 (q)))) (q)))) (q 1)) 1) 1)) 1)) (\"defmacro\" (c (q . \"list\") (c (f 1) (c (c (q . \"mod\") (c (f (r 1)) (c (f (r (r 1))) (q)))) (q)))))) (q))) 1)) 1)".to_string()
    );
}

#[test]
fn test_compile_assert_2() {
    let mut allocator = Allocator::new();
    let res = test_do_com_prog(
        &mut allocator,
        "(\"mod\" () (\"defmacro\" \"assert\" \"items\" (26982 (r \"items\") (\"list\" 26982 (f \"items\") (c \"assert\" (r \"items\")) (q 8)) (f \"items\"))) (\"assert\" 1))".to_string(),
        "((26982 (c (q . 2) (c (c (q . 3) (c 2 (c (c (q . \"function\") (c 5 ())) (c (c (q . \"function\") (c 11 ())) ())))) (q 64)))) (\"function\" (c (q . \"opt\") (c (c (q . \"com\") (c (c (q . 1) 2) (q (29041 (\"unquote\" (\"macros\"))) (29041 (\"unquote\" (\"symbols\")))))) ()))) (\"list\" (a (q 2 (q 2 2 (c 2 (c 3 (q)))) (c (q 2 (i 5 (q 4 (q . 4) (c 9 (c (a 2 (c 2 (c 13 (q)))) (q)))) (q 1)) 1) 1)) 1)) (\"defmacro\" (c (q . \"list\") (c (f 1) (c (c (q . \"mod\") (c (f (r 1)) (c (f (r (r 1))) (q)))) (q))))))".to_string(),
        "()".to_string()
    );

    assert_eq!(
        res,
        "(a (q \"opt\" (q 2 (\"opt\" (\"com\" (q \"assert\" 1) (q (\"assert\" (a (i 3 (q 4 (q . 26982) (c 2 (c (c (q . \"assert\") 3) (q (x))))) (q . 2)) 1)) (26982 (c (q . 2) (c (c (q . 3) (c 2 (c (c (q . \"function\") (c 5 ())) (c (c (q . \"function\") (c 11 ())) ())))) (q 64)))) (\"function\" (c (q . \"opt\") (c (c (q . \"com\") (c (c (q . 1) 2) (q (29041 (\"unquote\" (\"macros\"))) (29041 (\"unquote\" (\"symbols\")))))) ()))) (\"list\" (a (q 2 (q 2 2 (c 2 (c 3 (q)))) (c (q 2 (i 5 (q 4 (q . 4) (c 9 (c (a 2 (c 2 (c 13 (q)))) (q)))) (q 1)) 1) 1)) 1)) (\"defmacro\" (c (q . \"list\") (c (f 1) (c (c (q . \"mod\") (c (f (r 1)) (c (f (r (r 1))) (q)))) (q)))))) (q))) 1)) 1)".to_string()
    );
}

#[test]
fn test_stage_2_quote() {
    let mut allocator = Allocator::new();
    let assembled = assemble(&mut allocator, "(1 2 3)").unwrap();
    let quoted = quote(&mut allocator, assembled).unwrap();
    assert_eq!(disassemble(&mut allocator, quoted, Some(0)), "(q 1 2 3)");
}

#[test]
fn test_stage_2_evaluate() {
    let mut allocator = Allocator::new();
    let prog = assemble(&mut allocator, "(q 16 2 3)").unwrap();
    let args = assemble(&mut allocator, "(q 9 15)").unwrap();
    let to_eval = evaluate(&mut allocator, prog, args).unwrap();
    assert_eq!(
        disassemble(&mut allocator, to_eval, Some(0)),
        "(a (q 16 2 3) (q 9 15))"
    );
}

#[test]
fn test_stage_2_run() {
    let mut allocator = Allocator::new();
    let prog = assemble(&mut allocator, "(q 16 2 3)").unwrap();
    let macro_lookup_throw = assemble(&mut allocator, "(q 9)").unwrap();
    let to_eval = run(&mut allocator, prog, macro_lookup_throw).unwrap();
    assert_eq!(
        disassemble(&mut allocator, to_eval, Some(0)),
        "(a (\"com\" (q 16 2 3) (q 1 9)) 1)"
    );
}

#[test]
fn test_present_file_smoke_not_exists() {
    let mut allocator = Allocator::new();
    let runner =
        run_program_for_search_paths("*test*", &vec!["resources/tests".to_string()], false);
    let sexp_triggering_read = assemble(&mut allocator, "(embed-file test-file sexp embed.sexp)")
        .expect("should assemble");
    let res = read_file(
        runner,
        &mut allocator,
        sexp_triggering_read,
        "test-embed-not-exist.clsp",
    );
    assert!(res.is_err());
}

#[test]
fn test_present_file_smoke_exists() {
    let mut allocator = Allocator::new();
    let runner =
        run_program_for_search_paths("*test*", &vec!["resources/tests".to_string()], false);
    let sexp_triggering_read = assemble(&mut allocator, "(embed-file test-file sexp embed.sexp)")
        .expect("should assemble");
    let res = read_file(runner, &mut allocator, sexp_triggering_read, "embed.sexp")
        .expect("should exist");
    assert_eq!(decode_string(&res.data), "(23 24 25)");
}

#[test]
fn test_process_embed_file_as_sexp() {
    let mut allocator = Allocator::new();
    let runner =
        run_program_for_search_paths("*test*", &vec!["resources/tests".to_string()], false);
    let declaration_sexp = assemble(&mut allocator, "(embed-file test-embed sexp embed.sexp)")
        .expect("should assemble");
    let want_exp = assemble(&mut allocator, "(q 23 24 25)").expect("should assemble");
    let (name, content) =
        process_embed_file(&mut allocator, runner, declaration_sexp).expect("should work");
    assert_eq!(
        disassemble(&mut allocator, want_exp, Some(0)),
        disassemble(&mut allocator, content, Some(0))
    );
    assert_eq!(name, b"test-embed");
}

/// A test where a file is in an unexpected location was requested.
/// This test tries to read resources/tests/steprun/fact.clvm.hex but specifies
/// resources/tests/stage_2 as an include path.
#[test]
fn test_process_embed_file_as_sexp_in_an_unexpected_location() {
    let mut allocator = Allocator::new();
    let runner = run_program_for_search_paths(
        "*test*",
        &vec!["resources/tests/stage_2".to_string()],
        false,
    );
    let sexp_triggering_read = assemble(&mut allocator, "(embed-file test-file hex act.clvm.hex)")
        .expect("should assemble");
    let res = read_file(
        runner,
        &mut allocator,
        sexp_triggering_read,
        "fact.clvm.hex",
    );
    assert!(res.is_err());
}

/// Read hex test mirror of the above.
#[test]
fn test_process_embed_file_as_sexp_in_an_expected_location() {
    let mut allocator = Allocator::new();
    let runner = run_program_for_search_paths(
        "*test*",
        &vec!["resources/tests/steprun".to_string()],
        false,
    );
    let sexp_triggering_read = assemble(&mut allocator, "(embed-file test-file hex act.clvm.hex)")
        .expect("should assemble");
    let res = read_file(
        runner,
        &mut allocator,
        sexp_triggering_read,
        "fact.clvm.hex",
    )
    .expect("should exist");
    let real_file_content =
        fs::read_to_string("resources/tests/steprun/fact.clvm.hex").expect("should exist");
    assert_eq!(res.data, real_file_content.as_bytes().to_vec());
}

#[test]
fn test_process_embed_file_as_hex() {
    let mut allocator = Allocator::new();
    let runner =
        run_program_for_search_paths("*test*", &vec!["resources/tests".to_string()], false);
    let declaration_sexp = assemble(
        &mut allocator,
        "(embed-file test-embed-from-hex hex steprun/fact.clvm.hex)",
    )
    .expect("should assemble");
    let (name, content) =
        process_embed_file(&mut allocator, runner, declaration_sexp).expect("should work");
    let matching_part_of_decl = rest(&mut allocator, content).expect("should be quoted");
    let mut outstream = Stream::new(None);
    call_tool(
        &mut outstream,
        &mut allocator,
        "opd",
        &[
            "opd".to_string(),
            "resources/tests/steprun/fact.clvm.hex".to_string(),
        ],
    )
    .expect("should work");
    assert_eq!(
        disassemble(&mut allocator, matching_part_of_decl, Some(0)),
        decode_string(outstream.get_value().data())
    );
    assert_eq!(name, b"test-embed-from-hex");
}

#[derive(Clone)]
struct TestCompilerOptsPresentsOwnFiles {
    files: Rc<HashMap<String, String>>,
    opts: Rc<dyn CompilerOpts>,
}

impl TestCompilerOptsPresentsOwnFiles {
    fn new(filename: String, files: HashMap<String, String>) -> Self {
        TestCompilerOptsPresentsOwnFiles {
            files: Rc::new(files),
            opts: Rc::new(DefaultCompilerOpts::new(&filename)),
        }
    }
}

impl HasCompilerOptsDelegation for TestCompilerOptsPresentsOwnFiles {
    fn compiler_opts(&self) -> Rc<dyn CompilerOpts> {
        self.opts.clone()
    }

    fn update_compiler_opts<F: FnOnce(Rc<dyn CompilerOpts>) -> Rc<dyn CompilerOpts>>(
        &self,
        f: F,
    ) -> Rc<dyn CompilerOpts> {
        let new_opts = f(self.opts.clone());
        Rc::new(TestCompilerOptsPresentsOwnFiles {
            opts: new_opts,
            ..self.clone()
        })
    }

    fn override_read_new_file(
        &self,
        inc_from: String,
        filename: String,
    ) -> Result<(String, Vec<u8>), CompileErr> {
        if let Some(content) = self.files.get(&filename) {
            return Ok((filename.clone(), content.bytes().collect()));
        }

        Err(CompileErr(
            Srcloc::start(&inc_from),
            format!("could not read {filename}"),
        ))
    }
}

// Shows that we can inject a compiler opts and have it provide file data.
// This allows the compiler opts to provide abstract filesystem plumbing even
// in the classic compiler when requested.
#[test]
fn test_classic_compiler_with_compiler_opts() {
    let files_vec: Vec<(String, String)> = vec![("test.clinc", "( (defun F (X) (+ X 1)) )")]
        .iter()
        .map(|(n, v)| (n.to_string(), v.to_string()))
        .collect();
    let mut files = HashMap::new();
    for (k, v) in files_vec.into_iter() {
        files.insert(k, v);
    }
    let opts = Rc::new(TestCompilerOptsPresentsOwnFiles::new(
        "test.clsp".to_string(),
        files,
    ));
    let to_compile = "(mod (A) (include test.clinc) (F A))";
    let mut allocator = Allocator::new();
    let mut symbols = HashMap::new();
    // Verify injection
    let result = compile_clvm_text(
        &mut allocator,
        opts.clone(),
        &mut symbols,
        to_compile,
        "test.clsp",
        true,
    )
    .expect("should compile and find the content");
    assert_eq!(
        disassemble(&mut allocator, result, Some(0)),
        "(a (q 2 2 (c 2 (c 5 ()))) (c (q 16 5 (q . 1)) 1))"
    );
    // Verify lack of injection
    let result_no_injection = compile_clvm_text(
        &mut allocator,
        opts,
        &mut symbols,
        to_compile,
        "test.clsp",
        false,
    );
    assert!(result_no_injection.is_err());
}

#[test]
fn test_classic_runner_has_compile_filename() {
    let mut allocator = Allocator::new();
    let use_filename = "test-classic-runner-has-compile-filename.clsp";
    let runner = run_program_for_search_paths(use_filename, &vec![], false);

    let result_filename = get_compile_filename(runner, &mut allocator)
        .expect("should be able to tell us the file name given")
        .expect("should have returned some");

    assert_eq!(result_filename, use_filename);
}

#[test]
fn test_get_last_path_component_0() {
    let last_path_component = get_last_path_component("test/foo/bar.clsp");
    assert_eq!(last_path_component, "bar.clsp");
}
