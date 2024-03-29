use num_bigint::ToBigInt;

#[cfg(test)]
use rand::distributions::Standard;
#[cfg(test)]
use rand::prelude::*;
#[cfg(test)]
use rand::Rng;
#[cfg(test)]
use rand_chacha::ChaChaRng;

use std::borrow::Borrow;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use std::rc::Rc;

use clvmr::allocator::Allocator;

use crate::classic::clvm::__type_compatibility__::{bi_one, bi_zero, Stream};
use crate::classic::clvm_tools::binutils::disassemble;
use crate::classic::clvm_tools::cmds::launch_tool;
use crate::classic::clvm_tools::node_path::NodePath;

use crate::compiler::clvm::convert_to_clvm_rs;
use crate::compiler::sexp;
use crate::compiler::sexp::decode_string;
use crate::util::{number_from_u8, Number};

const NUM_GEN_ATOMS: usize = 16;

pub fn do_basic_brun(args: &Vec<String>) -> String {
    let mut s = Stream::new(None);
    launch_tool(&mut s, args, &"run".to_string(), 0);
    return s.get_value().decode();
}

pub fn do_basic_run(args: &Vec<String>) -> String {
    let mut s = Stream::new(None);
    launch_tool(&mut s, args, &"run".to_string(), 2);
    return s.get_value().decode();
}

#[test]
fn basic_run_test() {
    assert_eq!(
        do_basic_run(&vec!("run".to_string(), "(mod (A B) (+ A B))".to_string())).trim(),
        "(+ 2 5)".to_string()
    );
}

#[test]
fn add_1_test() {
    assert_eq!(
        do_basic_run(&vec!(
            "run".to_string(),
            "(opt (com (q . (+ 6 55))))".to_string()
        ))
        .trim(),
        "(q . 61)".to_string()
    );
}

#[test]
fn div_test() {
    assert_eq!(
        do_basic_run(&vec!("run".to_string(), "(mod (X) (/ X 10))".to_string())).trim(),
        "(f (divmod 2 (q . 10)))".to_string()
    );
}

#[test]
fn brun_y_1_test() {
    let testpath = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut sym_path = testpath.clone();
    sym_path.push("resources/tests/stage_2/brun-y-1.sym");
    assert_eq!(
        do_basic_brun(
            &vec!(
                "brun".to_string(),
                "-y".to_string(),
                sym_path.into_os_string().into_string().unwrap(),
                "(a (q . (a 2 (c 2 (c 5 (q . ()))))) (c (q . (a (i (= 5 (q . 1)) (q . (q . 1)) (q . (* 5 (a 2 (c 2 (c (- 5 (q . 1)) (q . ()))))))) 1)) 1))".to_string(),
                "(10)".to_string()
            )
        ).trim(),
        indoc! {"0x375f00
            
            (\"fact\" 10) => 0x375f00
            
            (\"fact\" 9) => 0x058980
            
            (\"fact\" 8) => 0x009d80
            
            (\"fact\" 7) => 5040
            
            (\"fact\" 6) => 720
            
            (\"fact\" 5) => 120
            
            (\"fact\" 4) => 24
            
            (\"fact\" 3) => 6
            
            (\"fact\" 2) => 2
            
            (\"fact\" 1) => 1"}
    );
}

#[test]
fn brun_v_test() {
    assert_eq!(
        do_basic_brun(&vec!(
            "brun".to_string(),
            "-v".to_string(),
            "(a (q + (q . 3) (q . 5)) 1)".to_string()
        ))
        .trim(),
        indoc! {"8
            
            (a 2 3) [((a (q 16 (q . 3) (q . 5)) 1))] => 8
            
            3 [((a (q 16 (q . 3) (q . 5)) 1))] => ()
            
            2 [((a (q 16 (q . 3) (q . 5)) 1))] => (a (q 16 (q . 3) (q . 5)) 1)
            
            (a (q 16 (q . 3) (q . 5)) 1) [()] => 8
            
            1 [()] => ()
            
            (q 16 (q . 3) (q . 5)) [()] => (+ (q . 3) (q . 5))
            
            (+ (q . 3) (q . 5)) [()] => 8
            
            (q . 5) [()] => 5
            
            (q . 3) [()] => 3"}
    );
}

#[test]
fn brun_constant_test() {
    assert_eq!(
        do_basic_run(&vec!(
            "run".to_string(),
            "(mod () (defconstant X 3) X)".to_string()
        ))
        .trim(),
        "(q . 3)".to_string()
    );
}

#[test]
fn at_capture_destructure_1() {
    assert_eq!(
        do_basic_run(&vec!(
            "run".to_string(),
            "(mod (A (@ Z (B C)) D) A)".to_string()
        ))
        .trim(),
        "2"
    );
}

#[test]
fn at_capture_destructure_2() {
    assert_eq!(
        do_basic_run(&vec!(
            "run".to_string(),
            "(mod (A (@ Z (B C)) D) Z)".to_string()
        ))
        .trim(),
        "5"
    );
}

#[test]
fn at_capture_destructure_3() {
    assert_eq!(
        do_basic_run(&vec!(
            "run".to_string(),
            "(mod (A (@ Z (B C)) D) B)".to_string()
        ))
        .trim(),
        "9"
    );
}

#[test]
fn at_capture_destructure_4() {
    assert_eq!(
        do_basic_run(&vec!(
            "run".to_string(),
            "(mod (A (@ Z (B C)) D) C)".to_string()
        ))
        .trim(),
        "21"
    );
}

#[test]
fn at_capture_destructure_5() {
    assert_eq!(
        do_basic_run(&vec!(
            "run".to_string(),
            "(mod (A (@ Z (B C)) D) D)".to_string()
        ))
        .trim(),
        "11"
    );
}

#[test]
fn at_capture_inline_1() {
    assert_eq!(
        do_basic_run(&vec!(
            "run".to_string(),
            "(mod () (defun-inline F (@ pt (X Y)) X) (F 97 98))".to_string()
        ))
        .trim(),
        "(q . 97)"
    );
}

#[test]
fn at_capture_inline_2() {
    assert_eq!(
        do_basic_run(&vec!(
            "run".to_string(),
            "(mod () (defun-inline F (@ pt (X Y)) Y) (F 97 98))".to_string()
        ))
        .trim(),
        "(q . 98)"
    );
}

#[test]
fn at_capture_inline_3() {
    assert_eq!(
        do_basic_run(&vec!(
            "run".to_string(),
            "(mod () (defun-inline F (@ pt (X Y)) pt) (F (+ 117 1) (+ 98 1)))".to_string()
        ))
        .trim(),
        "(q 118 99)"
    );
}

#[test]
fn at_capture_inline_4() {
    assert_eq!(
        do_basic_run(&vec!(
            "run".to_string(),
            "(mod () (defun-inline F (A (@ pt (X Y))) (list (list A X Y) pt)) (F 115 (list 99 77)))".to_string()
        ))
            .trim(),
        "(q (115 99 77) (99 77))"
    );
}

#[test]
fn inline_destructure_1() {
    assert_eq!(
        do_basic_run(&vec!(
            "run".to_string(),
            "(mod () (defun-inline F ((A . B)) (+ A B)) (F (c 3 7)))".to_string()
        ))
        .trim(),
        "(q . 10)"
    );
}

#[test]
fn test_forms_of_destructuring_allowed_by_classic_1() {
    assert_eq!(
        do_basic_run(&vec![
            "run".to_string(),
            "(mod (A) (defun-inline foo (X Y . Z) (i X Y . Z)) (foo A 2 3))".to_string()
        ])
        .trim(),
        "(i 2 (q . 2) (q . 3))"
    );
}

fn run_dependencies(filename: &str) -> HashSet<String> {
    let result_text = do_basic_run(&vec![
        "run".to_string(),
        "-i".to_string(),
        "resources/tests".to_string(),
        "-M".to_string(),
        filename.to_owned(),
    ])
    .trim()
    .to_string();

    eprintln!("run_dependencies:\n{}", result_text);

    let mut dep_set = HashSet::new();
    for l in result_text.lines() {
        if let Some(suffix_start) = l.find("resources/tests") {
            let copied_suffix: Vec<u8> = l.as_bytes().iter().skip(suffix_start).copied().collect();
            dep_set.insert(decode_string(&copied_suffix));
        } else {
            panic!("file {} isn't expected", l);
        }
    }

    dep_set
}

#[test]
fn test_get_dependencies_1() {
    let dep_set = run_dependencies("resources/tests/singleton_top_layer.clvm");

    eprintln!("dep_set {dep_set:?}");

    let mut expect_set = HashSet::new();
    expect_set.insert("resources/tests/condition_codes.clvm".to_owned());
    expect_set.insert("resources/tests/curry-and-treehash.clinc".to_owned());
    expect_set.insert("resources/tests/singleton_truths.clib".to_owned());

    assert_eq!(dep_set, expect_set);
}

#[test]
fn test_treehash_constant_embedded_classic() {
    let result_text = do_basic_run(&vec![
        "run".to_string(),
        "-i".to_string(),
        "resources/tests".to_string(),
        indoc! {"
            (mod ()
              (include sha256tree.clib)
              (defconst H (+ G (sha256tree (q 2 3 4))))
              (defconst G 1)
              H
              )
        "}
        .to_string(),
    ])
    .trim()
    .to_string();
    assert_eq!(
        result_text,
        "(q . 0x6fcb06b1fe29d132bb37f3a21b86d7cf03d636bf6230aa206486bef5e68f9874)"
    );
    let result_hash = do_basic_brun(&vec!["brun".to_string(), result_text, "()".to_string()])
        .trim()
        .to_string();
    assert_eq!(
        result_hash,
        "0x6fcb06b1fe29d132bb37f3a21b86d7cf03d636bf6230aa206486bef5e68f9874"
    );
}

#[test]
fn test_treehash_constant_embedded_fancy_order() {
    let result_text = do_basic_run(&vec![
        "run".to_string(),
        "-i".to_string(),
        "resources/tests".to_string(),
        indoc! {"
            (mod ()
              (include sha256tree.clib)
              (defconst C 18)
              (defconst H (+ C G (sha256tree (q 2 3 4))))
              (defconst G (+ B A))
              (defconst A 9)
              (defconst B (* A A))
              H
              )
        "}
        .to_string(),
    ])
    .trim()
    .to_string();
    assert_eq!(
        result_text,
        "(q . 0x6fcb06b1fe29d132bb37f3a21b86d7cf03d636bf6230aa206486bef5e68f98df)"
    );
    let result_hash = do_basic_brun(&vec!["brun".to_string(), result_text, "()".to_string()])
        .trim()
        .to_string();
    assert_eq!(
        result_hash,
        "0x6fcb06b1fe29d132bb37f3a21b86d7cf03d636bf6230aa206486bef5e68f98df"
    );
}

#[test]
fn test_treehash_constant_embedded_fancy_order_from_fun() {
    let result_text = do_basic_run(&vec![
        "run".to_string(),
        "-i".to_string(),
        "resources/tests".to_string(),
        indoc! {"
            (mod ()
              (include sha256tree.clib)
              (defconst C 18)
              (defconst H (+ C G (sha256tree (q 2 3 4))))
              (defconst G (+ B A))
              (defconst A 9)
              (defconst B (* A A))
              (defun F (X) (+ X H))
              (F 1)
              )
        "}
        .to_string(),
    ])
    .trim()
    .to_string();
    assert_eq!(
        result_text,
        "(q . 0x6fcb06b1fe29d132bb37f3a21b86d7cf03d636bf6230aa206486bef5e68f98e0)"
    );
    let result_hash = do_basic_brun(&vec!["brun".to_string(), result_text, "()".to_string()])
        .trim()
        .to_string();
    assert_eq!(
        result_hash,
        "0x6fcb06b1fe29d132bb37f3a21b86d7cf03d636bf6230aa206486bef5e68f98e0"
    );
}

#[test]
fn test_treehash_constant_embedded_classic_loop() {
    let result_text = do_basic_run(&vec![
        "run".to_string(),
        "-i".to_string(),
        "resources/tests".to_string(),
        indoc! {"
            (mod ()
              (include sha256tree.clib)
              (defconst H (+ G (sha256tree (q 2 3 4))))
              (defconst G (logand H 1))
              H
              )
        "}
        .to_string(),
    ])
    .trim()
    .to_string();
    assert!(result_text.starts_with("FAIL"));
    assert!(result_text.contains("got stuck untangling defconst dependencies"));
}

#[test]
fn test_treehash_constant_embedded_modern() {
    let result_text = do_basic_run(&vec![
        "run".to_string(),
        "-i".to_string(),
        "resources/tests".to_string(),
        indoc! {"
            (mod ()
              (include *standard-cl-21*)
              (include sha256tree.clib)
              (defconst H (+ G (sha256tree (q 2 3 4))))
              (defconst G 1)
              H
              )
        "}
        .to_string(),
    ])
    .trim()
    .to_string();
    assert_eq!(
        result_text,
        "(2 (1 1 . 50565442356047746631413349885570059132562040184787699607120092457326103992436) (4 (1 2 (1 2 (3 (7 5) (1 2 (1 11 (1 . 2) (2 2 (4 2 (4 (5 5) ()))) (2 2 (4 2 (4 (6 5) ())))) 1) (1 2 (1 11 (1 . 1) 5) 1)) 1) 1) 1))"
    );
    let result_hash = do_basic_brun(&vec!["brun".to_string(), result_text, "()".to_string()])
        .trim()
        .to_string();
    assert_eq!(
        result_hash,
        "0x6fcb06b1fe29d132bb37f3a21b86d7cf03d636bf6230aa206486bef5e68f9874"
    );
}

#[test]
fn test_treehash_constant_embedded_modern_fun() {
    let result_text = do_basic_run(&vec![
        "run".to_string(),
        "-i".to_string(),
        "resources/tests".to_string(),
        indoc! {"
            (mod ()
              (include *standard-cl-21*)
              (include sha256tree.clib)
              (defconst H (+ G (sha256tree (q 2 3 4))))
              (defconst G 1)
              (defun F (X) (+ X H))
              (F 1)
              )
        "}
        .to_string(),
    ])
    .trim()
    .to_string();
    assert_eq!(
        result_text,
        "(2 (1 2 6 (4 2 (4 (1 . 1) ()))) (4 (1 (2 (1 2 (3 (7 5) (1 2 (1 11 (1 . 2) (2 4 (4 2 (4 (5 5) ()))) (2 4 (4 2 (4 (6 5) ())))) 1) (1 2 (1 11 (1 . 1) 5) 1)) 1) 1) 2 (1 16 5 (1 . 50565442356047746631413349885570059132562040184787699607120092457326103992436)) 1) 1))".to_string()
    );
    let result_hash = do_basic_brun(&vec!["brun".to_string(), result_text, "()".to_string()])
        .trim()
        .to_string();
    assert_eq!(
        result_hash,
        "0x6fcb06b1fe29d132bb37f3a21b86d7cf03d636bf6230aa206486bef5e68f9875"
    );
}

#[test]
fn test_treehash_constant_embedded_modern_loop() {
    let result_text = do_basic_run(&vec![
        "run".to_string(),
        "-i".to_string(),
        "resources/tests".to_string(),
        indoc! {"
            (mod ()
              (include *standard-cl-21*)
              (include sha256tree.clib)
              (defconst H (+ G (sha256tree (q 2 3 4))))
              (defconst G (logand H 1))
              H
              )
        "}
        .to_string(),
    ])
    .trim()
    .to_string();
    eprintln!("{result_text}");
    // Asserting where the stack overflows isn't necessary.
    assert!(result_text.contains("stack limit exceeded"));
}

#[test]
fn test_embed_file_2() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "-i".to_string(),
        "resources/tests".to_string(),
        "(mod () (embed-file testhex hex hex-embed-01.hex) testhex)".to_string(),
    ])
    .trim()
    .to_string();
    let run_result = do_basic_brun(&vec!["brun".to_string(), program, "()".to_string()])
        .trim()
        .to_string();
    assert_eq!(run_result, "(65 66 67)");
}

#[test]
fn test_embed_file_4() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "-i".to_string(),
        "resources/tests".to_string(),
        "(mod () (include *standard-cl-21*) (embed-file testhex hex hex-embed-01.hex) testhex)"
            .to_string(),
    ])
    .trim()
    .to_string();
    let run_result = do_basic_brun(&vec!["brun".to_string(), program, "()".to_string()])
        .trim()
        .to_string();
    assert_eq!(run_result, "(65 66 67)");
}

#[test]
fn test_embed_file_5() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "-i".to_string(),
        "resources/tests".to_string(),
        "(mod () (embed-file testsexp sexp embed.sexp) testsexp)".to_string(),
    ])
    .trim()
    .to_string();
    let run_result = do_basic_brun(&vec![
        "brun".to_string(),
        "-n".to_string(),
        program,
        "()".to_string(),
    ])
    .trim()
    .to_string();
    assert_eq!(run_result, "(23 24 25)");
}

#[test]
fn test_embed_file_6() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "-i".to_string(),
        "resources/tests".to_string(),
        "(mod () (include *standard-cl-21*) (embed-file testsexp sexp embed.sexp) testsexp)"
            .to_string(),
    ])
    .trim()
    .to_string();
    let run_result = do_basic_brun(&vec!["brun".to_string(), program, "()".to_string()])
        .trim()
        .to_string();
    assert_eq!(run_result, "(lsh 24 25)");
}

#[test]
fn test_embed_file_7() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "-i".to_string(),
        "resources/tests".to_string(),
        "(mod () (embed-file hello bin a-binary-file-called-hello.dat) hello)".to_string(),
    ])
    .trim()
    .to_string();
    let run_result = do_basic_brun(&vec!["brun".to_string(), program, "()".to_string()])
        .trim()
        .to_string();
    assert_eq!(run_result, "\"hello\"");
}

#[test]
fn test_embed_file_8() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "-i".to_string(),
        "resources/tests".to_string(),
        "(mod () (include *standard-cl-21*) (embed-file hello bin a-binary-file-called-hello.dat) hello)".to_string(),
    ])
    .trim()
    .to_string();
    let run_result = do_basic_brun(&vec!["brun".to_string(), program, "()".to_string()])
        .trim()
        .to_string();
    assert_eq!(run_result, "\"hello\"");
}

#[test]
fn test_embed_file_9() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "-i".to_string(),
        "resources/tests".to_string(),
        "(mod () (include *standard-cl-21*) (embed-file hello bin a-binary-file-called-hello.dat) (sha256 (sha256 hello)))".to_string(),
    ])
        .trim()
        .to_string();
    let run_result = do_basic_brun(&vec!["brun".to_string(), program, "()".to_string()])
        .trim()
        .to_string();
    assert_eq!(
        run_result,
        "0x9595c9df90075148eb06860365df33584b75bff782a510c6cd4883a419833d50"
    );
}

#[test]
fn test_num_encoding_just_less_than_5_bytes() {
    let res = do_basic_run(&vec!["run".to_string(), "4281419728".to_string()])
        .trim()
        .to_string();
    assert_eq!(res, "0x00ff3147d0");
}

#[test]
fn test_divmod() {
    let res = do_basic_run(&vec![
        "run".to_string(),
        "(/ 78962960182680 4281419728)".to_string(),
    ])
    .trim()
    .to_string();
    assert_eq!(res, "18443");
}

#[cfg(test)]
pub struct RandomClvmNumber {
    pub intended_value: Number,
}

#[test]
fn test_classic_mod_form() {
    let res = do_basic_run(&vec![
        "run".to_string(),
        indoc! {"
(mod () (a (mod (X) (+ 1 (* X 2))) (list 3)))
"}
        .to_string(),
        "()".to_string(),
    ])
    .trim()
    .to_string();
    assert_eq!(res, "(q . 7)");
}

#[cfg(test)]
pub fn random_clvm_number<R: Rng + ?Sized>(rng: &mut R) -> RandomClvmNumber {
    // Make a number by creating some random atom bytes.
    // Set high bit randomly.
    let natoms = rng.gen_range(0..=NUM_GEN_ATOMS);
    let mut result_bytes = Vec::new();
    for _ in 0..=natoms {
        let mut new_bytes = sexp::random_atom_name(rng, 3)
            .iter()
            .map(|x| {
                if rng.gen() {
                    // The possibility of negative values.
                    x | 0x80
                } else {
                    *x
                }
            })
            .collect();
        result_bytes.append(&mut new_bytes);
    }
    let num = number_from_u8(&result_bytes);

    RandomClvmNumber {
        intended_value: num,
    }
}

#[cfg(test)]
impl Distribution<RandomClvmNumber> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> RandomClvmNumber {
        random_clvm_number(rng)
    }
}

// Finally add property based testing in here.
#[test]
fn test_encoding_properties() {
    let mut rng = ChaChaRng::from_entropy();
    for _ in 1..=200 {
        let number_spec: RandomClvmNumber = rng.gen();

        // We'll have it compile a constant value.
        // The representation of the number will come out most likely
        // as a hex constant.
        let serialized_through_run = do_basic_run(&vec![
            "run".to_string(),
            format!("(q . {})", number_spec.intended_value),
        ])
        .trim()
        .to_string();

        // If we can subtract the original value from the encoded value and
        // get zero, then we did the right thing.
        let cancelled_through_run = do_basic_run(&vec![
            "run".to_string(),
            format!(
                "(- {} {})",
                serialized_through_run, number_spec.intended_value
            ),
        ])
        .trim()
        .to_string();
        assert_eq!(cancelled_through_run, "()");
    }
}

const SEXP_RNG_HORIZON: usize = 13;
const SEXP_DEPTH: usize = 2;

#[cfg(test)]
fn gather_paths(
    path_map: &mut HashMap<Vec<u8>, Number>,
    p: Number,
    mask: Number,
    sexp: &sexp::SExp,
) {
    let this_path = p.clone() | mask.clone();
    match sexp {
        sexp::SExp::Atom(_, x) => {
            path_map.insert(x.clone(), this_path);
        }
        sexp::SExp::Cons(_, a, b) => {
            let next_mask = mask * 2_u32.to_bigint().unwrap();
            gather_paths(path_map, p.clone(), next_mask.clone(), a.borrow());
            gather_paths(path_map, this_path, next_mask.clone(), b.borrow());
        }
        _ => {}
    }
}

// Ensure our atoms are not taken up as operators during the reading process.
#[cfg(test)]
fn stringize(sexp: &sexp::SExp) -> sexp::SExp {
    match sexp {
        sexp::SExp::Cons(l, a, b) => sexp::SExp::Cons(
            l.clone(),
            Rc::new(stringize(a.borrow())),
            Rc::new(stringize(b.borrow())),
        ),
        sexp::SExp::Atom(l, n) => sexp::SExp::QuotedString(l.clone(), b'"', n.clone()),
        _ => sexp.clone(),
    }
}

#[test]
fn test_check_tricky_arg_path_random() {
    let mut rng = ChaChaRng::from_entropy();
    // Make a very deep random sexp and make a path table in it.
    let random_tree = Rc::new(stringize(&sexp::random_sexp(&mut rng, SEXP_RNG_HORIZON)));
    let mut deep_tree = random_tree.clone();

    let mut path_map = HashMap::new();
    gather_paths(&mut path_map, bi_zero(), bi_one(), &random_tree);
    let mut deep_path = bi_one();
    for _ in 1..=SEXP_DEPTH {
        deep_path *= 2_u32.to_bigint().unwrap();
        if rng.gen() {
            deep_path |= bi_one();
            deep_tree = Rc::new(sexp::SExp::Cons(
                random_tree.loc(),
                Rc::new(sexp::SExp::Nil(random_tree.loc())),
                deep_tree,
            ));
        } else {
            deep_tree = Rc::new(sexp::SExp::Cons(
                random_tree.loc(),
                deep_tree.clone(),
                Rc::new(sexp::SExp::Nil(random_tree.loc())),
            ));
        }
    }
    // Now we have a very deep tree and a path to our sexp.
    // We'll test whether node path serializes to the right thing by
    // checking that we can reach all the atoms in our tree.
    for (k, v) in path_map {
        let np = NodePath::new(Some(deep_path.clone()));
        let up = NodePath::new(Some(v.clone()));
        let path_bytes = np.add(up).as_path();
        let program = sexp::SExp::Cons(
            random_tree.loc(),
            Rc::new(sexp::SExp::Atom(random_tree.loc(), vec![b'a'])),
            Rc::new(sexp::SExp::Cons(
                random_tree.loc(),
                Rc::new(sexp::SExp::QuotedString(
                    random_tree.loc(),
                    b'"',
                    path_bytes.raw().clone(),
                )),
                Rc::new(sexp::SExp::Cons(
                    random_tree.loc(),
                    Rc::new(sexp::SExp::Cons(
                        random_tree.loc(),
                        Rc::new(sexp::SExp::Atom(random_tree.loc(), vec![b'q'])),
                        deep_tree.clone(),
                    )),
                    Rc::new(sexp::SExp::Nil(random_tree.loc())),
                )),
            )),
        );

        let res = do_basic_run(&vec![
            "run".to_string(),
            program.to_string(),
            "()".to_string(),
        ])
        .trim()
        .to_string();
        let mut allocator = Allocator::new();
        let converted = convert_to_clvm_rs(
            &mut allocator,
            Rc::new(sexp::SExp::Atom(random_tree.loc(), k.clone())),
        )
        .unwrap();
        let disassembled = disassemble(&mut allocator, converted, Some(0));
        assert_eq!(disassembled, res);
    }
}

pub fn read_json_from_file(fname: &str) -> HashMap<String, String> {
    let extra_symbols_text = fs::read_to_string(fname).expect("should have dropped main.sym");
    eprintln!("est {extra_symbols_text}");
    serde_json::from_str(&extra_symbols_text).expect("should be real json")
}

#[test]
fn test_generate_extra_symbols() {
    // Verify that extra symbols are generated.
    // These include ..._arguments: "(A B C)" <-- arguments of the function
    //               ..._left_env: "1" <-- specifies whether left env is used
    let _ = do_basic_run(&vec![
        "run".to_string(),
        "-g".to_string(),
        "-i".to_string(),
        "resources/tests".to_string(),
        "-i".to_string(),
        "resources/tests/usecheck-work".to_string(),
        "--symbol-output-file".to_string(),
        "/tmp/pmi_extra_symbols.sym".to_string(),
        "resources/tests/cldb_tree/pool_member_innerpuz.cl".to_string(),
    ])
    .trim()
    .to_string();
    let syms_with_extras = read_json_from_file("/tmp/pmi_extra_symbols.sym");
    let syms_want_extras =
        read_json_from_file("resources/tests/cldb_tree/pool_member_innerpuz_extra.sym");
    assert_eq!(syms_with_extras, syms_want_extras);
    let _ = do_basic_run(&vec![
        "run".to_string(),
        "-i".to_string(),
        "resources/tests".to_string(),
        "-i".to_string(),
        "resources/tests/usecheck-work".to_string(),
        "--symbol-output-file".to_string(),
        "/tmp/pmi_normal_symbols.sym".to_string(),
        "resources/tests/cldb_tree/pool_member_innerpuz.cl".to_string(),
    ])
    .trim()
    .to_string();
    let syms_normal = read_json_from_file("/tmp/pmi_normal_symbols.sym");
    let want_normal = read_json_from_file("resources/tests/cldb_tree/pool_member_innerpuz_ref.sym");
    assert_eq!(syms_normal, want_normal);
}

#[test]
fn test_classic_sets_source_file_in_symbols() {
    let tname = "test_classic_sets_source_file_in_symbols.sym".to_string();
    do_basic_run(&vec![
        "run".to_string(),
        "--extra-syms".to_string(),
        "--symbol-output-file".to_string(),
        tname.clone(),
        "resources/tests/assert.clvm".to_string(),
    ]);
    let read_in_file = fs::read_to_string(&tname).expect("should have dropped symbols");
    let decoded_symbol_file: HashMap<String, String> =
        serde_json::from_str(&read_in_file).expect("should decode");
    assert_eq!(
        decoded_symbol_file.get("source_file").cloned(),
        Some("resources/tests/assert.clvm".to_string())
    );
    fs::remove_file(tname).expect("should have dropped symbols");
}

#[test]
fn test_classic_sets_source_file_in_symbols_only_when_asked() {
    let tname = "test_classic_doesnt_source_file_in_symbols.sym".to_string();
    do_basic_run(&vec![
        "run".to_string(),
        "--symbol-output-file".to_string(),
        tname.clone(),
        "resources/tests/assert.clvm".to_string(),
    ]);
    let read_in_file = fs::read_to_string(&tname).expect("should have dropped symbols");
    fs::remove_file(&tname).expect("should have existed");
    let decoded_symbol_file: HashMap<String, String> =
        serde_json::from_str(&read_in_file).expect("should decode");
    assert_eq!(decoded_symbol_file.get("source_file"), None);
}

#[test]
fn test_modern_sets_source_file_in_symbols() {
    let tname = "test_modern_sets_source_file_in_symbols.sym".to_string();
    do_basic_run(&vec![
        "run".to_string(),
        "--extra-syms".to_string(),
        "--symbol-output-file".to_string(),
        tname.clone(),
        "resources/tests/steprun/fact.cl".to_string(),
    ]);
    let read_in_file = fs::read_to_string(&tname).expect("should have dropped symbols");
    let decoded_symbol_file: HashMap<String, String> =
        serde_json::from_str(&read_in_file).expect("should decode");
    fs::remove_file(&tname).expect("should have existed");
    assert_eq!(
        decoded_symbol_file.get("source_file").cloned(),
        Some("resources/tests/steprun/fact.cl".to_string())
    );
}

// Test that leaving off the lambda captures causes bare words for the
// requested values to find their way into the output and that having
// the capture catches it.  This shows that uses of uncaptured words
// are unencumbered.
#[test]
fn test_lambda_without_capture_reproduces_bare_word_in_output() {
    let compiled = do_basic_run(&vec![
        "run".to_string(),
        "-i".to_string(),
        "resources/tests".to_string(),
        "resources/tests/rps-referee-uncaptured.clsp".to_string(),
    ])
    .trim()
    .to_string();
    assert!(compiled.contains("AMOUNT"));
    assert!(compiled.contains("new_puzzle_hash"));
}

// Test that strict cl21 throws an error rather than compiling the above.
#[test]
fn test_lambda_without_capture_strict() {
    let compiler_result = do_basic_run(&vec![
        "run".to_string(),
        "-i".to_string(),
        "resources/tests".to_string(),
        "resources/tests/strict/rps-referee-uncaptured.clsp".to_string(),
    ])
    .trim()
    .to_string();
    assert!(compiler_result.contains("Unbound"));
    assert!(compiler_result.contains("new_puzzle_hash"));
}

// Test that having a lambda capture captures all the associated words.
#[test]
fn test_lambda_with_capture_defines_word() {
    let compiled = do_basic_run(&vec![
        "run".to_string(),
        "-i".to_string(),
        "resources/tests".to_string(),
        "resources/tests/rps-referee.clsp".to_string(),
    ])
    .trim()
    .to_string();
    assert!(!compiled.contains("AMOUNT"));
    assert!(!compiled.contains("new_puzzle_hash"));
}

#[test]
fn test_cost_reporting_0() {
    let program = "(2 (1 2 6 (4 2 (4 (1 . 1) ()))) (4 (1 (2 (1 2 (3 (7 5) (1 2 (1 11 (1 . 2) (2 4 (4 2 (4 (5 5) ()))) (2 4 (4 2 (4 (6 5) ())))) 1) (1 2 (1 11 (1 . 1) 5) 1)) 1) 1) 2 (1 16 5 (1 . 50565442356047746631413349885570059132562040184787699607120092457326103992436)) 1) 1))";
    let result = do_basic_brun(&vec![
        "brun".to_string(),
        "-c".to_string(),
        program.to_string(),
        "()".to_string(),
    ])
    .trim()
    .to_string();
    assert_eq!(
        result,
        "cost = 1978\n0x6fcb06b1fe29d132bb37f3a21b86d7cf03d636bf6230aa206486bef5e68f9875"
    );
}

#[test]
fn test_assign_lambda_code_generation() {
    let tname = "test_assign_lambda_code_generation.sym".to_string();
    do_basic_run(&vec![
        "run".to_string(),
        "--extra-syms".to_string(),
        "--symbol-output-file".to_string(),
        tname.clone(),
        "(mod (A) (include *standard-cl-21*) (defun F (X) (+ X 1)) (assign-lambda X (F A) X))"
            .to_string(),
    ]);
    let read_in_file = fs::read_to_string(&tname).expect("should have dropped symbols");
    fs::remove_file(&tname).expect("should have existed");
    let decoded_symbol_file: HashMap<String, String> =
        serde_json::from_str(&read_in_file).expect("should decode");
    let found_wanted_symbols: Vec<String> = decoded_symbol_file
        .iter()
        .filter(|(_, v)| *v == "F" || v.starts_with("letbinding"))
        .map(|(k, _)| k.clone())
        .collect();
    assert_eq!(found_wanted_symbols.len(), 2);
    // We should have these two functions.
    assert!(found_wanted_symbols
        .contains(&"ccd5be506752cebf01f9930b4c108fe18058c65e1ab57a72ca0a00d9788c7ca6".to_string()));
    assert!(found_wanted_symbols
        .contains(&"0a5af5ae61fae2e53cb309d4d9c2c64baf0261824823008b9cf2b21b09221e44".to_string()));
}

#[test]
fn test_assign_lambda_code_generation_normally_inlines() {
    let tname = "test_assign_inline_code_generation.sym".to_string();
    do_basic_run(&vec![
        "run".to_string(),
        "--extra-syms".to_string(),
        "--symbol-output-file".to_string(),
        tname.clone(),
        "(mod (A) (include *standard-cl-21*) (defun F (X) (+ X 1)) (assign-inline X (F A) X))"
            .to_string(),
    ]);
    let read_in_file = fs::read_to_string(&tname).expect("should have dropped symbols");
    fs::remove_file(&tname).expect("should have existed");
    let decoded_symbol_file: HashMap<String, String> =
        serde_json::from_str(&read_in_file).expect("should decode");
    let found_wanted_symbols: Vec<String> = decoded_symbol_file
        .iter()
        .filter(|(_, v)| *v == "F" || v.starts_with("letbinding"))
        .map(|(k, _)| k.clone())
        .collect();
    assert_eq!(found_wanted_symbols.len(), 1);
    // We should have these two functions.
    assert!(found_wanted_symbols
        .contains(&"ccd5be506752cebf01f9930b4c108fe18058c65e1ab57a72ca0a00d9788c7ca6".to_string()));
}

#[test]
fn test_assign_fancy_final_dot_rest() {
    let result_prog = do_basic_run(&vec![
        "run".to_string(),
        "-i".to_string(),
        "resources/tests/chia-gaming".to_string(),
        "resources/tests/chia-gaming/test-last.clsp".to_string(),
    ]);
    let result = do_basic_brun(&vec!["brun".to_string(), result_prog, "()".to_string()])
        .trim()
        .to_string();
    assert_eq!(result, "101");
}

#[test]
fn test_strict_smoke_0() {
    let result = do_basic_run(&vec![
        "run".to_string(),
        "-i".to_string(),
        "resources/tests/strict".to_string(),
        "resources/tests/strict/strict-test-fail.clsp".to_string(),
    ]);
    assert!(result.contains("Unbound"));
    assert!(result.contains("X1"));
}

#[test]
fn test_strict_smoke_1() {
    let result_prog = do_basic_run(&vec![
        "run".to_string(),
        "-i".to_string(),
        "resources/tests/strict".to_string(),
        "resources/tests/strict/strict-test-pass.clsp".to_string(),
    ]);
    let result = do_basic_brun(&vec!["brun".to_string(), result_prog, "(13)".to_string()])
        .trim()
        .to_string();
    assert_eq!(result, "15");
}

#[test]
fn test_strict_list_fail() {
    let result = do_basic_run(&vec![
        "run".to_string(),
        "-i".to_string(),
        "resources/tests/strict".to_string(),
        "resources/tests/strict/strict-list-fail.clsp".to_string(),
    ]);
    assert!(result.contains("Unbound"));
    assert!(result.contains("X2"));
}

#[test]
fn test_strict_list_pass() {
    let result_prog = do_basic_run(&vec![
        "run".to_string(),
        "-i".to_string(),
        "resources/tests/strict".to_string(),
        "resources/tests/strict/strict-list-pass.clsp".to_string(),
    ]);
    let result = do_basic_brun(&vec!["brun".to_string(), result_prog, "(13)".to_string()])
        .trim()
        .to_string();
    assert_eq!(result, "(strlen 14 15)");
}

#[test]
fn test_strict_nested_list_pass() {
    let result_prog = do_basic_run(&vec![
        "run".to_string(),
        "-i".to_string(),
        "resources/tests/strict".to_string(),
        "resources/tests/strict/strict-nested-list.clsp".to_string(),
    ]);
    let result = do_basic_brun(&vec!["brun".to_string(), result_prog, "(13)".to_string()])
        .trim()
        .to_string();
    assert_eq!(result, "(strlen (strlen) ((strlen)))");
}

#[test]
fn test_double_constant_pass() {
    let result_prog = do_basic_run(&vec![
        "run".to_string(),
        "-i".to_string(),
        "resources/tests/strict".to_string(),
        "resources/tests/strict/double-constant-pass.clsp".to_string(),
    ]);
    let result = do_basic_brun(&vec!["brun".to_string(), result_prog, "()".to_string()])
        .trim()
        .to_string();
    assert_eq!(result, "198");
}

#[test]
fn test_double_constant_fail() {
    let result = do_basic_run(&vec![
        "run".to_string(),
        "-i".to_string(),
        "resources/tests/strict".to_string(),
        "resources/tests/strict/double-constant-fail.clsp".to_string(),
    ]);
    assert!(result.contains("not a number given to only-integers"));
    assert!(result.contains("\"hithere\""));
}

#[test]
fn test_double_constant_pass_in_function() {
    let result_prog = do_basic_run(&vec![
        "run".to_string(),
        "-i".to_string(),
        "resources/tests/strict".to_string(),
        "resources/tests/strict/double-constant-pass-in-function.clsp".to_string(),
    ]);
    let result = do_basic_brun(&vec!["brun".to_string(), result_prog, "(13)".to_string()])
        .trim()
        .to_string();
    assert_eq!(result, "211");
}

#[test]
fn test_check_symbol_kinds_nested_if() {
    let result_prog = do_basic_run(&vec![
        "run".to_string(),
        "-i".to_string(),
        "resources/tests/strict".to_string(),
        "resources/tests/strict/strict-classify-expr-if.clsp".to_string(),
    ]);
    let result_1 = do_basic_brun(&vec![
        "brun".to_string(),
        result_prog.clone(),
        "(1)".to_string(),
    ])
    .trim()
    .to_string();
    assert_eq!(result_1, "2");
    let result_0 = do_basic_brun(&vec!["brun".to_string(), result_prog, "(0)".to_string()])
        .trim()
        .to_string();
    assert_eq!(result_0, "(q 1 2 3 4 4)");
}

// Check for successful deinlining of a large constant.
// The result program tables the constant.
#[test]
fn test_basic_deinlining_smoke_0() {
    let fname = "resources/tests/simple_deinline_case_23.clsp";
    let file_content = fs::read_to_string(fname).expect("should exist");
    let result_prog = do_basic_run(&vec!["run".to_string(), fname.to_string()]);
    eprintln!("result_prog 23 {result_prog}");
    assert_eq!(result_prog.matches("1000000").count(), 1);
    let old_prog = file_content.to_string().replace("23", "21");
    let result_prog_21 = do_basic_run(&vec!["run".to_string(), old_prog]);
    eprintln!("result_prog 21 {result_prog_21}");
    assert_eq!(result_prog_21.matches("1000000").count(), 6);
    assert!(result_prog.len() < result_prog_21.len());
}

// Check for the optimizer to reduce a fully constant program to a constant.
#[test]
fn test_optimizer_fully_reduces_constant_outcome_0() {
    let res = do_basic_run(&vec![
        "run".to_string(),
        "(mod () (include *standard-cl-23*) (defun F (X) (+ X 1)) (F 3))".to_string(),
    ]);
    assert_eq!(res, "(1 . 4)");
}

// Check for the optimizer to reduce a fully constant program to a constant.
#[test]
fn test_optimizer_fully_reduces_constant_outcome_sha256tree() {
    let res = do_basic_run(&vec![
        "run".to_string(),
        "-i".to_string(),
        "resources/tests".to_string(),
        "(mod () (include *standard-cl-23*) (include sha256tree.clib) (defun F (X) (sha256tree (+ X 1))) (F 3))".to_string(),
    ]);
    assert_eq!(
        res,
        "(1 . -39425664269051251592384450451821132878837081010681666327853404714379049572411)"
    );
}

// Check for the optimizer to reduce a fully constant function call to a constant
// and propogate through another expression.
#[test]
fn test_optimizer_fully_reduces_constant_outcome_sha256tree_1() {
    let res = do_basic_run(&vec![
        "run".to_string(),
        "-i".to_string(),
        "resources/tests".to_string(),
        "(mod () (include *standard-cl-23*) (include sha256tree.clib) (defun F (X) (sha256tree (+ X 1))) (+ (F 3) 1))".to_string(),
    ]);
    assert_eq!(
        res,
        "(1 . -39425664269051251592384450451821132878837081010681666327853404714379049572410)"
    );
}

#[test]
fn test_g1_map_op_modern() {
    let program = "(mod (S) (include *standard-cl-21*) (g1_map S \"BLS_SIG_BLS12381G1_XMD:SHA-256_SSWU_RO_AUG_\"))";
    let compiled = do_basic_run(&vec!["run".to_string(), program.to_string()]);
    let output = do_basic_brun(&vec![
        "brun".to_string(),
        compiled,
        "(abcdef0123456789)".to_string(),
    ])
    .trim()
    .to_string();
    assert_eq!(
        output,
        "0x88e7302bf1fa8fcdecfb96f6b81475c3564d3bcaf552ccb338b1c48b9ba18ab7195c5067fe94fb216478188c0a3bef4a"
    );
}

#[test]
fn test_optimizer_fully_reduces_constant_outcome_let_0() {
    let res = do_basic_run(&vec![
        "run".to_string(),
        "-i".to_string(),
        "resources/tests".to_string(),
        "(mod (A) (include *standard-cl-23*) (include sha256tree.clib) (defun F (X) (sha256tree (+ X 1))) (defun G (Q) (let ((R (F Q))) (+ R 1))) (+ A (G 3)))".to_string(),
    ]);
    // Tree shaking will remove the functions that became unused due to constant
    // reduction.  We now support suppressing the left env in stepping 23 and
    // above.
    assert_eq!(
        res,
        "(16 2 (1 . -39425664269051251592384450451821132878837081010681666327853404714379049572410))"
    );
}

// Test that the optimizer inverts (i (not x) a b) to (i x b a)
#[test]
fn test_not_inversion_body() {
    let res = do_basic_run(&vec![
        "run".to_string(),
        "(mod (X) (include *standard-cl-23*) (if (not X) (+ X 1) (* X 2)))".to_string(),
    ]);
    assert_eq!(res, "(2 (3 2 (1 18 2 (1 . 2)) (1 16 2 (1 . 1))) 1)");
}

// Test that we can test chialisp outcomes in chialisp.
#[test]
fn test_chialisp_in_chialisp_test_pos() {
    let compiled = do_basic_run(&vec![
        "run".to_string(),
        "-i".to_string(),
        "resources/tests".to_string(),
        "(mod (X) (include *standard-cl-23*) (if (= (f (mod () (include *standard-cl-23*) (include sha256tree.clib) (defun F (X) (sha256tree (+ X 1))) (+ (F 3) 1))) 1) \"worked\" \"didnt work\"))".to_string(),
    ]);
    assert_eq!(compiled, "(1 . \"worked\")");
}

// Test that we can test chialisp outcomes in chialisp.
#[test]
fn test_chialisp_in_chialisp_test_neg() {
    let compiled = do_basic_run(&vec![
        "run".to_string(),
        "-i".to_string(),
        "resources/tests".to_string(),
        "(mod (X) (include *standard-cl-23*) (if (= (f (mod () (include *standard-cl-23*) (include sha256tree.clib) (defun F (X) (sha256tree (+ (list X) 1))) (+ (F 3) 1))) 1) \"worked\" \"didnt work\"))".to_string(),
    ]);
    assert_eq!(compiled, "(1 . \"didnt work\")");
}

// Test CSE when detections are inside a lambda.  It's necessary to add a capture for
// the replaced expression.
#[test]
fn test_cse_replacement_inside_lambda_23_0() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "-i".to_string(),
        "resources/tests".to_string(),
        "resources/tests/more_exhaustive/lambda_cse_1.clsp".to_string(),
    ]);
    let res = do_basic_brun(&vec![
        "brun".to_string(),
        program.clone(),
        "(17 17)".to_string(),
    ])
    .trim()
    .to_string();
    assert_eq!(res, "0x15aa51");
    let res = do_basic_brun(&vec![
        "brun".to_string(),
        program.clone(),
        "(17 19)".to_string(),
    ])
    .trim()
    .to_string();
    assert_eq!(res, "0x1b1019");
    let res = do_basic_brun(&vec!["brun".to_string(), program, "(19 17)".to_string()])
        .trim()
        .to_string();
    assert_eq!(res, "0x1e3f2b");
}

#[test]
fn test_cse_replacement_inside_lambda_test_desugared_form_23() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "-i".to_string(),
        "resources/tests".to_string(),
        "resources/tests/more_exhaustive/lambda_cse_1_desugared_form.clsp".to_string(),
    ]);
    let res = do_basic_brun(&vec!["brun".to_string(), program, "(17 17)".to_string()])
        .trim()
        .to_string();
    assert_eq!(res, "0x15aa51");
}

// Note: this program is intentionally made to properly preprocess but trigger
// an error in strict compilation as a demonstration and test that the preprocessor
// is a mechanically separate step from compilation.  Separating them like this
// has the advantage that you can emit preprocessed compiler input on its own
// and also that it internally untangles the stages and makes compilation simpler.
#[test]
fn test_defmac_if_smoke_preprocess() {
    let result_prog = do_basic_run(&vec![
        "run".to_string(),
        "-i".to_string(),
        "resources/tests/strict".to_string(),
        "-E".to_string(),
        "resources/tests/strict/defmac_if_smoke.clsp".to_string(),
    ]);
    assert_eq!(
        result_prog,
        "(mod () (include *strict-cl-21*) (a (i t1 (com t2) (com t3)) @))"
    );
    let result2 = do_basic_run(&vec!["run".to_string(), result_prog]);
    assert!(result2.contains("Unbound use"));
    // Ensure that we're identifying one of the actually misused variables, but
    // do not make a claim about which one is identified first.
    assert!(result2.contains("of t1") || result2.contains("of t2") || result2.contains("of t3"));
}

#[test]
fn test_defmac_assert_smoke_preprocess() {
    let result_prog = do_basic_run(&vec![
        "run".to_string(),
        "-i".to_string(),
        "resources/tests/strict".to_string(),
        "-E".to_string(),
        "resources/tests/strict/assert.clsp".to_string(),
    ]);
    assert_eq!(
        result_prog,
        "(mod (A) (include *strict-cl-21*) (a (i 1 (com (a (i A (com 13) (com (x))) @)) (com (x))) @))"
    );
    let result_after_preproc = do_basic_run(&vec!["run".to_string(), result_prog]);
    let result_with_preproc = do_basic_run(&vec![
        "run".to_string(),
        "-i".to_string(),
        "resources/tests/strict".to_string(),
        "resources/tests/strict/assert.clsp".to_string(),
    ]);
    assert_eq!(result_after_preproc, result_with_preproc);
    let run_result_true = do_basic_brun(&vec![
        "brun".to_string(),
        result_with_preproc.clone(),
        "(15)".to_string(),
    ]);
    assert_eq!(run_result_true.trim(), "13");
    let run_result_false = do_basic_brun(&vec![
        "brun".to_string(),
        result_with_preproc.clone(),
        "(0)".to_string(),
    ]);
    assert_eq!(run_result_false.trim(), "FAIL: clvm raise ()");
}

#[test]
fn test_defmac_assert_smoke_preprocess_23() {
    let result_prog = do_basic_run(&vec![
        "run".to_string(),
        "-i".to_string(),
        "resources/tests/strict".to_string(),
        "-E".to_string(),
        "resources/tests/strict/assert23.clsp".to_string(),
    ]);
    assert_eq!(
        result_prog,
        "(mod (A) (include *standard-cl-23*) (a (i 1 (com (a (i A (com 13) (com (x))) @)) (com (x))) @))"
    );
    let result_after_preproc = do_basic_run(&vec!["run".to_string(), result_prog]);
    let result_with_preproc = do_basic_run(&vec![
        "run".to_string(),
        "-i".to_string(),
        "resources/tests/strict".to_string(),
        "resources/tests/strict/assert23.clsp".to_string(),
    ]);
    assert_eq!(result_after_preproc, result_with_preproc);
    let run_result_true = do_basic_brun(&vec![
        "brun".to_string(),
        result_with_preproc.clone(),
        "(15)".to_string(),
    ]);
    assert_eq!(run_result_true.trim(), "13");
    let run_result_false = do_basic_brun(&vec![
        "brun".to_string(),
        result_with_preproc.clone(),
        "(0)".to_string(),
    ]);
    assert_eq!(run_result_false.trim(), "FAIL: clvm raise ()");
}

#[test]
fn test_smoke_inline_at_expansion_23_0() {
    let prog = do_basic_run(&vec![
        "run".to_string(),
        indoc! {"
          (mod (Y)
            (include *standard-cl-23*)
            (defun A (X) (r @*env*)) ;; X = ((3)), returns (((3)))
            (defun B (X) (A (r @*env*))) ;; X = (3)
            (defun C (X) (B (r @*env*))) ;; X = 3
            (C Y) ;; Y = 3
            )
        "}
        .to_string(),
    ]);
    let run_result = do_basic_brun(&vec!["brun".to_string(), prog, "(3)".to_string()]);
    assert_eq!(run_result.trim(), "(((i)))");
}

#[test]
fn test_smoke_inline_at_expansion_23_1() {
    let prog = do_basic_run(&vec![
        "run".to_string(),
        indoc! {"
          (mod (Y)
            (include *standard-cl-23*)
            (defun-inline A (X) (r @*env*))
            (defun B (X) (A (r @*env*)))
            (defun C (X) (B (r @*env*)))
            (C Y)
            )
        "}
        .to_string(),
    ]);
    let run_result = do_basic_brun(&vec!["brun".to_string(), prog, "(3)".to_string()]);
    assert_eq!(run_result.trim(), "(((i)))");
}

#[test]
fn test_smoke_inline_at_expansion_23_2() {
    let prog = do_basic_run(&vec![
        "run".to_string(),
        indoc! {"
          (mod (Y)
            (include *standard-cl-23*)
            (defun A (X) (r @*env*))
            (defun-inline B (X) (A (r @*env*)))
            (defun C (X) (B (r @*env*)))
            (C Y)
            )
        "}
        .to_string(),
    ]);
    let run_result = do_basic_brun(&vec!["brun".to_string(), prog, "(3)".to_string()]);
    assert_eq!(run_result.trim(), "(((i)))");
}

#[test]
fn test_smoke_inline_at_expansion_23_3() {
    let prog = do_basic_run(&vec![
        "run".to_string(),
        indoc! {"
          (mod (Y)
            (include *standard-cl-23*)
            (defun-inline A (X) (r @*env*))
            (defun-inline B (X) (A (r @*env*)))
            (defun C (X) (B (r @*env*)))
            (C Y)
            )
        "}
        .to_string(),
    ]);
    let run_result = do_basic_brun(&vec!["brun".to_string(), prog, "(3)".to_string()]);
    assert_eq!(run_result.trim(), "(((i)))");
}

#[test]
fn test_smoke_inline_at_expansion_23_4() {
    let prog = do_basic_run(&vec![
        "run".to_string(),
        indoc! {"
          (mod (Y)
            (include *standard-cl-23*)
            (defun A (X) (r @*env*))
            (defun B (X) (A (r @*env*)))
            (defun-inline C (X) (B (r @*env*)))
            (C Y)
            )
        "}
        .to_string(),
    ]);
    let run_result = do_basic_brun(&vec!["brun".to_string(), prog, "(3)".to_string()]);
    assert_eq!(run_result.trim(), "(((i)))");
}

#[test]
fn test_smoke_inline_at_expansion_23_5() {
    let prog = do_basic_run(&vec![
        "run".to_string(),
        indoc! {"
          (mod (Y)
            (include *standard-cl-23*)
            (defun-inline A (X) (r @*env*))
            (defun B (X) (A (r @*env*)))
            (defun-inline C (X) (B (r @*env*)))
            (C Y)
            )
        "}
        .to_string(),
    ]);
    let run_result = do_basic_brun(&vec!["brun".to_string(), prog, "(3)".to_string()]);
    assert_eq!(run_result.trim(), "(((i)))");
}

#[test]
fn test_smoke_inline_at_expansion_23_6() {
    let prog = do_basic_run(&vec![
        "run".to_string(),
        indoc! {"
          (mod (Y)
            (include *standard-cl-23*)
            (defun A (X) (r @*env*))
            (defun-inline B (X) (A (r @*env*)))
            (defun-inline C (X) (B (r @*env*)))
            (C Y)
            )
        "}
        .to_string(),
    ]);
    let run_result = do_basic_brun(&vec!["brun".to_string(), prog, "(3)".to_string()]);
    assert_eq!(run_result.trim(), "(((i)))");
}

#[test]
fn test_smoke_inline_at_expansion_23_7() {
    let prog = do_basic_run(&vec![
        "run".to_string(),
        indoc! {"
          (mod (Y)
            (include *standard-cl-23*)
            (defun-inline A (X) (r @*env*))
            (defun-inline B (X) (A (r @*env*)))
            (defun-inline C (X) (B (r @*env*)))
            (C Y)
            )
        "}
        .to_string(),
    ]);
    let run_result = do_basic_brun(&vec!["brun".to_string(), prog, "(3)".to_string()]);
    assert_eq!(run_result.trim(), "(((i)))");
}

#[test]
fn test_smoke_inline_at_expansion_var_23_0() {
    let prog = do_basic_run(&vec![
        "run".to_string(),
        indoc! {"
          (mod (Y)
            (include *standard-cl-23*)
            (defun A (((PPX) PX) X) (list PPX PX X))
            (defun B ((PX) X) (A (r @*env*) (+ X 1)))
            (defun C (X) (B (r @*env*) (+ X 1)))
            (C Y)
            )
        "}
        .to_string(),
    ]);
    let run_result = do_basic_brun(&vec!["brun".to_string(), prog, "(3)".to_string()]);
    assert_eq!(run_result.trim(), "(i 4 5)");
}

#[test]
fn test_smoke_inline_at_expansion_var_23_1() {
    let prog = do_basic_run(&vec![
        "run".to_string(),
        indoc! {"
          (mod (Y)
            (include *standard-cl-23*)
            (defun-inline A (((PPX) PX) X) (list PPX PX X))
            (defun B ((PX) X) (A (r @*env*) (+ X 1)))
            (defun C (X) (B (r @*env*) (+ X 1)))
            (C Y)
            )
        "}
        .to_string(),
    ]);
    let run_result = do_basic_brun(&vec!["brun".to_string(), prog, "(3)".to_string()]);
    assert_eq!(run_result.trim(), "(i 4 5)");
}

#[test]
fn test_smoke_inline_at_expansion_var_23_2() {
    let prog = do_basic_run(&vec![
        "run".to_string(),
        indoc! {"
          (mod (Y)
            (include *standard-cl-23*)
            (defun A (((PPX) PX) X) (list PPX PX X))
            (defun-inline B ((PX) X) (A (r @*env*) (+ X 1)))
            (defun C (X) (B (r @*env*) (+ X 1)))
            (C Y)
            )
        "}
        .to_string(),
    ]);
    let run_result = do_basic_brun(&vec!["brun".to_string(), prog, "(3)".to_string()]);
    assert_eq!(run_result.trim(), "(i 4 5)");
}

#[test]
fn test_smoke_inline_at_expansion_var_23_3() {
    let prog = do_basic_run(&vec![
        "run".to_string(),
        indoc! {"
          (mod (Y)
            (include *standard-cl-23*)
            (defun-inline A (((PPX) PX) X) (list PPX PX X))
            (defun-inline B ((PX) X) (A (r @*env*) (+ X 1)))
            (defun C (X) (B (r @*env*) (+ X 1)))
            (C Y)
            )
        "}
        .to_string(),
    ]);
    let run_result = do_basic_brun(&vec!["brun".to_string(), prog, "(3)".to_string()]);
    assert_eq!(run_result.trim(), "(i 4 5)");
}

#[test]
fn test_smoke_inline_at_expansion_var_23_4() {
    let prog = do_basic_run(&vec![
        "run".to_string(),
        indoc! {"
          (mod (Y)
            (include *standard-cl-23*)
            (defun A (((PPX) PX) X) (list PPX PX X))
            (defun B ((PX) X) (A (r @*env*) (+ X 1)))
            (defun-inline C (X) (B (r @*env*) (+ X 1)))
            (C Y)
            )
        "}
        .to_string(),
    ]);
    let run_result = do_basic_brun(&vec!["brun".to_string(), prog, "(3)".to_string()]);
    assert_eq!(run_result.trim(), "(i 4 5)");
}

#[test]
fn test_smoke_inline_at_expansion_var_23_5() {
    let prog = do_basic_run(&vec![
        "run".to_string(),
        indoc! {"
          (mod (Y)
            (include *standard-cl-23*)
            (defun-inline A (((PPX) PX) X) (list PPX PX X))
            (defun B ((PX) X) (A (r @*env*) (+ X 1)))
            (defun-inline C (X) (B (r @*env*) (+ X 1)))
            (C Y)
            )
        "}
        .to_string(),
    ]);
    let run_result = do_basic_brun(&vec!["brun".to_string(), prog, "(3)".to_string()]);
    assert_eq!(run_result.trim(), "(i 4 5)");
}

#[test]
fn test_smoke_inline_at_expansion_var_23_6() {
    let prog = do_basic_run(&vec![
        "run".to_string(),
        indoc! {"
          (mod (Y)
            (include *standard-cl-23*)
            (defun A (((PPX) PX) X) (list PPX PX X))
            (defun-inline B ((PX) X) (A (r @*env*) (+ X 1)))
            (defun-inline C (X) (B (r @*env*) (+ X 1)))
            (C Y)
            )
        "}
        .to_string(),
    ]);
    let run_result = do_basic_brun(&vec!["brun".to_string(), prog, "(3)".to_string()]);
    assert_eq!(run_result.trim(), "(i 4 5)");
}

#[test]
fn test_smoke_inline_at_expansion_var_23_7() {
    let prog = do_basic_run(&vec![
        "run".to_string(),
        indoc! {"
          (mod (Y)
            (include *standard-cl-23*)
            (defun-inline A (((PPX) PX) X) (list PPX PX X))
            (defun-inline B ((PX) X) (A (r @*env*) (+ X 1)))
            (defun-inline C (X) (B (r @*env*) (+ X 1)))
            (C Y)
            )
        "}
        .to_string(),
    ]);
    let run_result = do_basic_brun(&vec!["brun".to_string(), prog, "(3)".to_string()]);
    assert_eq!(run_result.trim(), "(i 4 5)");
}

//
// Inside assert_ (items):
// items @ 5
//
// Inside letbinding_$_44 ((items) cse_$_43_$_24)
//
// items @ 9
// cse_$_43_$_24 @ 11
//
#[test]
fn test_inline_vs_deinline_23() {
    eprintln!("=== W2 ===");
    let compiled_2 = do_basic_run(&vec![
        "run".to_string(),
        "resources/tests/strict/use-w2.clsp".to_string(),
    ]);
    eprintln!("=== W3 ===");
    let compiled_3 = do_basic_run(&vec![
        "run".to_string(),
        "resources/tests/strict/use-w3.clsp".to_string(),
    ]);
    eprintln!("=== RUN W2 ===");
    let result_2 = do_basic_brun(&vec![
        "brun".to_string(),
        compiled_2,
        "((1 2 3))".to_string(),
    ]);
    eprintln!("=== RUN W3 ===");
    let result_3 = do_basic_brun(&vec![
        "brun".to_string(),
        compiled_3,
        "((1 2 3))".to_string(),
    ]);
    assert_eq!(result_2, result_3);
}

#[test]
fn test_rosetta_code_abc_example() {
    let test_words = &[
        ("A", true),
        ("BARK", true),
        ("TREAT", true),
        ("BOOK", false),
        ("COMMON", false),
        ("SQUAD", true),
        ("CONFUSE", true),
    ];
    let prog_pp = do_basic_run(&vec![
        "run".to_string(),
        "resources/tests/strict/rosetta_code_abc.clsp".to_string(),
    ]);
    let prog_np = do_basic_run(&vec![
        "run".to_string(),
        "resources/tests/strict/rosetta_code_abc_preprocessed.clsp".to_string(),
    ]);
    eprintln!("{prog_pp}");
    assert_eq!(prog_pp, prog_np);
    for (w, success) in test_words.iter() {
        eprintln!("{} {}", w, success);
        let result = do_basic_brun(&vec![
            "brun".to_string(),
            prog_pp.clone(),
            format!("({})", w),
        ])
        .trim()
        .to_string();
        if *success {
            assert_eq!(result, "1");
        } else {
            assert_eq!(result, "()");
        }
    }
}

#[test]
fn test_rosetta_code_babbage_problem() {
    let preprocessed = do_basic_run(&vec![
        "run".to_string(),
        "-E".to_string(),
        "resources/tests/strict/rosetta_code_babbage_problem.clsp".to_string(),
    ]);
    assert!(!preprocessed.contains("*macros*"));
    assert!(!preprocessed.contains("(defmacro list"));
    let compiled = do_basic_run(&vec![
        "run".to_string(),
        "resources/tests/strict/rosetta_code_babbage_problem.clsp".to_string(),
    ]);
    let output = do_basic_brun(&vec!["brun".to_string(), compiled, "(269696)".to_string()])
        .trim()
        .to_string();
    assert_eq!(output, "25264");
}

#[test]
fn test_cse_when_not_dominating_conditions() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "resources/tests/strict/cse_doesnt_dominate.clsp".to_string(),
    ]);
    let outcome = do_basic_brun(&vec!["brun".to_string(), program, "(33)".to_string()])
        .trim()
        .to_string();
    assert_eq!(outcome, "0x009988"); // 34 * 34 * 34
}

#[test]
fn test_cse_not_dominating_conditions_with_superior_let() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "resources/tests/strict/cse_doesnt_dominate_superior_let.clsp".to_string(),
    ]);
    let outcome = do_basic_brun(&vec!["brun".to_string(), program, "(33)".to_string()])
        .trim()
        .to_string();
    assert_eq!(outcome, "0x009988"); // 34 * 34 * 34
}

#[test]
fn test_cse_not_dominating_conditions_with_superior_let_outside() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "resources/tests/strict/cse_doesnt_dominate_superior_let_outside.clsp".to_string(),
    ]);
    let outcome = do_basic_brun(&vec!["brun".to_string(), program, "(33)".to_string()])
        .trim()
        .to_string();
    assert_eq!(outcome, "0x5c13d840"); // 34 * 34 * 34 * 34 * 34 * 34
}

#[test]
fn test_cse_not_dominating_conditions_with_superior_let_outside_in_inline() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "resources/tests/strict/cse_doesnt_dominate_superior_let_outside_in_inline.clsp"
            .to_string(),
    ]);
    let outcome = do_basic_brun(&vec!["brun".to_string(), program, "(33)".to_string()])
        .trim()
        .to_string();
    assert_eq!(outcome, "0x5c13d840"); // 34 * 34 * 34 * 34 * 34 * 34
}

#[test]
fn test_cse_not_dominating_conditions_with_superior_let_outside_in_defun() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "resources/tests/strict/cse_doesnt_dominate_superior_let_outside_in_defun.clsp".to_string(),
    ]);
    let outcome = do_basic_brun(&vec!["brun".to_string(), program, "(33)".to_string()])
        .trim()
        .to_string();
    assert_eq!(outcome, "0x5c13d840"); // 34 * 34 * 34 * 34 * 34 * 34
}

#[test]
fn test_cse_not_dominating_conditions_with_superior_let_odi() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "resources/tests/strict/cse_doesnt_dominate_superior_let_odi.clsp".to_string(),
    ]);
    let outcome = do_basic_brun(&vec!["brun".to_string(), program, "(33)".to_string()])
        .trim()
        .to_string();
    assert_eq!(outcome, "0x5c13d840"); // 34 * 34 * 34 * 34 * 34 * 34
}

#[test]
fn test_cse_not_dominating_conditions_with_superior_let_iodi() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "resources/tests/strict/cse_doesnt_dominate_superior_let_iodi.clsp".to_string(),
    ]);
    let outcome = do_basic_brun(&vec!["brun".to_string(), program, "(33)".to_string()])
        .trim()
        .to_string();
    assert_eq!(outcome, "0x5c13d840"); // 34 * 34 * 34 * 34 * 34 * 34
}

#[test]
fn test_chialisp_web_example() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "resources/tests/strict/chialisp-web-example.clsp".to_string(),
    ]);
    let outcome = do_basic_brun(&vec!["brun".to_string(), program, "(100)".to_string()])
        .trim()
        .to_string();
    assert_eq!(outcome, "(306 (101 103))");
}

#[test]
fn test_chialisp_web_example_defconst() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "resources/tests/strict/defconst.clsp".to_string(),
    ]);
    let outcome = do_basic_brun(&vec!["brun".to_string(), program, "(3)".to_string()])
        .trim()
        .to_string();
    assert_eq!(
        outcome,
        "0xf60efb25b9e6e3587acd9cf01c332707bb771801bdb5e4f50ea957a29c8dde89"
    );
}

#[test]
fn test_chialisp_web_example_big_maybe() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "resources/tests/strict/big-maybe.clsp".to_string(),
    ]);
    let outcome = do_basic_brun(&vec!["brun".to_string(), program, "(((3 5)))".to_string()])
        .trim()
        .to_string();
    assert_eq!(outcome, "8");
}

#[test]
fn test_chialisp_web_example_map() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "resources/tests/strict/map-example.clsp".to_string(),
    ]);
    let outcome = do_basic_brun(&vec!["brun".to_string(), program, "((1 2 3))".to_string()])
        .trim()
        .to_string();
    assert_eq!(outcome, "(a 3 4)");
}

#[test]
fn test_chialisp_web_example_map_lambda() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "resources/tests/strict/map-example.clsp".to_string(),
    ]);
    let outcome = do_basic_brun(&vec![
        "brun".to_string(),
        program,
        "((100 101 102))".to_string(),
    ])
    .trim()
    .to_string();
    assert_eq!(outcome, "(101 102 103)");
}

#[test]
fn test_chialisp_web_example_embed() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "-i".to_string(),
        "resources/tests/strict".to_string(),
        "resources/tests/strict/embed.clsp".to_string(),
    ]);
    let outcome = do_basic_brun(&vec!["brun".to_string(), program, "(world)".to_string()])
        .trim()
        .to_string();
    assert_eq!(
        outcome,
        "0x26c60a61d01db5836ca70fefd44a6a016620413c8ef5f259a6c5612d4f79d3b8"
    );
}

#[test]
fn test_g1_map_op_classic() {
    let program = "(mod (S) (g1_map S \"BLS_SIG_BLS12381G1_XMD:SHA-256_SSWU_RO_AUG_\"))";
    let compiled = do_basic_run(&vec!["run".to_string(), program.to_string()]);
    let output = do_basic_brun(&vec![
        "brun".to_string(),
        compiled,
        "(abcdef0123456789)".to_string(),
    ])
    .trim()
    .to_string();
    assert_eq!(
        output,
        "0x88e7302bf1fa8fcdecfb96f6b81475c3564d3bcaf552ccb338b1c48b9ba18ab7195c5067fe94fb216478188c0a3bef4a"
    );
}

#[test]
fn test_g2_map_op_modern() {
    let program = "(mod (S) (include *standard-cl-21*) (g2_map S \"BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_AUG_\"))";
    let compiled = do_basic_run(&vec!["run".to_string(), program.to_string()]);
    let output = do_basic_brun(&vec![
        "brun".to_string(),
        compiled,
        "(0x21473dab7ad0136f7488128d44247b04fa58a9c6b4fab6ef4d)".to_string(),
    ])
    .trim()
    .to_string();
    assert_eq!(
        output,
        "0x879584f6c205b4492abca2be331fc2875596b08c7fbd958fb8d5e725a479d1794b85add1266fb5d410de5c416ce12305166b1c3e2e5d5ae2720a058169b057520d8f2a315f6097c774f659ce5619a070e1cbc8212fb460758e459498d0e598d6"
    );
}

#[test]
fn test_g2_map_op_classic() {
    let program = "(mod (S) (g2_map S \"BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_AUG_\"))";
    let compiled = do_basic_run(&vec!["run".to_string(), program.to_string()]);
    let output = do_basic_brun(&vec![
        "brun".to_string(),
        compiled,
        "(0x21473dab7ad0136f7488128d44247b04fa58a9c6b4fab6ef4d)".to_string(),
    ])
    .trim()
    .to_string();
    assert_eq!(
        output,
        "0x879584f6c205b4492abca2be331fc2875596b08c7fbd958fb8d5e725a479d1794b85add1266fb5d410de5c416ce12305166b1c3e2e5d5ae2720a058169b057520d8f2a315f6097c774f659ce5619a070e1cbc8212fb460758e459498d0e598d6"
    );
}

#[test]
fn test_secp256k1_verify_modern_succeed() {
    let program = "(mod (S) (include *standard-cl-21*) (secp256k1_verify S 0x85932e4d075615be881398cc765f9f78204033f0ef5f832ac37e732f5f0cbda2 0x481477e62a1d02268127ae89cc58929e09ad5d30229721965ae35965d098a5f630205a7e69f4cb8084f16c7407ed7312994ffbf87ba5eb1aee16682dd324943e))";
    let compiled = do_basic_run(&vec!["run".to_string(), program.to_string()]);
    let output = do_basic_brun(&vec![
        "brun".to_string(),
        compiled,
        "(0x02390b19842e100324163334b16947f66125b76d4fa4a11b9ccdde9b7398e64076)".to_string(),
    ])
    .trim()
    .to_string();
    assert_eq!(output, "()");
}

#[test]
fn test_secp256k1_verify_modern_fail() {
    let program = "(mod (S) (include *standard-cl-21*) (secp256k1_verify S 0x935d863e2d28d8e5d399ea8af7393ef11fdffc7d862dcc6b5217a8ef15fb5442 0xbbf0712cc0a283a842011c19682629a5381c5f7ead576defcf12a9a19378e23b087cd0be730dbe78722dcfc81543fca17a30e41070ca2e5b3ae77ccec2cca935))";
    let compiled = do_basic_run(&vec!["run".to_string(), program.to_string()]);
    let output = do_basic_brun(&vec![
        "brun".to_string(),
        compiled,
        "(0x0215043e969dcf616fabe8e8d6b61ddcf6e274c5b04fce957b086dbeb7e899ac63)".to_string(),
    ])
    .trim()
    .to_string();
    assert!(output.starts_with("FAIL: secp256k1_verify failed"));
}

#[test]
fn test_secp256k1_verify_classic_succeed() {
    let program = "(mod (S) (secp256k1_verify S 0x85932e4d075615be881398cc765f9f78204033f0ef5f832ac37e732f5f0cbda2 0x481477e62a1d02268127ae89cc58929e09ad5d30229721965ae35965d098a5f630205a7e69f4cb8084f16c7407ed7312994ffbf87ba5eb1aee16682dd324943e))";
    let compiled = do_basic_run(&vec!["run".to_string(), program.to_string()]);
    let output = do_basic_brun(&vec![
        "brun".to_string(),
        compiled,
        "(0x02390b19842e100324163334b16947f66125b76d4fa4a11b9ccdde9b7398e64076)".to_string(),
    ])
    .trim()
    .to_string();
    assert_eq!(output, "()");
}

#[test]
fn test_secp256k1_verify_classic_fail() {
    let program = "(mod (S) (secp256k1_verify S 0x935d863e2d28d8e5d399ea8af7393ef11fdffc7d862dcc6b5217a8ef15fb5442 0xbbf0712cc0a283a842011c19682629a5381c5f7ead576defcf12a9a19378e23b087cd0be730dbe78722dcfc81543fca17a30e41070ca2e5b3ae77ccec2cca935))";
    let compiled = do_basic_run(&vec!["run".to_string(), program.to_string()]);
    let output = do_basic_brun(&vec![
        "brun".to_string(),
        compiled,
        "(0x0215043e969dcf616fabe8e8d6b61ddcf6e274c5b04fce957b086dbeb7e899ac63)".to_string(),
    ])
    .trim()
    .to_string();
    assert!(output.starts_with("FAIL: secp256k1_verify failed"));
}

#[test]
fn test_secp256k1_verify_modern_int_succeed() {
    // Ensure that even if translated to integer (for example via classic unhygenic macro invocation), this works.
    let program = "(mod (S) (if S (332799744 S 0x85932e4d075615be881398cc765f9f78204033f0ef5f832ac37e732f5f0cbda2 0x481477e62a1d02268127ae89cc58929e09ad5d30229721965ae35965d098a5f630205a7e69f4cb8084f16c7407ed7312994ffbf87ba5eb1aee16682dd324943e) \"empty-secp\"))";
    let compiled = do_basic_run(&vec!["run".to_string(), program.to_string()]);
    let output = do_basic_brun(&vec![
        "brun".to_string(),
        compiled,
        "(0x02390b19842e100324163334b16947f66125b76d4fa4a11b9ccdde9b7398e64076)".to_string(),
    ])
    .trim()
    .to_string();
    assert_eq!(output, "()");
}

#[test]
fn test_secp256k1_verify_modern_int_fail() {
    let program = "(mod (S) (include *standard-cl-21*) (if S (332799744 S 0x935d863e2d28d8e5d399ea8af7393ef11fdffc7d862dcc6b5217a8ef15fb5442 0xbbf0712cc0a283a842011c19682629a5381c5f7ead576defcf12a9a19378e23b087cd0be730dbe78722dcfc81543fca17a30e41070ca2e5b3ae77ccec2cca935) \"empty-secp\"))";
    let compiled = do_basic_run(&vec!["run".to_string(), program.to_string()]);
    let output = do_basic_brun(&vec![
        "brun".to_string(),
        compiled,
        "(0x0215043e969dcf616fabe8e8d6b61ddcf6e274c5b04fce957b086dbeb7e899ac63)".to_string(),
    ])
    .trim()
    .to_string();
    assert!(output.starts_with("FAIL: secp256k1_verify failed"));
}

#[test]
fn test_secp256r1_verify_modern_succeed() {
    let program = "(mod (S) (include *standard-cl-21*) (secp256r1_verify S 0x85932e4d075615be881398cc765f9f78204033f0ef5f832ac37e732f5f0cbda2 0xeae2f488080919bd0a7069c24cdd9c6ce2db423861b0c9d4236cdadbd0005f6d8f3709e6eb19249fd9c8bea664aba35218e67ea4b0f2239488dc3147f336e1e6))";
    let compiled = do_basic_run(&vec!["run".to_string(), program.to_string()]);
    let output = do_basic_brun(&vec![
        "brun".to_string(),
        compiled,
        "(0x033e1a1b2ccbc35883c60fdfc3f4a02175096ade6271fe85517ca5772594bbd0dc)".to_string(),
    ])
    .trim()
    .to_string();
    assert_eq!(output, "()");
}

#[test]
fn test_secp256r1_verify_modern_fail() {
    let program = "(mod (S) (include *standard-cl-21*) (secp256r1_verify S 0x935d863e2d28d8e5d399ea8af7393ef11fdffc7d862dcc6b5217a8ef15fb5442 0xecef274a7408e6cb0196eac64d2ae32fc54c2537f8a9efd5b75a4e8a53b0b156c64564306f38bade4adceac1073d464e4db3d0332141a7203dfd113ad36e393d))";
    let compiled = do_basic_run(&vec!["run".to_string(), program.to_string()]);
    let output = do_basic_brun(&vec![
        "brun".to_string(),
        compiled,
        "(0x025195db74b1902d53758a62ccd9dd01837aa5ae755e878eb0aeccbe8fe477c543)".to_string(),
    ])
    .trim()
    .to_string();
    assert!(output.starts_with("FAIL: secp256r1_verify failed"));
}

#[test]
fn test_secp256r1_verify_classic_succeed() {
    let program = "(mod (S) (secp256r1_verify S 0x85932e4d075615be881398cc765f9f78204033f0ef5f832ac37e732f5f0cbda2 0xeae2f488080919bd0a7069c24cdd9c6ce2db423861b0c9d4236cdadbd0005f6d8f3709e6eb19249fd9c8bea664aba35218e67ea4b0f2239488dc3147f336e1e6))";
    let compiled = do_basic_run(&vec!["run".to_string(), program.to_string()]);
    let output = do_basic_brun(&vec![
        "brun".to_string(),
        compiled,
        "(0x033e1a1b2ccbc35883c60fdfc3f4a02175096ade6271fe85517ca5772594bbd0dc)".to_string(),
    ])
    .trim()
    .to_string();
    assert_eq!(output, "()");
}

#[test]
fn test_secp256r1_verify_classic_fail() {
    let program = "(mod (S) (secp256r1_verify S 0x935d863e2d28d8e5d399ea8af7393ef11fdffc7d862dcc6b5217a8ef15fb5442 0xecef274a7408e6cb0196eac64d2ae32fc54c2537f8a9efd5b75a4e8a53b0b156c64564306f38bade4adceac1073d464e4db3d0332141a7203dfd113ad36e393d))";
    let compiled = do_basic_run(&vec!["run".to_string(), program.to_string()]);
    let output = do_basic_brun(&vec![
        "brun".to_string(),
        compiled,
        "(0x025195db74b1902d53758a62ccd9dd01837aa5ae755e878eb0aeccbe8fe477c543)".to_string(),
    ])
    .trim()
    .to_string();
    assert!(output.starts_with("FAIL: secp256r1_verify failed"));
}

#[test]
fn test_classic_obeys_operator_choice_at_compile_time_no_version() {
    let compiled = do_basic_run(&vec![
        "run".to_string(),
        "(mod () (coinid (sha256 99) (sha256 99) 1))".to_string(),
    ])
    .trim()
    .to_string();
    assert_eq!(
        compiled,
        "(q . 0x97c3f14ced4dfc280611fd8d9b158163e8981b3bce4d1bb6dd0bcc679a2e2455)"
    );
}

#[test]
fn test_classic_obeys_operator_choice_at_compile_time_version_1() {
    let compiled = do_basic_run(&vec![
        "run".to_string(),
        "--operators-version".to_string(),
        "1".to_string(),
        "(mod () (coinid (sha256 99) (sha256 99) 1))".to_string(),
    ])
    .trim()
    .to_string();
    assert_eq!(
        compiled,
        "(q . 0x97c3f14ced4dfc280611fd8d9b158163e8981b3bce4d1bb6dd0bcc679a2e2455)"
    );
}

#[test]
fn test_classic_obeys_operator_choice_at_compile_time_version_0() {
    let compiled = do_basic_run(&vec![
        "run".to_string(),
        "--operators-version".to_string(),
        "0".to_string(),
        "(mod () (coinid (sha256 99) (sha256 99) 1))".to_string(),
    ])
    .trim()
    .to_string();
    assert_eq!(compiled, "FAIL: unimplemented operator 48");
}

#[test]
fn test_continued_if() {
    let prog = indoc! {"
(mod X
  (include *strict-cl-21*)

  (defun bigatom (Xs Ys)
    (if
      Xs (concat (f Xs) (bigatom (r Xs) Ys))
      Ys (concat (f Ys) (bigatom (r Ys) ()))
      ()
      )
    )

  (bigatom (q . (3 4 5)) X)
  )"}
    .to_string();
    let compiled = do_basic_run(&vec!["run".to_string(), prog])
        .trim()
        .to_string();
    let res = do_basic_brun(&vec![
        "brun".to_string(),
        compiled,
        "(13 99 144)".to_string(),
    ])
    .trim()
    .to_string();
    assert_eq!(res.to_string(), "0x0304050d630090");
}

#[test]
fn test_preprocess_can_recurse() {
    let prog = "resources/tests/strict/test-inner-include.clsp".to_string();
    let res = do_basic_run(&vec![
        "run".to_string(),
        "-i".to_string(),
        "resources/tests/strict".to_string(),
        prog.clone(),
    ])
    .trim()
    .to_string();
    assert_eq!(
        res,
        "(2 (1 2 (3 5 (1 2 (1 18 5 (1 . 2)) 1) (1 2 (1 16 5 (1 . 1)) 1)) 1) (4 (1) 1))"
    );
}

#[test]
fn test_assign_rename_tricky() {
    let filename = "resources/tests/cse-complex-21.clsp";
    let program = do_basic_run(&vec!["run".to_string(), filename.to_string()])
        .trim()
        .to_string();

    let run_result_11 = do_basic_brun(&vec![
        "brun".to_string(),
        program.clone(),
        "(11)".to_string(),
    ])
    .trim()
    .to_string();
    assert_eq!(run_result_11, "506");

    let run_result_41 = do_basic_brun(&vec!["brun".to_string(), program, "(41)".to_string()])
        .trim()
        .to_string();
    assert_eq!(run_result_41, "15375");
}
