use std::rc::Rc;

use clvm_rs::allocator::{Allocator, NodePtr};

use crate::classic::clvm_tools::binutils::assemble;
use crate::classic::clvm_tools::stages::stage_0::TRunProgram;

/*
"function" is used in front of a constant uncompiled
program to indicate we want this program literal to be
compiled and quoted, so it can be passed as an argument
to a compiled clvm program.
EG: (function (+ 20 @)) should return (+ (q . 20) 1) when run.
Thus (opt (com (q . (function (+ 20 @)))))
should return (q . (+ (q . 20) 1))
(function PROG) => (opt (com (q . PROG) (q . MACROS)))
We have to use "opt" as (com PROG) might leave
some partial "com" operators in there and our
goals is to compile PROG as much as possible.
 */

fn default_macros_src() -> Vec<&'static str> {
    vec![
        indoc! {"
        ; we have to compile this externally, since it uses itself
        ;(defmacro defmacro (name params body)
        ;    (qq (list (unquote name) (mod (unquote params) (unquote body))))
        ;)
        (q . (\"defmacro\"
           (c (q . \"list\")
              (c (f 1)
                 (c (c (q . \"mod\")
                       (c (f (r 1))
                          (c (f (r (r 1)))
                             (q . ()))))
                    (q . ()))))))
        "},
        /*
               ;(defmacro list ARGS
               ;    ((c (mod args
               ;        (defun compile-list
               ;               (args)
               ;               (if args
               ;                   (qq (c (unquote (f args))
               ;                         (unquote (compile-list (r args)))))
               ;                   ()))
               ;            (compile-list args)
               ;        )
               ;        ARGS
               ;    ))
               ;)
        */
        indoc! {"
        (q \"list\"
            (a (q #a (q #a 2 (c 2 (c 3 (q))))
                     (c (q #a (i 5
                                 (q #c (q . 4)
                                       (c 9 (c (a 2 (c 2 (c 13 (q))))
                                               (q)))
                                 )
                                 (q 1))
                               1)
                        1))
                1))
        "},
        indoc! {"
        (defmacro function (BODY)
            (qq (opt (com (q . (unquote BODY))
                     (qq (unquote (macros)))
                     (qq (unquote (symbols)))))))
        "},
        indoc! {"
        (defmacro if (A B C)
          (qq (a
              (i (unquote A)
                 (function (unquote B))
                 (function (unquote C)))
              @)))
        "},
        indoc! {"
        (q \"__chia__enlist\"
            (a (q #a (q #a 2 (c 2 (c 3 (q))))
                     (c (q #a (i 5
                                 (q #c (q . 4)
                                       (c 9 (c (a 2 (c 2 (c 13 (q))))
                                               (q)))
                                 )
                                 (q 1))
                               1)
                        1))
                2))
        "},
        /*
         / operator at the clvm layer is becoming deprecated and
        will be implemented using divmod.
         */
        indoc! {"
        (defmacro / (A B) (qq (f (divmod (unquote A) (unquote B)))))
        "},
    ]
}

fn build_default_macro_lookup(
    allocator: &mut Allocator,
    eval_f: Rc<dyn TRunProgram>,
    macros_src: &[String],
) -> NodePtr {
    let run = assemble(allocator, "(a (com 2 3) 1)").unwrap();
    let mut default_macro_lookup: NodePtr = NodePtr::NIL;
    for macro_src in macros_src {
        let macro_sexp = assemble(allocator, macro_src).unwrap();
        let env = allocator
            .new_pair(macro_sexp, default_macro_lookup)
            .unwrap();
        let new_macro = eval_f.run_program(allocator, run, env, None).unwrap().1;
        default_macro_lookup = allocator.new_pair(new_macro, default_macro_lookup).unwrap();
    }
    default_macro_lookup
}

pub fn default_macro_lookup(allocator: &mut Allocator, runner: Rc<dyn TRunProgram>) -> NodePtr {
    let macro_srcs: Vec<String> = default_macros_src().iter().map(|s| s.to_string()).collect();
    build_default_macro_lookup(allocator, runner.clone(), &macro_srcs)
}
