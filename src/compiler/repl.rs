use std::collections::HashMap;
use std::collections::HashSet;

use std::env;
use std::mem::swap;
use std::rc::Rc;

use clvm_rs::allocator::Allocator;

use crate::classic::clvm_tools::stages::stage_0::TRunProgram;
use crate::compiler::comptypes::{BodyForm, CompileErr, CompilerOpts};
use crate::compiler::evaluate::{first_of_alist, second_of_alist, Evaluator, EVAL_STACK_LIMIT};
use crate::compiler::frontend::frontend;
use crate::compiler::sexp::{parse_sexp, SExp};
use crate::compiler::srcloc::Srcloc;
use crate::util::ErrInto;

/// An object implementing a full repl for the language of chialisp toplevel forms
/// and expressions.
///
/// Internally, it determines whether the next form it's been asked to act upon
/// is a helperform (preferentally) or an expression.  In the case of a HelperForm,
/// it uses the compiler frontend to absorb the new HelperForm, and in the case of
/// an expression, uses its Evaluator to simplify it, hopefully to a constant.
///
/// Each form used by the repl has a result, which is nil for helperforms and
/// code for expressions...  If the expression fully evaluated, the result is
/// a quoted constant.
pub struct Repl {
    depth: i32,
    input_exp: String,

    toplevel_forms: HashSet<String>,

    opts: Rc<dyn CompilerOpts>,
    evaluator: Evaluator,

    loc: Srcloc,
    stack_limit: Option<usize>,
}

fn program_with_helper(names: Vec<Rc<SExp>>, parsed_program: Rc<SExp>) -> Rc<SExp> {
    let mut body = Rc::new(SExp::Nil(parsed_program.loc()));

    for n in names.iter() {
        body = Rc::new(SExp::Cons(parsed_program.loc(), n.clone(), body));
    }

    body = Rc::new(SExp::Cons(
        parsed_program.loc(),
        Rc::new(SExp::atom_from_string(parsed_program.loc(), "x")),
        body,
    ));

    Rc::new(SExp::Cons(
        parsed_program.loc(),
        Rc::new(SExp::atom_from_string(parsed_program.loc(), "mod")),
        Rc::new(SExp::Cons(
            parsed_program.loc(),
            Rc::new(SExp::Nil(parsed_program.loc())),
            Rc::new(SExp::Cons(
                parsed_program.loc(),
                parsed_program.clone(),
                Rc::new(SExp::Cons(
                    parsed_program.loc(),
                    body,
                    Rc::new(SExp::Nil(parsed_program.loc())),
                )),
            )),
        )),
    ))
}

fn count_depth(s: &str) -> i32 {
    let mut count: i32 = 0;
    for ch in s.as_bytes().iter() {
        if *ch as char == '(' {
            count += 1;
        } else if *ch as char == ')' {
            count -= 1;
        }
    }
    count
}

impl Repl {
    /// Create a new Repl given a set of CompilerOpts and a chialisp program
    /// runner, TRunProgram.  The runner is used to evaluate arbitrary CLVM
    /// code when required.  Evaluator implements some primitives itself, but
    /// many are taken from clvmr.
    pub fn new(opts: Rc<dyn CompilerOpts>, runner: Rc<dyn TRunProgram>) -> Repl {
        let loc = Srcloc::start(&opts.filename());
        let mut toplevel_forms = HashSet::new();

        for w in ["defun", "defun-inline", "defconstant", "defmacro"].iter() {
            toplevel_forms.insert(w.to_string());
        }

        // Setup the stdenv
        let starter_empty_program = program_with_helper(
            vec![
                Rc::new(SExp::atom_from_string(loc.clone(), "if")),
                Rc::new(SExp::atom_from_string(loc.clone(), "list")),
            ],
            Rc::new(SExp::Cons(
                loc.clone(),
                Rc::new(SExp::atom_from_string(loc.clone(), "defconstant")),
                Rc::new(SExp::Cons(
                    loc.clone(),
                    Rc::new(SExp::atom_from_string(loc.clone(), "$interpreter-version")),
                    Rc::new(SExp::Cons(
                        loc.clone(),
                        Rc::new(SExp::atom_from_string(
                            loc.clone(),
                            env!("CARGO_PKG_VERSION"),
                        )),
                        Rc::new(SExp::Nil(loc.clone())),
                    )),
                )),
            )),
        );
        let start_program_fe = frontend(opts.clone(), &[starter_empty_program]).unwrap();
        let evaluator = Evaluator::new(
            opts.clone(),
            runner.clone(),
            start_program_fe.compileform().helpers.clone(),
        );

        Repl {
            depth: 0,
            input_exp: "".to_string(),
            toplevel_forms,
            evaluator,
            opts,
            loc,
            stack_limit: Some(EVAL_STACK_LIMIT),
        }
    }

    /// There is a stack depth limit in Evaluator which limits the depth to which
    /// evaluation can take place.  This configures that.
    pub fn set_stack_limit(&mut self, l: Option<usize>) {
        self.stack_limit = l;
    }

    /// Process one line of input.  If the line ends with the parenthesis depth
    /// greater than zero, then the data is taken, but no attempt is made at
    /// parsing.  When a line ends with parenthesis depth 0, parsing is done.
    ///
    /// SExp's parse_sexp is used, which can handle parsing of multiple complete
    /// forms in the same text.  The forms will be handled one at a time, first
    /// checking whether they qualify as HelperForms and second treating each
    /// as an expression.
    pub fn process_line(
        &mut self,
        allocator: &mut Allocator,
        line: String,
    ) -> Result<Option<Rc<BodyForm>>, CompileErr> {
        self.depth += count_depth(&line);

        let mut input_taken = String::new();
        swap(&mut input_taken, &mut self.input_exp);
        let input_taken = input_taken + "\n" + &line;

        if self.depth < 0 {
            let loc = self.loc.clone();
            let result = parse_sexp(loc, input_taken.bytes())
                .map(|_v| {
                    panic!("too many parens but parsed anyway");
                })
                .err_into();
            self.input_exp = "".to_string();
            self.depth = 0;
            return result;
        }

        if self.depth > 0 {
            self.input_exp = input_taken;
            return Ok(None);
        }

        self.input_exp = "".to_string();

        parse_sexp(self.loc.clone(), input_taken.bytes())
            .err_into()
            .and_then(|parsed_program| {
                if parsed_program.is_empty() {
                    return Ok(None);
                }
                let fa = first_of_alist(parsed_program[0].clone());
                let is_helper = fa
                    .map(|fa| self.toplevel_forms.contains(&fa.to_string()))
                    .unwrap_or_else(|_| false);

                if is_helper {
                    let prog0 = parsed_program[0].clone();
                    let name = second_of_alist(prog0.clone())?;
                    let built_program = program_with_helper(vec![name], prog0);
                    let program = frontend(self.opts.clone(), &[built_program])?;
                    self.evaluator.add_helper(
                        &program.compileform().helpers[program.compileform().helpers.len() - 1],
                    );
                    Ok(Some(Rc::new(BodyForm::Quoted(SExp::Nil(self.loc.clone())))))
                } else {
                    frontend(self.opts.clone(), &parsed_program)
                        .and_then(|program| {
                            self.evaluator.shrink_bodyform(
                                allocator,
                                program.compileform().args.clone(),
                                Rc::new(HashMap::new()),
                                program.compileform().exp.clone(),
                                false,
                                self.stack_limit,
                            )
                        })
                        .map(Some)
                }
            })
    }
}
