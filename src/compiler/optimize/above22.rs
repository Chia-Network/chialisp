use std::borrow::Borrow;
use std::collections::HashMap;
use std::rc::Rc;

use clvmr::allocator::Allocator;

use crate::classic::clvm_tools::stages::stage_0::TRunProgram;

#[cfg(any(test, feature = "fuzz"))]
use crate::compiler::compiler::FUZZ_TEST_PRE_CSE_MERGE_FIX_FLAG;
use crate::compiler::comptypes::{
    BodyForm, CompileErr, CompileForm, CompilerOpts, DefunData, HelperForm, PrimaryCodegen,
};
use crate::compiler::frontend::compute_live_helpers;
use crate::compiler::optimize::brief::brief_path_selection;
use crate::compiler::optimize::cse::cse_optimize_bodyform;
use crate::compiler::optimize::deinline::deinline_opt;
use crate::compiler::optimize::double_apply::remove_double_apply;
use crate::compiler::optimize::{
    null_optimization, optimize_expr, run_optimizer, CompileContextWrapper, Optimization,
};
use crate::compiler::sexp::SExp;
use crate::compiler::StartOfCodegenOptimization;

/// Captures the strategy for cl23 and above.
/// Until formally released, we can take action in here.
#[derive(Default, Clone)]
pub struct Strategy23 {}

#[cfg(any(test, feature = "fuzz"))]
fn enable_cse_merge_fix_so_can_be_disabled_for_tests(opts: Rc<dyn CompilerOpts>) -> bool {
    !opts
        .diag_flags()
        .contains(&FUZZ_TEST_PRE_CSE_MERGE_FIX_FLAG)
}

#[cfg(not(any(test, feature = "fuzz")))]
fn enable_cse_merge_fix_so_can_be_disabled_for_tests(_opts: Rc<dyn CompilerOpts>) -> bool {
    true
}

impl Strategy23 {
    pub fn new() -> Self {
        Strategy23 {}
    }
}

impl Optimization for Strategy23 {
    fn frontend_optimization(
        &mut self,
        _allocator: &mut Allocator,
        _runner: Rc<dyn TRunProgram>,
        opts: Rc<dyn CompilerOpts>,
        mut p0: CompileForm,
    ) -> Result<CompileForm, CompileErr> {
        let mut rebuilt_helpers = Vec::new();
        let enable_merge_disable_for_tests =
            enable_cse_merge_fix_so_can_be_disabled_for_tests(opts.clone());
        for h in p0.helpers.iter() {
            if let HelperForm::Defun(inline, d) = h {
                let function_body = cse_optimize_bodyform(
                    &h.loc(),
                    h.name(),
                    enable_merge_disable_for_tests,
                    d.body.borrow(),
                )?;
                // Ok we've got a body that is now a let stack.
                let new_helper = HelperForm::Defun(
                    *inline,
                    Box::new(DefunData {
                        body: Rc::new(function_body),
                        ..*d.clone()
                    }),
                );

                rebuilt_helpers.push(new_helper);
            } else {
                rebuilt_helpers.push(h.clone());
            }
        }

        p0.helpers = rebuilt_helpers;
        Ok(p0)
    }

    fn post_desugar_optimization(
        &mut self,
        allocator: &mut Allocator,
        runner: Rc<dyn TRunProgram>,
        opts: Rc<dyn CompilerOpts>,
        cf: CompileForm,
    ) -> Result<CompileForm, CompileErr> {
        let mut symbols = HashMap::new();
        let mut wrapper =
            CompileContextWrapper::new(allocator, runner, &mut symbols, self.duplicate());
        deinline_opt(&mut wrapper.context, opts.clone(), cf)
    }

    fn start_of_codegen_optimization(
        &mut self,
        allocator: &mut Allocator,
        runner: Rc<dyn TRunProgram>,
        opts: Rc<dyn CompilerOpts>,
        mut to_optimize: StartOfCodegenOptimization,
    ) -> Result<StartOfCodegenOptimization, CompileErr> {
        let new_helpers: Vec<HelperForm> = to_optimize
            .program
            .helpers
            .iter()
            .map(|h| {
                if let HelperForm::Defun(inline, defun) = h {
                    let new_body = optimize_expr(
                        allocator,
                        opts.clone(),
                        runner.clone(),
                        &to_optimize.code_generator,
                        defun.body.clone(),
                    )
                    .map(|x| x.1)
                    .unwrap_or_else(|| defun.body.clone());
                    HelperForm::Defun(
                        *inline,
                        Box::new(DefunData {
                            body: new_body,
                            ..*defun.clone()
                        }),
                    )
                } else {
                    h.clone()
                }
            })
            .collect();

        to_optimize.program.helpers = new_helpers;
        to_optimize.program.exp = optimize_expr(
            allocator,
            opts.clone(),
            runner.clone(),
            &to_optimize.code_generator,
            to_optimize.program.exp.clone(),
        )
        .map(|x| x.1)
        .unwrap_or_else(|| to_optimize.program.exp.clone());

        let used_helpers = compute_live_helpers(
            opts,
            &to_optimize.program.helpers,
            to_optimize.program.exp.clone(),
        );

        to_optimize.program.helpers = used_helpers;

        Ok(to_optimize)
    }

    fn macro_optimization(
        &mut self,
        allocator: &mut Allocator,
        runner: Rc<dyn TRunProgram>,
        _opts: Rc<dyn CompilerOpts>,
        code: Rc<SExp>,
    ) -> Result<Rc<SExp>, CompileErr> {
        run_optimizer(allocator, runner, code)
    }

    fn defun_body_optimization(
        &mut self,
        allocator: &mut Allocator,
        runner: Rc<dyn TRunProgram>,
        opts: Rc<dyn CompilerOpts>,
        codegen: &PrimaryCodegen,
        defun: &DefunData,
    ) -> Result<Rc<BodyForm>, CompileErr> {
        // Run optimizer on frontend style forms.
        let mut changed = true;
        let mut new_body = defun.body.clone();

        while changed {
            if let Some((did_optimize, optimized_form)) = optimize_expr(
                allocator,
                opts.clone(),
                runner.clone(),
                codegen,
                new_body.clone(),
            ) {
                changed = did_optimize && optimized_form.to_sexp() != new_body.to_sexp();
                new_body = optimized_form;
            } else {
                changed = false;
            }
        }

        Ok(new_body)
    }

    fn post_codegen_function_optimize(
        &mut self,
        _allocator: &mut Allocator,
        _runner: Rc<dyn TRunProgram>,
        _opts: Rc<dyn CompilerOpts>,
        _helper: Option<&HelperForm>,
        code: Rc<SExp>,
    ) -> Result<Rc<SExp>, CompileErr> {
        let (null_worked, result) = null_optimization(code.clone(), true);
        let (double_worked, dbl_result) = remove_double_apply(result, true);
        let (brief_worked, brief_result) = brief_path_selection(dbl_result);
        if null_worked || double_worked || brief_worked {
            Ok(brief_result)
        } else {
            Ok(code)
        }
    }

    fn pre_final_codegen_optimize(
        &mut self,
        allocator: &mut Allocator,
        runner: Rc<dyn TRunProgram>,
        opts: Rc<dyn CompilerOpts>,
        codegen: &PrimaryCodegen,
    ) -> Result<Rc<BodyForm>, CompileErr> {
        let mut changed = true;
        let mut new_body = codegen.final_expr.clone();

        while changed {
            if let Some((did_optimize, optimized_form)) = optimize_expr(
                allocator,
                opts.clone(),
                runner.clone(),
                codegen,
                new_body.clone(),
            ) {
                changed = did_optimize && optimized_form.to_sexp() != new_body.to_sexp();
                new_body = optimized_form;
            } else {
                changed = false;
            }
        }

        Ok(new_body)
    }

    fn post_codegen_output_optimize(
        &mut self,
        _opts: Rc<dyn CompilerOpts>,
        generated: SExp,
    ) -> Result<SExp, CompileErr> {
        let (null_worked, result) = null_optimization(Rc::new(generated.clone()), true);
        let (double_worked, dbl_result) = remove_double_apply(result, true);
        let (brief_worked, brief_result) = brief_path_selection(dbl_result);
        if null_worked || double_worked || brief_worked {
            let borrowed: &SExp = brief_result.borrow();
            Ok(borrowed.clone())
        } else {
            Ok(generated)
        }
    }

    fn duplicate(&self) -> Box<dyn Optimization> {
        Box::new(self.clone())
    }
}
