use std::rc::Rc;

use crate::compiler::compiler::DefaultCompilerOpts;
use crate::compiler::comptypes::{CompilerOpts, ModulePhase};
use crate::compiler::dialect::AcceptedDialect;
use crate::compiler::optimize::deinline::stepping_over_24;

#[test]
fn stepping_over_24_returns_true_for_module_compile_without_stepping() {
    let opts: Rc<dyn CompilerOpts> =
        Rc::new(DefaultCompilerOpts::new("*test*")).set_dialect(AcceptedDialect {
            stepping: None,
            strict: false,
            int_fix: false,
            extra_numeric_constants: false,
        });
    assert!(
        !stepping_over_24(opts.clone()),
        "without module_phase, stepping_over_24 should be false"
    );

    let module_opts = opts.set_module_phase(Some(ModulePhase::CommonPhase(false)));
    assert!(
        stepping_over_24(module_opts),
        "with module_phase set, stepping_over_24 should be true even when stepping is None"
    );
}

#[test]
fn stepping_over_24_returns_false_for_stepping_23() {
    let opts: Rc<dyn CompilerOpts> =
        Rc::new(DefaultCompilerOpts::new("*test*")).set_dialect(AcceptedDialect {
            stepping: Some(23),
            strict: false,
            int_fix: false,
            extra_numeric_constants: false,
        });
    assert!(!stepping_over_24(opts));
}

#[test]
fn stepping_over_24_returns_true_for_stepping_25() {
    let opts: Rc<dyn CompilerOpts> =
        Rc::new(DefaultCompilerOpts::new("*test*")).set_dialect(AcceptedDialect {
            stepping: Some(25),
            strict: true,
            int_fix: true,
            extra_numeric_constants: false,
        });
    assert!(stepping_over_24(opts));
}
