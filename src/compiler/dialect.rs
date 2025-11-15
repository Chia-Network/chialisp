use std::collections::HashMap;

use clvmr::allocator::{Allocator, NodePtr, SExp};

use crate::classic::clvm::sexp::proper_list;

use crate::compiler::sexp::decode_string;

/// Stepping 21 and 22 do optimization in special ways with flags
/// I made this more general for other dialects, starting at
/// BASE_STEPPING, all of which should use similar optimizer objects.
pub const OPT_STRATEGY_BASE_STEPPING: i32 = 23;
/// The maximum stepping of the language so far.
pub const MAX_STEPPING: i32 = 24;

/// Specifying how the language is spoken.
#[derive(Clone, Debug, Default)]
pub struct AcceptedDialect {
    pub stepping: Option<i32>,
    pub strict: bool,
    pub int_fix: bool,
}

/// A package containing the content we should insert when a dialect include is
/// used, plus the compilation flags.
#[derive(Clone, Debug)]
pub struct DialectDescription {
    pub accepted: AcceptedDialect,
    pub content: String,
}

lazy_static! {
    pub static ref KNOWN_DIALECTS: HashMap<String, DialectDescription> = {
        let mut dialects: HashMap<String, DialectDescription> = HashMap::new();
        let dialect_list = [
            (
                "*standard-cl-21*",
                DialectDescription {
                    accepted: AcceptedDialect {
                        stepping: Some(21),
                        ..AcceptedDialect::default()
                    },
                    content: indoc! {"(
                    (defconstant *chialisp-version* 21)
                )"}
                    .to_string(),
                },
            ),
            (
                "*strict-cl-21*",
                DialectDescription {
                    accepted: AcceptedDialect {
                        stepping: Some(21),
                        strict: true,
                        int_fix: false,
                    },
                    content: indoc! {"(
                    (defconstant *chialisp-version* 22)
                )"}
                    .to_string(),
                },
            ),
            (
                "*standard-cl-22*",
                DialectDescription {
                    accepted: AcceptedDialect {
                        stepping: Some(22),
                        strict: false,
                        int_fix: false,
                    },
                    content: indoc! {"(
                    (defconstant *chialisp-version* 22)
                )"}
                    .to_string(),
                },
            ),
            (
                "*standard-cl-23*",
                DialectDescription {
                    accepted: AcceptedDialect {
                        stepping: Some(23),
                        strict: true,
                        int_fix: false,
                    },
                    content: indoc! {"(
                    (defconstant *chialisp-version* 23)
                )"}
                    .to_string(),
                },
            ),
            (
                "*standard-cl-23.1*",
                DialectDescription {
                    accepted: AcceptedDialect {
                        stepping: Some(23),
                        strict: true,
                        int_fix: true,
                    },
                    content: indoc! {"(
                    (defconstant *chialisp-version* 23)
                )"}
                    .to_string(),
                },
            ),
            (
                "*standard-cl-24*",
                DialectDescription {
                    accepted: AcceptedDialect {
                        stepping: Some(24),
                        strict: true,
                        int_fix: true,
                    },
                    content: indoc! {"(
                    (defconstant *chialisp-version* 24)
                )"}
                    .to_string(),
                },
            ),
            (
                "*standard-cl-25*",
                DialectDescription {
                    accepted: AcceptedDialect {
                        stepping: Some(25),
                        strict: true,
                        int_fix: true,
                    },
                    content: indoc! {"(
                    (defconstant *chialisp-version* 25)
                )"}
                    .to_string(),
                },
            ),
        ];
        for (n, v) in dialect_list.iter() {
            dialects.insert(n.to_string(), v.clone());
        }
        dialects
    };
}

fn include_dialect(allocator: &Allocator, e: &[NodePtr]) -> Option<AcceptedDialect> {
    let include_keyword_sexp = e[0];
    let name_sexp = e[1];
    if let (SExp::Atom, SExp::Atom) = (
        allocator.sexp(include_keyword_sexp),
        allocator.sexp(name_sexp),
    ) {
        let include_keyword_atom = allocator.atom(include_keyword_sexp);
        let name_atom = allocator.atom(name_sexp);
        if include_keyword_atom.as_ref() == b"include" {
            if let Some(dialect) = KNOWN_DIALECTS.get(&decode_string(name_atom.as_ref())) {
                return Some(dialect.accepted.clone());
            }
        }
    }

    None
}

// Now return more parameters about the "modern" dialect, including in the future,
// strictness.  This will allow us to support the transition to modern macros which
// in turn allow us to turn on strictness in variable naming.  Often multiple moves
// are needed to get from one point to another and there's a tension between
// unitary changes and smaller PRs which do fewer things by themselves.  This is
// part of a broader narrative, which many requested that sets us on the path of
// being able to include more information in the dialect result.
pub fn detect_modern(allocator: &mut Allocator, sexp: NodePtr) -> AcceptedDialect {
    let mut result = AcceptedDialect::default();

    if let Some(l) = proper_list(allocator, sexp, true) {
        for elt in l.iter() {
            let detect_modern_result = detect_modern(allocator, *elt);
            if detect_modern_result.stepping.is_some() {
                result = detect_modern_result;
                break;
            }

            match proper_list(allocator, *elt, true) {
                None => {
                    continue;
                }

                Some(e) => {
                    if e.len() != 2 {
                        continue;
                    }

                    if let Some(dialect) = include_dialect(allocator, &e) {
                        result = dialect;
                        break;
                    }
                }
            }
        }
    }

    result
}
