use num_bigint::ToBigInt;
use rand::Rng;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Debug;
use std::rc::Rc;

use crate::compiler::comptypes::{Binding, BindingPattern, BodyForm, LetData, LetFormKind};
use crate::compiler::sexp::{decode_string, enlist, SExp};
use crate::compiler::srcloc::Srcloc;

use crate::tests::compiler::fuzz::{simple_seeded_rng, SupportedOperators, ValueSpecification};

/// Make a set of numbered variables to use for fuzzing.
pub fn create_variable_set(_srcloc: Srcloc, vars: usize) -> BTreeSet<Vec<u8>> {
    (0..vars)
        .map(|n| format!("v{n}").bytes().collect())
        .collect()
}

#[derive(Default, Clone)]
/// An object that records information about some variables that are defined in
/// a complex, possibly multilevel assign form.  This object has the ability to
/// both generate this relationship based on randomness and also to extract
/// needed information about it.
///
/// In particular, we can ask it what variable this variable's definition
/// contributes to in the hierarchy (find_parent_of_var),
pub struct ComplexAssignExpression {
    pub toplevel: BTreeSet<Vec<u8>>,
    pub bindings: BTreeMap<Vec<u8>, Vec<Vec<u8>>>,
}

#[derive(Debug, Clone)]
pub struct GeneratedExpr {
    definition: Rc<ValueSpecification>,
    sexp: Rc<SExp>,
}

impl ComplexAssignExpression {
    fn fmtvar(
        &self,
        writer: &mut std::fmt::Formatter<'_>,
        lvl: usize,
        v: &[u8],
    ) -> Result<(), std::fmt::Error> {
        for _ in 0..(2 * lvl) {
            write!(writer, " ")?;
        }
        writeln!(writer, "{}:", decode_string(v))?;
        if let Some(children) = self.bindings.get(v) {
            for c in children.iter() {
                self.fmtvar(writer, lvl + 1, c)?;
            }
        }

        Ok(())
    }

    /// Find the parent of this var.
    /// If its definition occurs in an assign form which gives another variable
    /// its value, give the name of that variable.
    pub fn find_parent_of_var<'a>(&'a self, var: &[u8]) -> Option<&'a Vec<u8>> {
        for (parent, bindings) in self.bindings.iter() {
            if bindings.iter().any(|c| c == var) {
                return Some(parent);
            }
        }

        None
    }

    /// Find the path to this var.
    /// Given a variable name, return the list of parents whose values it appears
    /// in.
    pub fn find_path_to_var<'a>(&'a self, var: &[u8]) -> Vec<&'a Vec<u8>> {
        let mut result = Vec::new();
        let mut checking = var;
        while let Some(parent) = self.find_parent_of_var(checking) {
            checking = parent;
            result.push(parent);
        }
        result
    }

    /// Give the set of variables in scope for the definition of var.
    /// This allows us, given a variable name to determine which other variables
    /// could appear in an expression that gives the indicated var its value.
    pub fn variables_in_scope<'a>(&'a self, var: &[u8]) -> Vec<&'a Vec<u8>> {
        // If this variable itself use an assign form as its definition, then
        // all the innermost bindings are in scope.
        let mut result = self
            .bindings
            .get(var)
            .map(|v| v.iter().map(|e| &(*e)).collect())
            .unwrap_or_else(|| vec![]);

        // Get the parents of var.
        let parents = self.find_path_to_var(var);
        eprintln!("for {}", decode_string(var));
        for p in parents.iter() {
            eprintln!("parent {}", decode_string(p));
        }

        // If there are no parents, then the variables in scope are the toplevel
        // ones that appear before var.
        let mut from_scopes = if parents.is_empty() {
            self.toplevel
                .iter()
                .take_while(|t| *t != var)
                .map(|t| &(*t))
                .collect()
        } else {
            let mut scopes = Vec::new();
            let mut target = var;
            for p in parents.iter() {
                let p_borrowed: &[u8] = &p;
                if let Some(children) = self.bindings.get(p_borrowed) {
                    eprintln!("checking parent: {}", decode_string(p));
                    let mut appear_before_in_parent: Vec<&'a Vec<u8>> = children
                        .iter()
                        .take_while(|c| *c != target)
                        .map(|t| &(*t))
                        .collect();
                    eprintln!(
                        "{} in scope with parent {} of {}",
                        appear_before_in_parent.len(),
                        decode_string(p),
                        decode_string(var)
                    );
                    scopes.append(&mut appear_before_in_parent);
                    target = p;
                }
            }
            scopes
        };

        // Add the visible toplevel definitions if they won
        result.append(&mut from_scopes);
        result
    }

    /// Generate an expression to define one variable.
    /// Given a source of randomness, a desired complexity a set of variables that
    /// are bound from elsewhere (args) and a variable name, generate a candidate
    /// expression which respects the available scope at the definition of var.
    /// The resulting expression is given as sexp (to be compiled as chialisp)
    /// and as ValueSpecification to be separately evaulated.
    pub fn generate_expression<R: Rng>(
        &self,
        srcloc: &Srcloc,
        wanted_complexity: usize,
        rng: &mut R,
        args: &[Vec<u8>],
        var: &[u8],
    ) -> GeneratedExpr {
        let mut in_scope: Vec<&Vec<u8>> = args.iter().collect();
        let mut assignments_in_scope = self.variables_in_scope(var);
        in_scope.append(&mut assignments_in_scope);
        eprintln!("for {}", decode_string(var));
        for s in in_scope.iter() {
            eprintln!("in scope {}", decode_string(s));
        }

        let generate_constant = |rng: &mut R| {
            // Constant value
            let random_number: i8 = rng.random();
            let sexp = Rc::new(SExp::Integer(
                srcloc.clone(),
                random_number.to_bigint().unwrap(),
            ));
            let definition = Rc::new(ValueSpecification::ConstantValue(sexp.clone()));
            GeneratedExpr { definition, sexp }
        };

        let generate_reference = |rng: &mut R| {
            let variable_choice: usize = rng.random::<u64>() as usize;
            let variable = in_scope[variable_choice % in_scope.len()].to_vec();
            let var_sexp = Rc::new(SExp::Atom(srcloc.clone(), variable.clone()));
            let reference = Rc::new(ValueSpecification::VarRef(variable.clone()));

            GeneratedExpr {
                definition: reference,
                sexp: var_sexp,
            }
        };

        let generate_simple = |rng: &mut R| {
            if in_scope.is_empty() || rng.random() {
                generate_constant(rng)
            } else {
                generate_reference(rng)
            }
        };

        let mut result = generate_simple(rng);
        let complexity: usize = rng.random::<u64>() as usize;

        // Generate up to a certain number of operations.
        for _ in 0..(complexity % wanted_complexity) {
            // Generate the other branch.
            let other_result = generate_simple(rng);

            // Generate a binop.
            let random_op: SupportedOperators = rng.random();
            let (left, right) = if rng.random() {
                (result, other_result)
            } else {
                (other_result, result)
            };

            result = GeneratedExpr {
                sexp: Rc::new(enlist(
                    srcloc.clone(),
                    &[Rc::new(random_op.to_sexp(&srcloc)), left.sexp, right.sexp],
                )),
                definition: Rc::new(ValueSpecification::ClvmBinop(
                    random_op,
                    left.definition,
                    right.definition,
                )),
            };
        }

        result
    }

    // Create the assignments for the assign form.
    fn create_assign_form_for_var(
        &self,
        srcloc: &Srcloc,
        expressions: &BTreeMap<Vec<u8>, GeneratedExpr>,
        var: &[u8],
    ) -> BodyForm {
        let bound_in_var = self.bindings.get(var).cloned().unwrap_or_else(|| vec![]);
        let expr = expressions.get(var).unwrap();

        // No bindings below: just output the expr.
        if bound_in_var.is_empty() {
            return expr.definition.to_bodyform(srcloc);
        }

        let bindings: Vec<Rc<Binding>> = bound_in_var
            .iter()
            .map(|t| {
                let body = self.create_assign_form_for_var(srcloc, expressions, t);
                Rc::new(Binding {
                    loc: srcloc.clone(),
                    nl: srcloc.clone(),
                    body: Rc::new(body),
                    pattern: BindingPattern::Complex(Rc::new(SExp::Atom(
                        srcloc.clone(),
                        t.to_vec(),
                    ))),
                })
            })
            .collect();

        BodyForm::Let(
            LetFormKind::Assign,
            Box::new(LetData {
                kw: None,
                loc: srcloc.clone(),
                bindings,
                body: Rc::new(expr.definition.to_bodyform(srcloc)),
                inline_hint: None,
            }),
        )
    }

    /// Output the assign form whose structure is defined by this object and given
    /// a map of expressions to use for each variable's definition.
    pub fn create_assign_form(
        &self,
        srcloc: &Srcloc,
        expressions: &BTreeMap<Vec<u8>, GeneratedExpr>,
    ) -> BodyForm {
        assert!(!self.toplevel.is_empty());
        let last_top = self
            .toplevel
            .iter()
            .skip(self.toplevel.len() - 1)
            .next()
            .unwrap();
        let bindings: Vec<Rc<Binding>> = self
            .toplevel
            .iter()
            .map(|t| {
                let body = self.create_assign_form_for_var(srcloc, expressions, t);
                Rc::new(Binding {
                    loc: srcloc.clone(),
                    nl: srcloc.clone(),
                    body: Rc::new(body),
                    pattern: BindingPattern::Complex(Rc::new(SExp::Atom(
                        srcloc.clone(),
                        t.to_vec(),
                    ))),
                })
            })
            .collect();

        BodyForm::Let(
            LetFormKind::Assign,
            Box::new(LetData {
                loc: srcloc.clone(),
                kw: None,
                bindings,
                inline_hint: None,
                body: Rc::new(BodyForm::Value(SExp::Atom(
                    srcloc.clone(),
                    last_top.clone(),
                ))),
            }),
        )
    }
}

#[test]
fn test_complex_assign_expression() {
    let srcloc = Srcloc::start("*test*");
    let mut rng = simple_seeded_rng(0x02020202);
    let vars = create_variable_set(srcloc.clone(), 5);
    let structure_graph = create_complex_assign_expression(&mut rng, &vars);

    assert_eq!(
        format!("{structure_graph:?}"),
        indoc! {"
        v0:
          v4:
          v1:
        v2:
        v3:
        "}
    );
    assert_eq!(
        structure_graph.find_parent_of_var(b"v1"),
        Some(&b"v0".to_vec())
    );
    assert_eq!(structure_graph.find_path_to_var(b"v1"), vec![b"v0"]);
    assert_eq!(structure_graph.variables_in_scope(b"v1"), vec![b"v4"]);
    assert_eq!(
        structure_graph.variables_in_scope(b"v0"),
        vec![b"v4", b"v1"]
    );
    assert_eq!(
        structure_graph.variables_in_scope(b"v3"),
        vec![b"v0", b"v2"]
    );
    let g3 = structure_graph.generate_expression(
        &srcloc,
        5,
        &mut rng,
        &[b"a1".to_vec(), b"a2".to_vec()],
        b"v3",
    );
    assert_eq!(g3.sexp.to_string(), "(18 (16 122 (17 a1 43)) -53)");
    assert_eq!(
        g3.definition,
        Rc::new(ValueSpecification::ClvmBinop(
            SupportedOperators::Times,
            Rc::new(ValueSpecification::ClvmBinop(
                SupportedOperators::Plus,
                Rc::new(ValueSpecification::ConstantValue(Rc::new(SExp::Integer(
                    srcloc.clone(),
                    122.to_bigint().unwrap()
                )))),
                Rc::new(ValueSpecification::ClvmBinop(
                    SupportedOperators::Minus,
                    Rc::new(ValueSpecification::VarRef(b"a1".to_vec())),
                    Rc::new(ValueSpecification::ConstantValue(Rc::new(SExp::Integer(
                        srcloc.clone(),
                        43.to_bigint().unwrap()
                    ))))
                )),
            )),
            Rc::new(ValueSpecification::ConstantValue(Rc::new(SExp::Integer(
                srcloc.clone(),
                -53.to_bigint().unwrap()
            ))))
        ))
    );
    let g1 = structure_graph.generate_expression(&srcloc, 10, &mut rng, &[b"a1".to_vec()], b"v1");
    assert_eq!(
        g1.sexp.to_string(),
        "(16 v4 (16 (17 (17 (16 v4 v4) 29) 109) a1))"
    );
    assert_eq!(
        g1.definition,
        Rc::new(ValueSpecification::ClvmBinop(
            SupportedOperators::Plus,
            Rc::new(ValueSpecification::VarRef(b"v4".to_vec())),
            Rc::new(ValueSpecification::ClvmBinop(
                SupportedOperators::Plus,
                Rc::new(ValueSpecification::ClvmBinop(
                    SupportedOperators::Minus,
                    Rc::new(ValueSpecification::ClvmBinop(
                        SupportedOperators::Minus,
                        Rc::new(ValueSpecification::ClvmBinop(
                            SupportedOperators::Plus,
                            Rc::new(ValueSpecification::VarRef(b"v4".to_vec())),
                            Rc::new(ValueSpecification::VarRef(b"v4".to_vec()))
                        )),
                        Rc::new(ValueSpecification::ConstantValue(Rc::new(SExp::Integer(
                            srcloc.clone(),
                            29.to_bigint().unwrap()
                        ))))
                    )),
                    Rc::new(ValueSpecification::ConstantValue(Rc::new(SExp::Integer(
                        srcloc.clone(),
                        109.to_bigint().unwrap()
                    ))))
                )),
                Rc::new(ValueSpecification::VarRef(b"a1".to_vec()))
            ))
        ))
    );
    let free_vars: Vec<Vec<u8>> = g1.definition.get_free_vars().into_iter().collect();
    assert_eq!(free_vars, vec![b"a1".to_vec(), b"v4".to_vec()]);

    // Generate simple constant expressions to make it clear how these relate to
    // the definitions that are emitted.
    let expressions: BTreeMap<Vec<u8>, GeneratedExpr> = (0..=4)
        .map(|n| {
            let name = format!("v{n}").as_bytes().to_vec();
            let sexp = Rc::new(SExp::Integer(srcloc.clone(), n.to_bigint().unwrap()));
            let expr = GeneratedExpr {
                definition: Rc::new(ValueSpecification::ConstantValue(sexp.clone())),
                sexp,
            };
            (name, expr)
        })
        .collect();

    let assign_form = structure_graph.create_assign_form(&srcloc, &expressions);
    // Each variable is defined as a constant with the same number in this
    // example elaboration.
    assert_eq!(
        assign_form.to_sexp().to_string(),
        "(assign v0 (assign v4 (q . 4) v1 (q . 1) (q)) v2 (q . 2) v3 (q . 3) v3)"
    );
}

impl Debug for ComplexAssignExpression {
    fn fmt(&self, writer: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        for t in self.toplevel.iter() {
            self.fmtvar(writer, 0, t)?;
        }
        Ok(())
    }
}

/// Create a complex assign structure and provide methods for generating
/// expressions that can be a candidate definition for it.
/// Useful for fuzzing code that relates to assign forms.
pub fn create_complex_assign_expression<R: Rng>(
    rng: &mut R,
    v: &BTreeSet<Vec<u8>>,
) -> ComplexAssignExpression {
    let mut v_start = v.clone();
    let mut usage = ComplexAssignExpression::default();

    while !v_start.is_empty() {
        // Choose a variable.
        let chosen_idx: usize = rng.random::<u64>() as usize;
        let chosen = v_start
            .iter()
            .skip(chosen_idx % v_start.len())
            .next()
            .cloned()
            .unwrap();

        // Decide whether it's toplevel (we always choose one if there are
        // no toplevel choices.
        let coin_flip_toplevel: usize = rng.random::<u64>() as usize;
        if (usage.toplevel.is_empty() || (coin_flip_toplevel % 3) == 0) && usage.toplevel.len() < 5
        {
            // if so, copy it to the toplevel set.
            usage.toplevel.insert(chosen.clone());
        } else {
            // otherwise, choose a key from result, add it there.
            let parent_idx: usize = rng.random::<u64>() as usize;
            let parent = usage
                .bindings
                .keys()
                .skip(parent_idx % usage.bindings.len())
                .next()
                .cloned()
                .unwrap();
            if let Some(children) = usage.bindings.get_mut(&parent) {
                children.push(chosen.clone());
            }
        }

        // Remove the chosen var from v_start, add an empty entry to result.
        v_start.remove(&chosen);
        usage.bindings.insert(chosen, Vec::new());
    }

    usage
}
