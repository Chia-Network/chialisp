use std::borrow::Borrow;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::compiler::optimize::SyntheticType;
use crate::compiler::sexp::enlist;
use crate::compiler::srcloc::Srcloc;
use crate::compiler::{BodyForm, CompileForm, DefunData, HelperForm, SExp};

#[derive(Debug)]
pub enum DepgraphKind {
    UserNonInline,
    UserInline,
    Synthetic(SyntheticType),
}

pub struct FunctionDependencyEntry {
    pub loc: Srcloc,
    pub name: Vec<u8>,
    pub status: DepgraphKind,
    pub depends_on: HashSet<Vec<u8>>,
    pub is_depended_on_by: HashSet<Vec<u8>>,
}

impl FunctionDependencyEntry {
    pub fn to_sexp(&self) -> Rc<SExp> {
        let depends_on: Vec<Rc<SExp>> = self
            .depends_on
            .iter()
            .map(|x| Rc::new(SExp::Atom(self.loc.clone(), x.clone())))
            .collect();

        let is_depended_on_by: Vec<Rc<SExp>> = self
            .is_depended_on_by
            .iter()
            .map(|x| Rc::new(SExp::Atom(self.loc.clone(), x.clone())))
            .collect();

        Rc::new(enlist(
            self.loc.clone(),
            &[
                Rc::new(SExp::Atom(self.loc.clone(), self.name.clone())),
                Rc::new(SExp::Atom(
                    self.loc.clone(),
                    format!("{:?}", self.status).as_bytes().to_vec(),
                )),
                Rc::new(SExp::Atom(
                    self.loc.clone(),
                    "depends_on".as_bytes().to_vec(),
                )),
                Rc::new(enlist(self.loc.clone(), &depends_on)),
                Rc::new(SExp::Atom(
                    self.loc.clone(),
                    "is_depended_on_by".as_bytes().to_vec(),
                )),
                Rc::new(enlist(self.loc.clone(), &is_depended_on_by)),
            ],
        ))
    }

    pub fn new(name: &[u8], loc: Srcloc, status: DepgraphKind) -> Self {
        FunctionDependencyEntry {
            loc,
            name: name.to_vec(),
            status,
            depends_on: HashSet::default(),
            is_depended_on_by: HashSet::default(),
        }
    }
}

pub struct FunctionDependencyGraph {
    pub loc: Srcloc,
    pub helpers: HashMap<Vec<u8>, FunctionDependencyEntry>,
}

fn status_from_defun(inline: bool, defun: &DefunData) -> DepgraphKind {
    match (inline, defun.synthetic.as_ref()) {
        (true, None) => DepgraphKind::UserInline,
        (false, None) => DepgraphKind::UserNonInline,
        (_, Some(st)) => DepgraphKind::Synthetic(st.clone()),
    }
}

impl FunctionDependencyGraph {
    /// Find leaf functions.
    pub fn leaves(&self) -> HashSet<Vec<u8>> {
        self.helpers
            .iter()
            .filter(|(_k, h)| h.depends_on.is_empty())
            .map(|(k, _h)| k.clone())
            .collect()
    }

    pub fn parents(&self, helper: &[u8]) -> Option<HashSet<Vec<u8>>> {
        self.helpers.get(helper).map(|h| {
            let mut result_set = h.is_depended_on_by.clone();
            result_set.remove(helper);
            result_set
        })
    }

    pub fn get_full_depended_on_by(
        &self,
        depended_on_by: &mut HashSet<Vec<u8>>,
        helper_name: &[u8],
    ) {
        if let Some(helper) = self.helpers.get(helper_name) {
            for d in helper.is_depended_on_by.iter() {
                if !depended_on_by.contains(d) {
                    depended_on_by.insert(d.to_vec());
                    self.get_full_depended_on_by(depended_on_by, d);
                }
            }
        }
    }

    pub fn get_full_depends_on(&self, depends_on_fun: &mut HashSet<Vec<u8>>, helper_name: &[u8]) {
        if let Some(helper) = self.helpers.get(helper_name) {
            for d in helper.depends_on.iter() {
                if !depends_on_fun.contains(d) {
                    depends_on_fun.insert(d.clone());
                    self.get_full_depends_on(depends_on_fun, d);
                }
            }
        }
    }

    fn add_depends_on_relationship(&mut self, helper_name: &[u8], name: &[u8]) {
        if !self.helpers.contains_key(helper_name) || !self.helpers.contains_key(name) {
            return;
        }

        if let Some(function_entry) = self.helpers.get_mut(helper_name) {
            function_entry.depends_on.insert(name.to_vec());
        }

        if let Some(function_entry) = self.helpers.get_mut(name) {
            function_entry
                .is_depended_on_by
                .insert(helper_name.to_vec());
        }
    }

    fn process_expr(&mut self, helper_name: &[u8], expr: Rc<BodyForm>) {
        match expr.borrow() {
            BodyForm::Value(SExp::Atom(_, name)) => {
                // This introduces a function dependency.
                self.add_depends_on_relationship(helper_name, name);
            }
            BodyForm::Value(_) => {}
            BodyForm::Let(_, letdata) => {
                for b in letdata.bindings.iter() {
                    self.process_expr(helper_name, b.body.clone());
                }

                self.process_expr(helper_name, letdata.body.clone());
            }
            BodyForm::Call(_, args, tail) => {
                if let Some(t) = tail.as_ref() {
                    self.process_expr(helper_name, t.clone());
                }

                for a in args.iter() {
                    self.process_expr(helper_name, a.clone());
                }
            }
            BodyForm::Lambda(ldata) => {
                self.process_expr(helper_name, ldata.captures.clone());
                self.process_expr(helper_name, ldata.body.clone());
            }
            BodyForm::Quoted(_) => {}
            BodyForm::Mod(_, _) => {}
        }
    }

    fn process_helper(&mut self, h: &HelperForm) {
        if let HelperForm::Defun(_, defun) = h {
            self.process_expr(&defun.name, defun.body.clone());
        }
    }

    pub fn new(program: &CompileForm) -> Self {
        let mut helpers: HashMap<Vec<u8>, FunctionDependencyEntry> = HashMap::new();

        for h in program.helpers.iter() {
            if let HelperForm::Defun(inline, d) = h {
                helpers.insert(
                    h.name().to_vec(),
                    FunctionDependencyEntry::new(h.name(), h.loc(), status_from_defun(*inline, d)),
                );
            }
        }

        let mut graph = FunctionDependencyGraph {
            loc: program.loc.clone(),
            helpers,
        };

        for h in program.helpers.iter() {
            graph.process_helper(h);
        }

        graph
    }

    pub fn to_sexp(&self) -> Rc<SExp> {
        let helpers: Vec<Rc<SExp>> = self.helpers.values().map(|v| v.to_sexp()).collect();

        Rc::new(enlist(self.loc.clone(), &helpers))
    }
}
