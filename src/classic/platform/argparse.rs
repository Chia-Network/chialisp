#![allow(
    clippy::disallowed_names,
    clippy::redundant_clone,
    clippy::trivially_copy_pass_by_ref
)]

use std::borrow::Borrow;
use std::cmp::PartialEq;
use std::collections::HashMap;
use std::rc::Rc;

use crate::classic::clvm::__type_compatibility__::Record;
use crate::util::{index_of_match, skip_leading};

#[derive(PartialEq, Debug, Clone, Eq)]
pub enum TArgOptionAction {
    Store,
    StoreTrue,
    Append,
}

#[derive(PartialEq, Debug, Clone, Eq)]
pub enum NArgsSpec {
    KleeneStar,
    Plus,
    Optional,
    Definite(usize),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArgumentValue {
    ArgString(Option<String>, String),
    ArgInt(i64),
    ArgBool(bool),
    ArgArray(Vec<ArgumentValue>),
}

pub trait ArgumentValueConv {
    fn convert(&self, s: &str) -> Result<ArgumentValue, String>;
}

struct EmptyConversion {}
impl ArgumentValueConv for EmptyConversion {
    fn convert(&self, s: &str) -> Result<ArgumentValue, String> {
        Ok(ArgumentValue::ArgString(None, s.to_string()))
    }
}

pub struct IntConversion {
    help_messager: Rc<dyn Fn() -> String>,
}

impl ArgumentValueConv for IntConversion {
    fn convert(&self, v: &str) -> Result<ArgumentValue, String> {
        match v.parse::<i64>() {
            Ok(n) => Ok(ArgumentValue::ArgInt(n)),
            _ => {
                let m: &dyn Fn() -> String = self.help_messager.borrow();
                let usage = m();
                Err(format!("{usage}\n\nError: Invalid parameter: {v}"))
            }
        }
    }
}

impl IntConversion {
    pub fn new(f: Rc<dyn Fn() -> String>) -> Self {
        IntConversion { help_messager: f }
    }
}

#[derive(Clone)]
pub struct Argument {
    action: TArgOptionAction,
    typeofarg: Rc<dyn ArgumentValueConv>,
    default: Option<ArgumentValue>,
    help: String,
    nargs: Option<NArgsSpec>,
}

impl Argument {
    pub fn new() -> Self {
        Argument {
            action: TArgOptionAction::Store,
            typeofarg: Rc::new(EmptyConversion {}),
            default: None,
            help: "".to_string(),
            nargs: None,
        }
    }

    pub fn set_action(self, a: TArgOptionAction) -> Self {
        let mut s = self;
        s.action = a;
        s
    }
    pub fn set_type(self, t: Rc<dyn ArgumentValueConv>) -> Self {
        let mut s = self;
        s.typeofarg = t;
        s
    }
    pub fn set_default(self, v: ArgumentValue) -> Self {
        let mut s = self;
        s.default = Some(v);
        s
    }
    pub fn set_help(self, h: String) -> Self {
        let mut s = self;
        s.help = h;
        s
    }
    pub fn set_n_args(self, n: NArgsSpec) -> Self {
        let mut s = self;
        s.nargs = Some(n);
        s
    }
}

impl Default for Argument {
    fn default() -> Self {
        Argument::new()
    }
}

#[derive(Clone)]
pub struct Arg {
    names: Vec<String>,
    options: Argument,
}

pub fn is_optional(arg: &str) -> bool {
    #[allow(clippy::single_char_pattern)]
    arg.starts_with("-")
}

#[derive(Debug, Clone)]
pub struct TArgumentParserProps {
    pub description: String,
    pub prog: String,
}

#[derive(Clone)]
pub struct ArgumentParser {
    prog: String,
    // desc: String,
    positional_args: Vec<Arg>,
    optional_args: Vec<Arg>,
}

impl ArgumentParser {
    pub fn new(props: Option<TArgumentParserProps>) -> ArgumentParser {
        let mut start = ArgumentParser {
            prog: props
                .as_ref()
                .map(|x| x.prog.clone())
                .unwrap_or_else(|| "prog".to_string()),
            // FIXME: use desc
            // desc: props
            //    .as_ref()
            //    .map(|x| x.description.clone())
            //    .unwrap_or_else(|| "".to_string()),
            positional_args: vec![],
            optional_args: vec![],
        };

        match props {
            None => {}
            Some(_props) => {
                start.add_argument(
                    vec!["-h".to_string(), "--help".to_string()],
                    Argument::new()
                        .set_help("Show help message".to_string())
                        .set_action(TArgOptionAction::StoreTrue),
                );
            }
        }

        start
    }

    pub fn add_argument(&mut self, arg_name: Vec<String>, options: Argument) {
        if arg_name.len() == 1 && !is_optional(&arg_name[0]) {
            self.positional_args.push(Arg {
                names: arg_name,
                options,
            });
        } else {
            self.optional_args.push(Arg {
                names: arg_name,
                options,
            });
        }
    }

    pub fn parse_args(
        &mut self,
        args: &[String],
    ) -> Result<HashMap<String, ArgumentValue>, String> {
        let normalized_args = self.normalize_args(args);
        let mut params: Record<String, ArgumentValue> = HashMap::new();

        // Set default value
        for optional_arg_k in &self.optional_args {
            let default_value = &optional_arg_k.options.default;

            if let Some(dv) = default_value {
                match self.get_optional_arg_name(optional_arg_k) {
                    Ok(name) => {
                        params.insert(name, dv.clone());
                    }
                    Err(e) => {
                        return Err(e);
                    }
                }
            }
        }

        let mut input_positional_args: Vec<String> = vec![];
        let mut ioff = 0;

        if !normalized_args.is_empty() {
            for i in 0..normalized_args.len() {
                if i + ioff >= normalized_args.len() {
                    break;
                }

                let arg = &normalized_args[i + ioff];

                // positional argument
                if !is_optional(arg) {
                    input_positional_args.push(arg.clone());
                    continue;
                }

                let optional_arg_idx = index_of_match(
                    |a: &Arg| index_of_match(|a: &String| arg.starts_with(a), &a.names) >= 0,
                    &self.optional_args,
                );

                if optional_arg_idx < 0 {
                    let usage = self.compile_help_messages();
                    return Err(format!("{usage}\n\nError: Unknown option: {arg}"));
                }

                let optional_arg = &self.optional_args[optional_arg_idx as usize];

                let name = match self.get_optional_arg_name(optional_arg) {
                    Ok(n) => n,
                    Err(e) => {
                        return Err(e);
                    }
                };

                if optional_arg.options.action == TArgOptionAction::StoreTrue {
                    params.insert(name, ArgumentValue::ArgBool(true));
                    continue;
                }

                let converter = self.get_converter(Some(optional_arg.options.typeofarg.clone()));

                ioff += 1;

                // Simplify and fix the ability to read outside the vector bounds
                // when an optional argument is given without a value.
                let value = if i + ioff >= normalized_args.len()
                    || (normalized_args[i + ioff].is_empty()
                        && optional_arg.options.default.is_none())
                {
                    let usage = self.compile_help_messages();
                    return Err(format!("{usage}\n\nError: {name} requires a value"));
                } else {
                    &normalized_args[i + ioff]
                };
                if optional_arg.options.action == TArgOptionAction::Store {
                    if let Ok(c) = converter.convert(value) {
                        params.insert(name, c);
                    } else if let Some(v) = &optional_arg.options.default {
                        params.insert(name, v.clone());
                    }
                } else if optional_arg.options.action == TArgOptionAction::Append {
                    match params.get(&name) {
                        Some(ArgumentValue::ArgArray(l)) => match converter.convert(value) {
                            Ok(v) => {
                                let mut lcopy = l.clone();
                                lcopy.push(v);
                                params.insert(name, ArgumentValue::ArgArray(lcopy));
                            }
                            _ => {
                                if let Some(v) = &optional_arg.options.default {
                                    let mut lcopy = l.clone();
                                    lcopy.push(v.clone());
                                    params.insert(name, ArgumentValue::ArgArray(lcopy));
                                }
                            }
                        },
                        _ => {
                            if let Ok(v) = converter.convert(value) {
                                params.insert(name, ArgumentValue::ArgArray(vec![v]));
                            } else if let Some(v) = &optional_arg.options.default {
                                params.insert(name, ArgumentValue::ArgArray(vec![v.clone()]));
                            }
                        }
                    }
                } else {
                    let usage = self.compile_help_messages();
                    return Err(format!(
                        "{usage}\n\nError: Unknown action: {:?}",
                        optional_arg.options.action
                    ));
                }
            }
        }

        let mut i = 0;
        for positional_arg_k in &self.positional_args {
            let empty_string = "".to_string();
            let input_arg = if i < input_positional_args.len() {
                &input_positional_args[i]
            } else {
                &empty_string
            };

            let name = &positional_arg_k.names[0];
            let nargs = &positional_arg_k.options.nargs;
            let converter = self.get_converter(Some(positional_arg_k.options.typeofarg.clone()));

            match nargs {
                None => {
                    if let Ok(v) = converter.convert(input_arg) {
                        params.insert(name.to_string(), v);
                        i += 1;
                    }
                }
                Some(NArgsSpec::Definite(nargs)) => {
                    for _j in 0..nargs - 1 {
                        if i >= input_positional_args.len() {
                            let usage = self.compile_help_messages();
                            return Err(format!(
                                "{usage}\n\nError: Requires {nargs} positional arguments but got {i}"
                            ));
                        }

                        let input_arg = &input_positional_args[i];
                        if let Some(ArgumentValue::ArgArray(l)) = params.get(name) {
                            if let Ok(v) = converter.convert(input_arg) {
                                let mut lcopy = l.clone();
                                lcopy.push(v);
                                params.insert(name.to_string(), ArgumentValue::ArgArray(lcopy));
                            }
                        } else if let Ok(v) = converter.convert(input_arg) {
                            params.insert(name.to_string(), ArgumentValue::ArgArray(vec![v]));
                        }
                        i += 1;
                    }
                }
                Some(NArgsSpec::Optional) => {
                    if i >= input_positional_args.len() {
                        if let Ok(v) = converter.convert("") {
                            params.insert(name.to_string(), v);
                        }
                    } else if let Ok(v) = converter.convert(input_arg) {
                        params.insert(name.to_string(), v);
                    }
                    i += 1;
                }

                _ => {
                    if i >= input_positional_args.len() {
                        if nargs == &Some(NArgsSpec::Plus) {
                            let usage = self.compile_help_messages();
                            return Err(format!(
                                "{usage}\n\nError: The following arguments are required: {name}"
                            ));
                        }

                        params.insert(name.to_string(), ArgumentValue::ArgArray(vec![]));

                        i += 1;
                        continue;
                    }

                    for input_arg in input_positional_args.iter() {
                        if let Some(ArgumentValue::ArgArray(l)) = params.get(name) {
                            if let Ok(v) = converter.convert(input_arg) {
                                let mut lcopy = l.clone();
                                lcopy.push(v);
                                params.insert(name.to_string(), ArgumentValue::ArgArray(lcopy));
                            }
                        } else if let Ok(v) = converter.convert(input_arg) {
                            params.insert(name.to_string(), ArgumentValue::ArgArray(vec![v]));
                        }
                    }
                }
            }
        }

        if params.contains_key("help") {
            let usage = self.compile_help_messages();
            return Err(usage);
        }

        Ok(params)
    }

    pub fn compile_help_messages(&self) -> String {
        let iterator = |a: &Arg| {
            let mut msg = " ".to_string() + &a.names.join(", ");
            let default_value = match &a.options.default {
                Some(ArgumentValue::ArgString(None, s)) => s.to_string(),
                _ => "".to_string(),
            };

            if !a.options.help.is_empty() {
                msg += &("  ".to_string() + &a.options.help);
                msg = msg.replace("%(prog)", &self.prog);
                msg = msg.replace("%(default)", &default_value);
            }
            msg
        };

        let mut arg_conversions = vec![];

        for arg in &self.optional_args {
            arg_conversions.push(format!("[{}]", arg.names[0]));
        }

        let mut messages = vec![format!(
            "usage: {}, {}",
            self.prog,
            arg_conversions.join(" ")
        )];

        if !self.positional_args.is_empty() {
            messages.push("".to_string());
            messages.push("positional arguments:".to_string());
            for a in &self.positional_args {
                messages.push(iterator(&a.clone()));
            }
        }
        if !self.optional_args.is_empty() {
            messages.push("".to_string());
            messages.push("optional arguments:".to_string());
            for a in &self.optional_args {
                messages.push(iterator(&a.clone()));
            }
        }

        messages.join("\n")
    }

    /**
     * Separate short form argument which doesn't have space character between name and value.
     * For example, turn:
     *   "-x1" => ["-x", "1"]
     *   "-x 1" => ["-x", "1"]
     *   "-xxxxx" => ["-x", "xxxx"]
     * @param args - arguments passed
     */
    pub fn normalize_args(&self, args: &[String]) -> Vec<String> {
        if self.optional_args.is_empty() {
            return args.to_vec();
        }

        let mut norm: Vec<String> = vec![];

        for arg in args {
            let mut optional_arg_without_space_found = false;

            // Only short form args like '-x' are targets.
            if arg.starts_with("--") {
                norm.push(arg.to_string());
                continue;
            }

            for k in 0..self.optional_args.len() - 1 {
                let index = index_of_match(
                    |n: &String| {
                        n.starts_with(&"-".to_string())
                            && !n.starts_with(&"--".to_string())
                            && n != arg
                            && arg.starts_with(n)
                    },
                    &self.optional_args[k].names,
                );

                if index < 0 {
                    continue;
                }

                let name = &self.optional_args[k].names[index as usize];
                let index2 = arg.find(name).unwrap_or_default() + name.len();
                let value = &arg[index2..].to_string();

                norm.push(name.to_string());
                norm.push(value.to_string());
                optional_arg_without_space_found = true;
                break;
            }

            if !optional_arg_without_space_found {
                norm.push(arg.to_string());
            }
        }

        norm
    }

    pub fn get_optional_arg_name(&self, arg: &Arg) -> Result<String, String> {
        let names = &arg.names;

        let double_hyphen_arg_index = index_of_match(|n: &String| n.starts_with("--"), names);

        if double_hyphen_arg_index > -1 {
            let name = &names[double_hyphen_arg_index as usize];
            #[allow(clippy::single_char_pattern)]
            let first_non_dash = skip_leading(name, "-").replace("-", "_");
            return Ok(first_non_dash);
        }

        #[allow(clippy::single_char_pattern)]
        let single_hyphen_arg_index = index_of_match(
            |n: &String| n.starts_with("-") && !n.starts_with("--"),
            names,
        );
        if single_hyphen_arg_index > -1 {
            let name = &names[single_hyphen_arg_index as usize];
            #[allow(clippy::single_char_pattern)]
            let first_non_dash = skip_leading(name, "-").replace("-", "_");
            return Ok(first_non_dash);
        }
        Err("Invalid argument name".to_string())
    }

    fn get_converter(
        &self,
        typeofarg: Option<Rc<dyn ArgumentValueConv>>,
    ) -> Rc<dyn ArgumentValueConv> {
        match typeofarg {
            None => Rc::new(EmptyConversion {}),
            Some(ty) => ty,
        }
    }
}
