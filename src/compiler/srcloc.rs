use std::borrow::Borrow;
use std::fmt::Display;
use std::rc::Rc;

use serde::Serialize;

/// If a Srcloc identifies a range of characters in the source file, this
/// identifies the tail of the range.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Until {
    pub line: usize,
    pub col: usize,
    pub offset: usize,
}

impl Until {
    pub fn new(line: usize, col: usize, offset: usize) -> Self {
        Until {
            line,
            col,
            offset,
        }
    }
}

/// Specifies the coordinates of an object in a source file, including the file
/// name.  The name is held by reference count so they can be held and cloned
/// relatively freely.
///
/// They are intended to be relatively small.  Every SExp is decorated with one
/// Srcloc to identify where it came from in the source.  These allow tools to
/// report errors precisely downstream in the compiler (for example, reporting
/// the specific atom on which an attempt to do first or rest was made, deep in
/// the compiler infra.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Srcloc {
    pub file: Rc<String>,
    pub line: usize,
    pub col: usize,
    pub offset: usize,
    pub until: Option<Until>,
}

impl Display for Srcloc {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match &self.until {
            None => formatter.write_str(&format!("{}({}):{}", self.file, self.line, self.col)),
            Some(u) => formatter.write_str(&format!(
                "{}({}):{}-{}({}):{}",
                self.file, self.line, self.col, self.file, u.line, u.col
            )),
        }
    }
}

impl Srcloc {
    /// Create a srcloc given a refcounted pointer to the filename (so they're
    /// always shareable) and the line number and column (1-based).
    pub fn new(name: Rc<String>, line: usize, col: usize, offset: usize) -> Self {
        Srcloc {
            file: name,
            line,
            col,
            offset,
            until: None,
        }
    }

    /// Get the ending of the range as a Srcloc.  If there's no tail then
    /// it's also the beginning.
    pub fn ending(&self) -> Srcloc {
        if let Some(u) = &self.until {
            return Srcloc {
                file: self.file.clone(),
                line: u.line,
                col: u.col,
                offset: u.offset,
                until: None,
            };
        }
        self.clone()
    }

    /// Tell whether the srcloc overlaps another srcloc.  This is used by the
    /// language server to determine the target of an autocompletion or what
    /// form the cursor is in, among other things.
    pub fn overlap(&self, other: &Srcloc) -> bool {
        let mf: &String = self.file.borrow();
        let of: &String = other.file.borrow();
        if mf != of {
            return false;
        }

        if self.line == other.line && self.col == other.col {
            return true;
        }

        match (self.until.as_ref(), other.until.as_ref()) {
            (None, None) => self.line == other.line && self.col == other.col,
            (None, Some(_)) => other.overlap(self),
            (Some(self_until), None) => {
                if self.line < other.line && self_until.line > other.line {
                    return true;
                }
                // The case where we have len means we have only one line in the self srcloc.
                // In that case, we want to encompass other with our range (since it's singular).
                if let Some(len) = self.len() {
                    if self.line == other.line
                        && self.col <= other.col
                        && self.col + len >= other.col
                    {
                        return true;
                    }
                } else {
                    // In this case, we have match if other is on the same line as self and after
                    // self.col or on the same line as self_until and before col.
                    if self.line == other.line && self.col <= other.col
                        || self_until.line == other.line && self_until.col >= other.col
                    {
                        return true;
                    }
                }

                false
            }
            (Some(my_until), Some(their_until)) => {
                let self_start = Srcloc::new(self.file.clone(), self.line, self.col, self.offset);
                let self_until = Srcloc::new(self.file.clone(), my_until.line, my_until.col, my_until.offset);
                let other_start = Srcloc::new(self.file.clone(), other.line, other.col, my_until.offset);
                let other_until = Srcloc::new(self.file.clone(), their_until.line, their_until.col, my_until.offset);
                other.overlap(&self_start)
                    || other.overlap(&self_until)
                    || self.overlap(&other_start)
                    || self.overlap(&other_until)
            }
        }
    }

    /// Returns whether the srcloc identifies an empty range (re: clippy).
    /// Currently impossible.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Length of the string representation for the srcloc's range if it's on
    /// the same line.  Some thought is needed to know what we want for a range
    /// over lines.
    pub fn len(&self) -> Option<usize> {
        if let Some(self_until) = &self.until {
            if self_until.line != self.line {
                None
            } else {
                Some(self_until.col - self.col)
            }
        } else {
            Some(1)
        }
    }

    /// Create a srcloc that begins where this srcloc begins and ends at the given
    /// at the farther of the two range endings.
    pub fn ext(&self, other: &Srcloc) -> Srcloc {
        if other.file == self.file {
            combine_src_location(self, other)
        } else {
            self.clone()
        }
    }

    /// Given a u8 byte from a stream, figure out the next source location.
    /// Respects newline and tab.
    pub fn advance(&self, ch: u8) -> Srcloc {
        match ch as char {
            '\n' => Srcloc {
                file: self.file.clone(),
                col: 1,
                line: self.line + 1,
                until: self.until.clone(),
                offset: self.offset + 1,
            },
            '\t' => {
                let next_tab = (self.col + 8) & !7;
                Srcloc {
                    file: self.file.clone(),
                    col: next_tab,
                    line: self.line,
                    until: self.until.clone(),
                    offset: self.offset + 1,
                }
            }
            _ => Srcloc {
                file: self.file.clone(),
                col: self.col + 1,
                line: self.line,
                until: self.until.clone(),
                offset: self.offset + 1,
            },
        }
    }

    /// Given a filename, create a Srcloc that's logically at the beginning of
    /// that file.
    pub fn start(file: &str) -> Srcloc {
        Srcloc {
            file: Rc::new(file.to_string()),
            line: 1,
            col: 1,
            until: None,
            offset: 0,
        }
    }

    pub fn compiler_internal_srcloc() -> Srcloc {
        Srcloc::start("*sym*")
    }
}

/// Return the first character addressed by the srcloc.
pub fn src_location_min(a: &Srcloc) -> (usize, usize) {
    (a.line, a.col)
}

/// Return the last character addressed by the srcloc.
pub fn src_location_max(a: &Srcloc) -> (usize, usize) {
    match &a.until {
        None => (a.line, a.col + 1),
        Some(u) => (u.line, u.col),
    }
}

fn add_onto(x: &Srcloc, y: &Srcloc) -> Srcloc {
    let (sm_row, sm_col) = src_location_max(y);
    Srcloc {
        until: Some(Until::new(sm_row, sm_col, y.offset)),
        .. x.clone()
    }
}

/// Helper function for Srcloc::ext.
fn combine_src_location(a: &Srcloc, b: &Srcloc) -> Srcloc {
    match (a.line < b.line, a.line == b.line) {
        (true, _) => add_onto(a, b),
        (_, true) => match (a.col < b.col, a.col == b.col) {
            (true, _) => add_onto(a, b),
            (_, true) => a.clone(),
            _ => add_onto(b, a),
        },
        _ => add_onto(b, a),
    }
}
