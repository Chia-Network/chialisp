use num_bigint::BigInt;
use std::collections::HashSet;
use std::fs;
use std::io::Write;
use std::mem::swap;
use std::path::Path;
use tempfile::NamedTempFile;
use unicode_segmentation::UnicodeSegmentation;

pub type Number = BigInt;

// Thanks: https://www.reddit.com/r/rust/comments/bkkpkz/pkgversion_access_your_crates_version_number_as/
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

pub fn number_from_u8(v: &[u8]) -> Number {
    let len = v.len();
    if len == 0 {
        0.into()
    } else {
        Number::from_signed_bytes_be(v)
    }
}

pub fn u8_from_number(v: Number) -> Vec<u8> {
    v.to_signed_bytes_be()
}

pub fn index_of_match<F, T>(cb: F, haystack: &[T]) -> i32
where
    F: Fn(&T) -> bool,
{
    for (i, ch) in haystack.iter().enumerate() {
        if cb(ch) {
            return i as i32;
        }
    }
    -1
}

pub fn skip_leading(s: &str, dash: &str) -> String {
    s.graphemes(true).skip_while(|ch| dash == *ch).collect()
}

pub fn collapse<A>(r: Result<A, A>) -> A {
    match r {
        Ok(a) => a,
        Err(e) => e,
    }
}

#[derive(Debug, Clone)]
pub struct TopoSortItem<K> {
    pub index: usize,
    pub needs: HashSet<K>,
    pub has: HashSet<K>,
}

// F: tells whether t2 includes t1.
pub fn toposort<K, T, E, Needs, Has>(
    list: &[T],
    deadlock: E,
    needs: Needs,
    has: Has,
) -> Result<Vec<TopoSortItem<K>>, E>
where
    Needs: Fn(&HashSet<K>, &T) -> Result<HashSet<K>, E>,
    Has: Fn(&T) -> HashSet<K>,
    K: std::cmp::Eq,
    K: std::hash::Hash,
    K: Clone,
{
    let mut possible = HashSet::new();
    let mut done = HashSet::new();
    let mut items: Vec<TopoSortItem<K>> = list
        .iter()
        .enumerate()
        .map(|(i, item)| TopoSortItem {
            index: i,
            needs: HashSet::new(),
            has: has(item),
        })
        .collect();
    let mut finished_idx = 0;

    // Determine what's defined in these bindings.
    for item in items.iter() {
        for new_item in item.has.iter() {
            possible.insert(new_item.clone());
        }
    }

    // Set needs based on possible.  We may fail.
    for i in 0..items.len() {
        items[i].needs = needs(&possible, &list[i])?;
    }

    while finished_idx < items.len() {
        // Find things with no new dependencies.
        let move_to_front: Vec<(usize, TopoSortItem<K>)> = items
            .iter()
            .enumerate()
            .skip(finished_idx)
            .filter(|(_, item)| item.needs.is_subset(&done))
            .map(|(i, item)| (i, item.clone()))
            .collect();

        if move_to_front.is_empty() {
            // Circular dependency somewhere.
            return Err(deadlock);
        }

        // Swap items into place earlier in the list.
        for (idx, _tomove) in move_to_front.iter() {
            if *idx != finished_idx {
                let mut tmp = items[*idx].clone();
                swap(&mut tmp, &mut items[finished_idx]);
                items[*idx] = tmp;
            }

            // Add new 'has' items to done.
            let mut tmp = HashSet::new();
            for u in done.union(&items[finished_idx].has) {
                tmp.insert(u.clone());
            }
            let intersection = tmp.intersection(&possible);
            done.clear();
            for i in intersection {
                done.insert(i.clone());
            }

            finished_idx += 1;
        }
    }

    Ok(items)
}

pub trait ErrInto<D> {
    fn err_into(self) -> D;
}

impl<SrcErr, DestErr, DestRes> ErrInto<Result<DestRes, DestErr>> for Result<DestRes, SrcErr>
where
    DestErr: From<SrcErr>,
{
    fn err_into(self) -> Result<DestRes, DestErr> {
        self.map_err(|e| e.into())
    }
}

pub fn atomic_write_file(
    input_path: &str,
    output_path: &str,
    target_data: &str,
) -> Result<(), String> {
    let output_path_obj = Path::new(output_path);
    let output_dir = output_path_obj
        .parent()
        .map(Ok)
        .unwrap_or_else(|| Err("could not get parent of output path"))?;

    // Make the contents appear atomically so that other test processes
    // won't mistake an empty file for intended output.
    let mut temp_output_file = NamedTempFile::new_in(output_dir)
        .map_err(|e| format!("error creating temporary compiler output for {input_path}: {e:?}"))?;

    let err_text = format!("failed to write to {:?}", temp_output_file.path());
    let translate_err = |_| err_text.clone();

    temp_output_file
        .write_all(target_data.as_bytes())
        .map_err(translate_err)?;

    temp_output_file
        .persist(output_path)
        .map_err(|e| format!("error persisting temporary compiler output {output_path}: {e:?}"))?;

    Ok(())
}

pub fn gentle_overwrite(
    input_path: &str,
    output_path: &str,
    target_data: &str,
) -> Result<(), String> {
    if let Ok(prev_content) = fs::read_to_string(output_path) {
        let prev_trimmed = prev_content.trim();
        let trimmed = target_data.trim();
        if prev_trimmed == trimmed {
            // We should try to overwrite here, but not fail if it doesn't
            // work.  This will accomodate both the read only scenario and
            // the scenario where a target file is newer and people want the
            // date to be updated.
            atomic_write_file(input_path, output_path, target_data).ok();

            // It's the same program, bail regardless.
            return Ok(());
        }
    }

    atomic_write_file(input_path, output_path, target_data)
}
