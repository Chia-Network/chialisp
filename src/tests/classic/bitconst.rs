use std::rc::Rc;

use clvmr::Allocator;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

use crate::tests::classic::run::{do_basic_brun, do_basic_run};

use crate::classic::clvm::sexp::atom;
use crate::classic::clvm_tools::binutils::{assemble_from_ir, disassemble};
use crate::classic::clvm_tools::ir::r#type::NEW_BIT_CONSTANTS;
use crate::classic::clvm_tools::ir::reader::read_ir;
use crate::classic::clvm_tools::ir::writer::write_ir;

#[test]
fn test_binary_numeric_constant_classic_0() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "(mod () (include *bitconst*) 0o0)".to_string(),
    ]);
    assert_eq!(program.trim(), "()");
}

#[test]
fn test_binary_numeric_constant_modern_0() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "(mod () (include *standard-cl-nc25*) 0o0)".to_string(),
    ]);
    assert_eq!(program.trim(), "(1)");
}

#[test]
fn test_binary_numeric_constant_classic_1() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "(mod () (include *bitconst*) 0o377)".to_string(),
    ]);
    assert_eq!(program.trim(), "(q . -1)");
}

#[test]
fn test_binary_numeric_constant_classic_01() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "(mod () (include *bitconst*) 0o0377)".to_string(),
    ]);
    assert_eq!(program.trim(), "(q . 255)");
}

#[test]
fn test_binary_numeric_constant_modern_01() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "(mod () (include *standard-cl-nc25*) 0o377)".to_string(),
    ]);
    assert_eq!(program.trim(), "(1 . 0xff)");
}

#[test]
fn test_binary_numeric_constant_modern_02() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "(mod () (include *standard-cl-nc25*) (defconst X (concat 0o177 0x00)) X)".to_string(),
    ]);
    assert_eq!(program.trim(), "(2 (1 . 2) (4 (1 . 32512) 1))");
}

#[test]
fn test_binary_numeric_constant_classic_02() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "(mod () (include *bitconst*) (defconst X (concat 0o177 0x00)) X)".to_string(),
    ]);
    assert_eq!(program.trim(), "(q . 32512)");
}

#[test]
fn test_binary_numeric_constant_modern_03() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "(mod () (include *standard-cl-nc25*) (defconst X (concat 0o0177 0x00)) X)".to_string(),
    ]);
    assert_eq!(program.trim(), "(2 (1 . 2) (4 (1 . 0x007f00) 1))");
}

#[test]
fn test_binary_numeric_constant_classic_03() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "(mod () (include *bitconst*) (defconst X (concat 0o0177 0x00)) X)".to_string(),
    ]);
    assert_eq!(program.trim(), "(q . 0x007f00)");
}

#[test]
fn test_binary_numeric_constant_modern_1() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "(mod () (include *standard-cl-nc25*) 0o0377)".to_string(),
    ]);
    assert_eq!(program.trim(), "(1 . 0x00ff)");
}

#[test]
fn test_binary_numeric_constant_classic_2() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "(mod () (include *bitconst*) 0b1100)".to_string(),
    ]);
    assert_eq!(program.trim(), "(q . 12)");
}

#[test]
fn test_binary_numeric_constant_modern_2() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "(mod () (include *standard-cl-nc25*) 0b1100)".to_string(),
    ]);
    assert_eq!(program.trim(), "(1 . 0x0c)");
}

#[test]
fn test_binary_numeric_constant_classic_3() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "(mod () (include *bitconst*) 0b000001100)".to_string(),
    ]);
    assert_eq!(program.trim(), "(q . 0x000c)");
}

#[test]
fn test_binary_numeric_constant_modern_3() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "(mod () (include *standard-cl-nc25*) 0b000001100)".to_string(),
    ]);
    assert_eq!(program.trim(), "(1 . 0x000c)");
}

#[test]
fn test_binary_numeric_constant_classic_4() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "(mod () (include *bitconst*) 0b00)".to_string(),
    ]);
    assert_eq!(program.trim(), "(q . 0x00)");
}

#[test]
fn test_binary_numeric_constant_modern_4() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "(mod () (include *standard-cl-nc25*) (defconst X (concat 0b00 0x00)) X)".to_string(),
    ]);
    assert_eq!(program.trim(), "(2 (1 . 2) (4 (1 . 0x0000) 1))");
}

#[test]
fn test_binary_numeric_constant_classic_5() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "(mod () (include *bitconst*) 0o00)".to_string(),
    ]);
    assert_eq!(program.trim(), "(q . 0x00)");
}

#[test]
fn test_binary_numeric_constant_modern_5() {
    let program = do_basic_run(&vec![
        "run".to_string(),
        "(mod () (include *standard-cl-nc25*) (defconst X (concat 0o00 0x00)) X)".to_string(),
    ]);
    assert_eq!(program.trim(), "(2 (1 . 2) (4 (1 . 0x0000) 1))");
}

#[test]
fn test_binary_numeric_0377_constant_writer() {
    let ir_repr = Rc::new(read_ir("0o0377", NEW_BIT_CONSTANTS).unwrap());
    let reproduced_from_ir = write_ir(ir_repr.clone(), NEW_BIT_CONSTANTS);
    assert_eq!("0o0377", reproduced_from_ir);
}

// Fuzzing for these constants:
// Any constant with just a single 0 digit contains no bytes (except hex)
// Any binary constant whose #bits is not a multiple of 8.
// Any octal constant whose number of bits is 3 more than the next lowest multiple of 8 is padded.
#[test]
fn test_fuzz_bit_constants() {
    let allowed_digits = b"0123456789abcdef";
    let compile_run_prog = |sigil: &str, val: &[u8]| {
        let prog_in = format!(
            "(mod () (include {}) {})",
            sigil,
            String::from_utf8_lossy(val)
        );
        let prog_out = do_basic_run(&vec!["run".to_string(), prog_in]);
        do_basic_brun(&vec!["brun".to_string(), prog_out])
    };
    let choose_random = |rng: &mut ChaCha8Rng, choices: &[u8]| {
        let rv: u32 = rng.random();
        choices[(rv as usize) % choices.len()]
    };
    let find_digit_value = |digit: u8| {
        allowed_digits
            .iter()
            .enumerate()
            .find(|(_i, d)| **d == digit)
            .map(|(i, _d)| i)
            .unwrap()
    };

    let eval_const = |bits: usize, digits_vec: &[u8], result: &str| {
        let mut allocator = Allocator::new();
        let original_string = String::from_utf8_lossy(digits_vec);
        eprintln!("testing {result} <- {original_string}");
        let ir_repr = Rc::new(read_ir(&original_string, NEW_BIT_CONSTANTS).unwrap());
        // Check ir writer.
        let reproduced_from_ir = write_ir(ir_repr.clone(), NEW_BIT_CONSTANTS);
        assert_eq!(original_string, reproduced_from_ir);

        // Check assembly layer.
        let assembled = assemble_from_ir(&mut allocator, ir_repr.clone()).unwrap();
        let disassembled = disassemble(&mut allocator, assembled, None);
        assert_eq!(result.trim(), disassembled);

        let atom_data = atom(&allocator, assembled).unwrap();
        // The length should be what we expect.
        let digits = &digits_vec[2..];
        let rounded_up_bytes = ((digits.len() * bits) + 7) / 8;
        let rounded_down_bytes = (digits.len() * bits) / 8;
        let bits_from_left = find_digit_value(digits[0]);
        let high_byte_overhang = (digits.len() * bits) - (rounded_down_bytes * 8);
        let high_byte_bits = if high_byte_overhang >= bits {
            1
        } else {
            bits_from_left >> bits - high_byte_overhang
        };

        let expected_length = if (digits[0] == b'0' && digits.len() == 1 && bits != 4)
            || (high_byte_overhang < bits && high_byte_bits == 0)
        {
            rounded_down_bytes
        } else {
            rounded_up_bytes
        };

        assert_eq!(expected_length, atom_data.len());

        if atom_data.is_empty() {
            return;
        }

        // The content should be what we expect.
        for (i, digit) in digits.iter().rev().enumerate() {
            let raw_bit_offset = i * bits;
            let bit_offset = raw_bit_offset % 8;
            let byte_offset = raw_bit_offset / 8;
            let digit_value = find_digit_value(*digit);
            let atom_data_idx = atom_data.len() - byte_offset - 1;
            let mut mask: u16 = ((1 << bits) - 1) << bit_offset;
            let mut shifted_digit_value: u16 = (digit_value << bit_offset) as u16;
            assert_eq!(
                (shifted_digit_value & 0xff),
                mask & atom_data[atom_data_idx] as u16
            );
            if mask >= 256 && byte_offset + 1 < atom_data.len() {
                mask >>= 8;
                shifted_digit_value >>= 8;
                assert_eq!(
                    (mask & shifted_digit_value),
                    mask & atom_data[atom_data_idx - 1] as u16
                );
            }
        }
    };

    let merge_left_pad_zeros = |bits: usize, digit_vec: &mut Vec<u8>| {
        // Hex constants always decode to a multiple of 2 digits.
        if bits == 4 {
            if !digit_vec.len().is_multiple_of(2) {
                digit_vec.insert(2, b'0');
            }
            return;
        }

        loop {
            if digit_vec.len() <= 3 {
                return;
            }

            let provided_bits = (digit_vec.len() - 2) * bits;
            let start_byte_of_top_digit = (provided_bits - bits) / 8;
            let bits_into_left_byte = (provided_bits - bits) - (start_byte_of_top_digit * 8);
            // Last 2 zeros of last byte.
            if digit_vec.len() == 4 && digit_vec.iter().skip(2).take(2).all(|b| *b == b'0') {
                return;
            }

            let bits_in_byte_for_2_digits = ((find_digit_value(digit_vec[2])
                << 8 + bits_into_left_byte)
                | (find_digit_value(digit_vec[3]) << 8 + bits_into_left_byte - bits))
                >> 8;
            if (bits_in_byte_for_2_digits != 0 && digit_vec[2] == b'0')
                || (bits_in_byte_for_2_digits == 0 && bits_into_left_byte >= bits)
            {
                digit_vec.remove(2);
                continue;
            }

            return;
        }
    };

    let do_test = |sigil: &str| {
        let mut digit_vec = Vec::new();
        let rng_seed: [u8; 32] = [0; 32];
        let mut randomizer = ChaCha8Rng::from_seed(rng_seed);

        for d in [(1, b'b'), (3, b'o'), (4, b'x')].iter() {
            digit_vec.clear();
            digit_vec.push(b'0');
            digit_vec.push(d.1);
            digit_vec.push(b'0');

            if d.0 == 4 {
                assert_eq!(compile_run_prog(sigil, &digit_vec).trim(), "0x00");
            } else {
                assert_eq!(compile_run_prog(sigil, &digit_vec).trim(), "()");
            }

            for _digits in 1..20 {
                for use_digit in allowed_digits[0..(1 << d.0)].iter() {
                    digit_vec.pop();
                    digit_vec.push(*use_digit);
                    merge_left_pad_zeros(d.0, &mut digit_vec);
                    eval_const(d.0, &digit_vec, &compile_run_prog(sigil, &digit_vec));
                }

                // All zeroes
                for i in 2..digit_vec.len() {
                    digit_vec[i] = b'0';
                }

                // If 2 consecutive octal or binary digits are in the same byte, then they'll get
                // merged.
                merge_left_pad_zeros(d.0, &mut digit_vec);
                eval_const(d.0, &digit_vec, &compile_run_prog(sigil, &digit_vec));

                // Random digits
                for i in 2..digit_vec.len() {
                    digit_vec[i] = choose_random(&mut randomizer, &allowed_digits[0..(1 << d.0)]);
                }

                merge_left_pad_zeros(d.0, &mut digit_vec);
                eval_const(d.0, &digit_vec, &compile_run_prog(sigil, &digit_vec));

                digit_vec.push(b'0');
            }
        }
    };

    do_test("*bitconst*");
    do_test("*standard-cl-nc25*");
}
