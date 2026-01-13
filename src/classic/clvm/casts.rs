use clvm_rs::NodePtr;
use num_bigint::ToBigInt;

use clvm_rs::error::EvalErr;

use crate::classic::clvm::__type_compatibility__::{
    bi_one, bi_zero, get_u32, Bytes, BytesFromType,
};
use crate::util::Number;

pub struct TConvertOption {
    pub signed: bool,
}

pub fn int_from_bytes(b: Bytes, option: Option<TConvertOption>) -> Result<u64, EvalErr> {
    if b.length() == 0 {
        return Ok(0);
    } else if b.length() * 8 > 64 {
        return Err(EvalErr::InternalError(
            NodePtr::NIL,
            "Cannot convert Bytes to Integer larger than 64bit. Use bigint_from_bytes instead."
                .to_string(),
        ));
    }

    let signed = option.map(|cvt| cvt.signed).unwrap_or_else(|| false);
    let mut unsigned64: u64 = 0;
    let dv = b.raw();

    let bytes4_remain = dv.len() % 4;
    let bytes4_length = (dv.len() - bytes4_remain) / 4;

    let mut order: u64 = 1;

    if bytes4_length > 0 {
        for i_reverse in 0..bytes4_length {
            let i = bytes4_length - i_reverse - 1;
            let byte32 = get_u32(&dv, i * 4 + bytes4_remain) as u64;
            unsigned64 += byte32 * order;
            order <<= 32;
        }
    }

    if bytes4_remain > 0 {
        if bytes4_length == 0 {
            order = 1;
        }
        for i_reverse in 0..bytes4_remain {
            let i = bytes4_remain - i_reverse - 1;
            let byte = dv[i] as u64;
            unsigned64 += byte * order;
            order <<= 8;
        }
    }

    // If the first bit is 1, it is recognized as a negative number.
    if signed && ((dv[0] & 0x80) != 0) {
        return Ok(unsigned64 - (1 << (b.length() * 8)));
    }
    Ok(unsigned64)
}

pub fn bigint_from_bytes(b: &Bytes, option: Option<TConvertOption>) -> Number {
    if b.length() == 0 {
        return bi_zero();
    }

    let signed = option.map(|cvt| cvt.signed).unwrap_or_else(|| false);
    let mut unsigned = bi_zero();
    let dv = b.raw();

    let bytes4_remain = dv.len() % 4;
    let bytes4_length = (dv.len() - bytes4_remain) / 4;

    let mut order = bi_one();

    if bytes4_length > 0 {
        for i_reverse in 0..bytes4_length {
            let i = bytes4_length - i_reverse - 1;
            let byte32 = get_u32(&dv, i * 4 + bytes4_remain);
            unsigned += byte32.to_bigint().unwrap() * order.clone();
            order <<= 32;
        }
    }

    if bytes4_remain > 0 {
        if bytes4_length == 0 {
            order = bi_one();
        }
        for i_reverse in 0..bytes4_remain {
            let i = bytes4_remain - i_reverse - 1;
            let byte = dv[i];
            unsigned += byte.to_bigint().unwrap() * order.clone();
            order <<= 8;
        }
    }

    // If the first bit is 1, it is recognized as a negative number.
    if signed && ((dv[0] & 0x80) != 0) {
        return unsigned - (bi_one() << (b.length() * 8));
    }
    unsigned
}

// Convert to bytes with no leading zeros added.  This allows numbers to appear
// negative in the result stream but are treated as positive (for node paths).
pub fn bigint_to_bytes_unsigned(v: &Number) -> Bytes {
    if *v == bi_zero() {
        return Bytes::new(None);
    }

    // This is only for positive numbers.
    assert!(*v > bi_zero());

    let (_, result) = v.to_bytes_be();
    Bytes::new(Some(BytesFromType::Raw(result)))
}

// Pulled in code from clvm_rs to replace this function, re: PR comments.
// The whole function relies on allocator so this is just the brass tacks.
// Conversion type was here for completeness in the original code I ported
// from the typescript.
//
// The unsigned option is easier, as used in as_path.
pub fn bigint_to_bytes_clvm(v: &Number) -> Bytes {
    let bytes: Vec<u8> = v.to_signed_bytes_be();
    let mut slice = bytes.as_slice();

    // make number minimal by removing leading zeros
    while (!slice.is_empty()) && (slice[0] == 0) {
        if slice.len() > 1 && (slice[1] & 0x80 == 0x80) {
            break;
        }
        slice = &slice[1..];
    }

    Bytes::new(Some(BytesFromType::Raw(slice.to_vec())))
}

// /**
//  * Return the number of bytes required to represent this integer.
//  * @param {number} v
//  */
// export function limbs_for_int(v: number|bigint): number {
//   return ((v >= 0 ? v : -v).toString(2).length + 7) >> 3;
// }
