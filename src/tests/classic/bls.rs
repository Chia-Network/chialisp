use crate::classic::clvm::__type_compatibility__::{Bytes, BytesFromType};
use crate::tests::classic::run::{do_basic_brun, do_basic_run};

const MSG1: &[u8] = &[
    0x97, 0x90, 0x63, 0x5d, 0xe8, 0x74, 0x0e, 0x9a, 0x6a, 0x6b, 0x15, 0xfb, 0x6b, 0x72, 0xf3, 0xa1,
    0x6a, 0xfa, 0x09, 0x73, 0xd9, 0x71, 0x97, 0x9b, 0x6b, 0xa5, 0x47, 0x61, 0xd6, 0xe2, 0x50, 0x2c,
    0x50, 0xdb, 0x76, 0xf4, 0xd2, 0x61, 0x43, 0xf0, 0x54, 0x59, 0xa4, 0x2c, 0xfd, 0x52, 0x0d, 0x44,
];

fn bls_map_to_g1(msg: &[u8]) -> Vec<u8> {
    let point = chia_bls::hash_to_g1(msg);
    let expected_output: [u8; 48] = point.to_bytes();
    expected_output.to_vec()
}

#[test]
fn test_using_bls_operators_0() {
    let expected_output = bls_map_to_g1(&MSG1);
    // Run a program which uses the map_to_g1 operator and check the output.
    let result = do_basic_run(&vec![
        "run".to_string(),
        "resources/tests/bls/classic-bls-op-test-1.clsp".to_string(),
    ])
    .trim()
    .to_string();
    let hex = Bytes::new(Some(BytesFromType::Raw(expected_output.to_vec()))).hex();
    assert_eq!(result, format!("(q . 0x{hex})"));
}

#[test]
fn test_using_bls_verify_signature_good_msg_classic() {
    let right_msg = "(0x0102030405)";

    let prog = do_basic_run(&vec![
        "run".to_string(),
        "resources/tests/bls/classic-bls-verify-signature.clsp".to_string(),
    ])
    .trim()
    .to_string();
    let result = do_basic_brun(&vec!["brun".to_string(), prog, right_msg.to_string()])
        .trim()
        .to_string();
    assert_eq!(result, format!("()"));
}

#[test]
fn test_using_bls_verify_signature_bad_msg_classic() {
    let wrong_msg = "(0x0102030415)";

    let prog = do_basic_run(&vec![
        "run".to_string(),
        "resources/tests/bls/classic-bls-verify-signature.clsp".to_string(),
    ])
    .trim()
    .to_string();
    let result = do_basic_brun(&vec!["brun".to_string(), prog, wrong_msg.to_string()])
        .trim()
        .to_string();
    assert!(result.starts_with("FAIL"));
}
#[test]
fn test_using_bls_verify_signature_good_msg() {
    let right_msg = "(0x0102030405)";

    let prog = do_basic_run(&vec![
        "run".to_string(),
        "resources/tests/bls/modern-bls-verify-signature.clsp".to_string(),
    ])
    .trim()
    .to_string();
    let result = do_basic_brun(&vec!["brun".to_string(), prog, right_msg.to_string()])
        .trim()
        .to_string();
    assert_eq!(result, format!("()"));
}

#[test]
fn test_using_bls_verify_signature_bad_msg() {
    let wrong_msg = "(0x0102030415)";

    let prog = do_basic_run(&vec![
        "run".to_string(),
        "resources/tests/bls/modern-bls-verify-signature.clsp".to_string(),
    ])
    .trim()
    .to_string();
    let result = do_basic_brun(&vec!["brun".to_string(), prog, wrong_msg.to_string()])
        .trim()
        .to_string();
    assert!(result.starts_with("FAIL"));
}

#[test]
fn test_coinid_in_softfork_bad() {
    let prog = do_basic_run(&vec![
        "run".to_string(),
        "resources/tests/bls/coinid-fail.clsp".to_string(),
    ])
    .trim()
    .to_string();
    let result = do_basic_brun(&vec!["brun".to_string(), prog, "()".to_string()])
        .trim()
        .to_string();
    assert!(result.starts_with("FAIL"));
}

#[test]
fn test_coinid_in_softfork_good() {
    let prog = do_basic_run(&vec![
        "run".to_string(),
        "resources/tests/bls/coinid-good.clsp".to_string(),
    ])
    .trim()
    .to_string();
    let result = do_basic_brun(&vec!["brun".to_string(), prog, "()".to_string()])
        .trim()
        .to_string();
    assert!(result.starts_with("()"));
}
