import type { IProgram, ITuple } from '../../../../../pkg/clvm_tools_wasm.js';
import type { G1Element } from 'bls-signatures';

import * as fs from 'fs';
import { resolve } from 'path';
import * as assert from 'assert';
import * as bls_loader from 'bls-signatures';
const {h, t, Program} = require('../../../../../pkg/clvm_tools_wasm.js');

// @ts-ignore
import {SHA256} from 'sha2';
// @ts-ignore
import MersenneTwister from 'mersenne-twister';

interface ConvertibleValue {
    toPlainValue: () => any;
    toHexValue: () => string;
    shatree: () => Uint8Array;
}

//
// Value kinds:
//
// Some interesting constant values to be followed by generators.
//

function rngu32(rng: MersenneTwister): number {
    return (rng.random_int() + 0xffffffff + 1) % (0xffffffff + 1);
}

// @ts-ignore
function bi_add(a: BigInt, b: BigInt): BigInt {
    return <any>a + <any>b;
}

function rev_iter<T>(array: Array<T>, f: (v: T, i: number) => void) {
    for (let i_rev = 0; i_rev < array.length; i_rev++) {
        let i = array.length - i_rev - 1;
        f(array[i], i);
    }
}

class Sha256 {
    bytes: Array<number>;

    constructor() {
        this.bytes = [];
    }
    update(array: Uint8Array): void {
        for (let i = 0; i < array.length; i++) {
            this.bytes.push(array[i]);
        }
    }
    digest(): Uint8Array {
        return new Uint8Array(SHA256(Buffer.from(this.bytes)));
    }
}

class FixedValue implements ConvertibleValue {
    value: any;
    hex: string;
    hash: Uint8Array;

    constructor(value: any, hex: string, hash: Uint8Array) {
        this.value = value;
        this.hex = hex;
        this.hash = hash;
    }
    toPlainValue(): any {
        return this.value;
    }
    toHexValue(): string {
        return this.hex;
    }
    shatree(): Uint8Array {
        return this.hash;
    }
}

const nilHash = new Uint8Array([
    0x4b, 0xf5, 0x12, 0x2f, 0x34, 0x45, 0x54, 0xc5, 0x3b, 0xde, 0x2e, 0xbb, 0x8c, 0xd2, 0xb7, 0xe3, 0xd1, 0x60, 0x0a, 0xd6, 0x31, 0xc3, 0x85, 0xa5, 0xd7, 0xcc, 0xe2, 0x3c, 0x77, 0x85, 0x45, 0x9a
]);
const nilValue = new FixedValue(
    [],
    "80",
    nilHash);
const generateBasicValues = [
    (rng: MersenneTwister, lv: number) => nilValue,
    (rng: MersenneTwister, lv: number) => new FixedValue(
        0,
        "80",
        nilHash),
    (rng: MersenneTwister, lv: number) => new FixedValue(
        0n,
        "80",
        nilHash),
   (rng: MersenneTwister, lv: number) => new FixedValue(
           -1,
       "81ff",
       new Uint8Array([
           0x4b, 0x3a, 0x43, 0xf5, 0x92, 0xf5, 0x77, 0xfc, 0xfc, 0xb5, 0xb0, 0xe1, 0xf4, 0x2b, 0xec, 0x51, 0x82, 0xc9, 0xed, 0xc4, 0x14, 0xe1, 0xf6, 0x67, 0x52, 0x8f, 0x56, 0xe7, 0xcf, 0x0b, 0xe1, 0x1d
       ])),
    (rng: MersenneTwister, lv: number) => new FixedValue(
        255,
        "8200ff",
        new Uint8Array([
            0xb7, 0xae, 0x77, 0x29, 0x55, 0x5e, 0xc6, 0x82, 0x9c, 0x57, 0x9c, 0x26, 0x02, 0xed, 0xc6, 0xcb, 0x94, 0xb4, 0xed, 0x3d, 0x82, 0x0d, 0xdd, 0xa0, 0xa4, 0x5a, 0xc5, 0x40, 0x30, 0xf8, 0xa5, 0x3d
        ])),
    (rng: MersenneTwister, lv: number) => new FixedValue(
        254,
        "8200fe",
        new Uint8Array([
            0xd8, 0xec, 0xcb, 0x85, 0xf0, 0x66, 0x3d, 0x1f, 0x44, 0xb7, 0x64, 0xea, 0x6b, 0xc0, 0x65, 0x31, 0x68, 0x02, 0x0c, 0x5d, 0x7f, 0x65, 0x8d, 0x94, 0x93, 0x15, 0xfd, 0x69, 0xcb, 0x78, 0x3b, 0x83
        ])),
    (rng: MersenneTwister, lv: number) => new FixedValue(
        128,
        "820080",
        new Uint8Array([
            0x0c, 0xa4, 0xbf, 0x05, 0x38, 0xdc, 0x6e, 0x2d, 0x92, 0xd0, 0x1e, 0xc7, 0x82, 0xe4, 0xea, 0xca, 0x60, 0x5d, 0x02, 0x06, 0xec, 0xdc, 0x1e, 0xe8, 0x84, 0x5a, 0xab, 0x03, 0xa2, 0x7a, 0xa6, 0x23
        ])),
    (rng: MersenneTwister, lv: number) => new FixedValue(
        127,
        "7f",
        new Uint8Array([
            0x3f, 0x5c, 0xf1, 0x31, 0xde, 0x77, 0x40, 0x7c, 0xc5, 0x0a, 0x23, 0x0f, 0x76, 0xae, 0x5e, 0x02, 0xff, 0x62, 0xd0, 0x1e, 0xf4, 0x0e, 0x87, 0x39, 0xc5, 0x11, 0x82, 0x95, 0xa2, 0xd8, 0x14, 0x32
        ])),
    (rng: MersenneTwister, lv: number) => new FixedValue(
        0x7fff,
        "827fff",
        new Uint8Array([
            0x05, 0xa3, 0xe7, 0x78, 0x15, 0x92, 0xd1, 0x03, 0x2b, 0x71, 0x92, 0x2b, 0x03, 0x69, 0xa2, 0xf3, 0x3e, 0x4e, 0x19, 0xb7, 0x37, 0x1f, 0xfd, 0xbb, 0xaf, 0x51, 0x15, 0xfd, 0x8f, 0xf5, 0xc7, 0x5d
        ])),
    (rng: MersenneTwister, lv: number) => new FixedValue(
        0xffff,
        "8300ffff",
        new Uint8Array([
            0x16, 0xb8, 0xcb, 0x1f, 0xe7, 0x34, 0xfb, 0xc6, 0x0c, 0x67, 0x63, 0xc9, 0x4c, 0x9e, 0x4c, 0xc5, 0x58, 0x40, 0xae, 0x96, 0x6e, 0x7e, 0x50, 0x8b, 0xa8, 0x2f, 0x53, 0x9d, 0x82, 0x70, 0x25, 0x11
        ])),
    (rng: MersenneTwister, lv: number) => new FixedValue(
        new Uint8Array([0]),
        "00",
        new Uint8Array([
            0x47, 0xdc, 0x54, 0x0c, 0x94, 0xce, 0xb7, 0x04, 0xa2, 0x38, 0x75, 0xc1, 0x12, 0x73, 0xe1, 0x6b, 0xb0, 0xb8, 0xa8, 0x7a, 0xed, 0x84, 0xde, 0x91, 0x1f, 0x21, 0x33, 0x56, 0x81, 0x15, 0xf2, 0x54
        ])),
    (rng: MersenneTwister, lv: number) => new FixedValue(
        new Uint8Array([0,255]),
        "8200ff",
        new Uint8Array([
            0xb7, 0xae, 0x77, 0x29, 0x55, 0x5e, 0xc6, 0x82, 0x9c, 0x57, 0x9c, 0x26, 0x02, 0xed, 0xc6, 0xcb, 0x94, 0xb4, 0xed, 0x3d, 0x82, 0x0d, 0xdd, 0xa0, 0xa4, 0x5a, 0xc5, 0x40, 0x30, 0xf8, 0xa5, 0x3d
        ])),
    (rng: MersenneTwister, lv: number) => new FixedValue(
        new Uint8Array([255,0]),
        "82ff00",
        new Uint8Array([
            0xf4, 0xa0, 0x24, 0x42, 0xda, 0x2c, 0x7e, 0x7e, 0x0a, 0xbe, 0xa3, 0x36, 0xad, 0x09, 0x8e, 0x1b, 0x86, 0x0c, 0x22, 0x5a, 0x17, 0x0a, 0x67, 0x1e, 0x32, 0x47, 0xfd, 0xba, 0xaa, 0x69, 0xb9, 0xa5
        ])),
    (rng: MersenneTwister, lv: number) => new FixedValue(
        new Uint8Array([0,127]),
        "82007f",
        new Uint8Array([
            0x88, 0xfb, 0xa5, 0xe6, 0xf9, 0xc8, 0xce, 0x4c, 0xbc, 0x7d, 0x4b, 0xcf, 0x2c, 0x66, 0x9b, 0xd9, 0xcd, 0x75, 0x8f, 0x5c, 0xfd, 0x48, 0xfb, 0x34, 0x35, 0x23, 0x1c, 0x16, 0xfd, 0x2b, 0x91, 0xaa
        ])),
    (rng: MersenneTwister, lv: number) => new FixedValue(
        new Uint8Array([0,128]),
        "820080",
        new Uint8Array([
            0x0c, 0xa4, 0xbf, 0x05, 0x38, 0xdc, 0x6e, 0x2d, 0x92, 0xd0, 0x1e, 0xc7, 0x82, 0xe4, 0xea, 0xca, 0x60, 0x5d, 0x02, 0x06, 0xec, 0xdc, 0x1e, 0xe8, 0x84, 0x5a, 0xab, 0x03, 0xa2, 0x7a, 0xa6, 0x23
        ])),
    (rng: MersenneTwister, lv: number) => new FixedValue(
        new Uint8Array([127,0]),
        "827f00",
        new Uint8Array([
            0x53, 0xff, 0x18, 0xb4, 0xb7, 0x83, 0x38, 0x80, 0x10, 0xf2, 0x52, 0xdf, 0x71, 0x82, 0xf6, 0x23, 0x65, 0x1e, 0x0c, 0xd2, 0x25, 0xb1, 0x93, 0x9f, 0x72, 0x4f, 0x44, 0x4b, 0x95, 0x01, 0x0e, 0x6d
        ])),
    (rng: MersenneTwister, lv: number) => new FixedValue(
        new Uint8Array([128,0]),
        "828000",
        new Uint8Array([
            0xcc, 0xbe, 0x8b, 0x97, 0xca, 0xfa, 0xa6, 0x52, 0x5e, 0xd5, 0xd6, 0x7b, 0x5b, 0x0e, 0xab, 0x41, 0xd7, 0xf4, 0xa1, 0xe0, 0x42, 0x4e, 0x73, 0xea, 0xc5, 0xdf, 0x2b, 0x02, 0xe1, 0x5a, 0xe3, 0x4d
        ])),
    (rng: MersenneTwister, lv: number) => new FixedValue(
        new Uint8Array([]),
        "80",
        nilHash),
    (rng: MersenneTwister, lv: number) => new FixedValue(
        new Uint8Array([0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00]),
        "c05a800080008000800080008000800080008000800080008000800080008000800080008000800080008000800080008000800080008000800080008000800080008000800080008000800080008000800080008000800080008000",
        new Uint8Array([
            0x99, 0x83, 0xcb, 0xec, 0x49, 0xa7, 0x46, 0x76, 0xb8, 0x43, 0x2e, 0x65, 0xb6, 0xa4, 0xd1, 0x3e, 0xaf, 0x44, 0xe3, 0xd3, 0x2e, 0xf1, 0x77, 0x73, 0x84, 0x25, 0x90, 0x38, 0xce, 0x19, 0x83, 0x90
        ])),
    (rng: MersenneTwister, lv: number) => new FixedValue(
        new Uint8Array([0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
        "bd80008000800080008000800080008000800080008000800080008000800080008000800080008000800080008000000000000000000000000000000000",
        new Uint8Array([
            0x87, 0xab, 0x82, 0x83, 0x3e, 0x7f, 0x9b, 0xbe, 0x1e, 0x6e, 0x61, 0x9a, 0x64, 0x8f, 0x9e, 0xb1, 0xe9, 0x0d, 0x0b, 0x42, 0x8f, 0x74, 0xb0, 0x30, 0xa5, 0x3b, 0x9b, 0xae, 0xc6, 0xcd, 0x43, 0x7d
        ])),
    (rng: MersenneTwister, lv: number) => new FixedValue(
        t([], []),
        "ff8080",
        new Uint8Array([
            0x52, 0xdb, 0x9e, 0xf9, 0x79, 0x86, 0xe7, 0x38, 0x2e, 0xf7, 0x8b, 0x8e, 0xae, 0x2d, 0xac, 0xdb, 0xb2, 0xce, 0x82, 0x3e, 0xd1, 0x39, 0x6a, 0x0f, 0xb2, 0xf7, 0xf1, 0x20, 0xa2, 0xb4, 0x0a, 0x63
        ])),
    (rng: MersenneTwister, lv: number) => new FixedValue(
        -129,
        "82ff7f",
        new Uint8Array([
            0x5a, 0x0c, 0x1f, 0xec, 0x64, 0x75, 0x1e, 0x82, 0xc0, 0xd4, 0x86, 0x1d, 0x0b, 0xc1, 0x9c, 0x75, 0x80, 0x52, 0x5d, 0x2f, 0x47, 0x66, 0x79, 0x56, 0xbb, 0xd9, 0xd7, 0x9e, 0x26, 0x0a, 0xae, 0x00
        ]))
];

function atom_hex(bytes: Uint8Array): string {
    let pad = (hex: string) => {
        if (hex.length % 2 == 1) {
            return '0' + hex;
        } else {
            return hex;
        }
    };
    if (bytes.length == 1 && bytes[0] < 128) {
        return pad(bytes[0].toString(16));
    }
    let result = (bytes.length >= 64) ?
        (0xc000 + bytes.length).toString(16) :
        (0x80 + bytes.length).toString(16);

    for (let i = 0; i < bytes.length; i++) {
        let byte_res = bytes[i].toString(16);
        result += pad(byte_res);
    }
    return result;
}

function find_first_non_pad_byte(bytes: Uint8Array, pad_byte: number): number {
    let i = 0;

    // Find the first non-pad byte.
    for (i = 0; i < 3 && bytes[i] == pad_byte; i++) { }

    return i;
}

//
// Generated values
//
// random i32
// random BigInt
// "String"
// Uint8Array
// a pair that evaulates to t(x,y)
// List of N things
//
class RandomS32Generator implements ConvertibleValue {
    value: number;
    constructor(rng: MersenneTwister) {
        this.value = rngu32(rng) - 0x80000000;
    }
    bytes(): Uint8Array {
        let result: Array<number> = [];

        if (this.value == 0) {
            return new Uint8Array(result);
        }

        let bytes = new Uint8Array([
            (this.value >> 24) & 0xff,
            (this.value >> 16) & 0xff,
            (this.value >> 8) & 0xff,
            this.value & 0xff
        ]);

        let is_negative = this.value < 0;
        let check_for_sign_bit = is_negative ? 0 : 0x80;
        let pad_byte = is_negative ? 0xff : 0;

        let i = find_first_non_pad_byte(bytes, pad_byte);

        if ((bytes[i] & 0x80) == check_for_sign_bit) {
            result.push(pad_byte);
        }
        for (let j = i; j < 4; j++ ) {
            result.push(bytes[j]);
        }

        return new Uint8Array(result);
    }
    toPlainValue(): any {
        return this.value;
    }
    toHexValue(): string {
        return atom_hex(this.bytes());
    }
    shatree(): Uint8Array {
        const hasher = new Sha256();
        hasher.update(new Uint8Array([1]));
        hasher.update(this.bytes());
        return new Uint8Array(hasher.digest());
    }
}

class RandomBigIntGenerator implements ConvertibleValue {
    sign: boolean;
    value: BigInt;
    subtract_from: BigInt;
    constructor(rng: MersenneTwister) {
        this.sign = (rngu32(rng) % 2) != 0;
        let words = 1 + rngu32(rng) % 2;
        let value_string = "0x";

        let subtract_from = this.sign ? BigInt(1) : BigInt(0);

        for (let i = 0; i < words; i++) {
            value_string += rngu32(rng).toString(16);
            subtract_from *= BigInt("0x100000000");
        }
        let sign_big = this.sign ? BigInt(-1) : BigInt(1);
        this.value = BigInt(value_string) * sign_big;
        this.subtract_from = subtract_from;
    }
    toPlainValue(): any {
        return this.value;
    }
    bytes(): Uint8Array {
        let result: Array<number> = [];
        if (this.value == BigInt(0)) {
            return new Uint8Array(result);
        }

        // Generate a sign extended unsigned image of the value.
        let value_string =
            (this.sign ? bi_add(this.subtract_from, this.value) : this.value).toString(16);

        // Pad to bytes.
        if (value_string.length % 2 == 1) {
            value_string = "0" + value_string;
        }

        // Parse as a list of int
        let int_list = [];
        for (let i = 0; i < value_string.length; i += 2) {
            int_list.push(parseInt(value_string.substr(i,2), 16));
        }

        let is_negative = this.value < BigInt(0);
        let check_for_sign_bit = is_negative ? 0 : 0x80;
        let pad_byte = is_negative ? 0xff : 0;

        // Find the first non-pad byte.
        let i = find_first_non_pad_byte(new Uint8Array(int_list), pad_byte);

        if ((int_list[i] & 0x80) == check_for_sign_bit) {
            result.push(pad_byte);
        }

        for (let j = i; j < int_list.length; j++) {
            result.push(int_list[j]);
        }
        return new Uint8Array(result);
    }
    toHexValue(): string {
        return atom_hex(this.bytes());
    }
    shatree(): Uint8Array {
        const hasher = new Sha256();
        hasher.update(new Uint8Array([1]));
        hasher.update(this.bytes());
        return new Uint8Array(hasher.digest());
    }
}

class RandomString implements ConvertibleValue {
    value: string;
    constructor(rng: MersenneTwister) {
        let jumbo = rngu32(rng) % 5 == 0;
        let use_chars = (jumbo ? 100 : 0) + rngu32(rng) % 15;
        let result = "";
        for (let i = 0; i < use_chars; i++) {
            result += String.fromCharCode(0x40 + rngu32(rng) % 256);
        }
        this.value = result;
    }
    toPlainValue() {
        return this.value;
    }
    bytes(): Uint8Array {
        let ab = Buffer.from(this.value, 'utf8');
        return new Uint8Array(ab);
    }
    toHexValue() {
        return atom_hex(this.bytes());
    }
    shatree() {
        const hasher = new Sha256();
        hasher.update(new Uint8Array([1]));
        hasher.update(this.bytes());
        return new Uint8Array(hasher.digest());
    }
};

class RandomBytes implements ConvertibleValue {
    value: Uint8Array;
    constructor(rng: MersenneTwister) {
        let jumbo = rngu32(rng) % 5 == 0;
        let use_chars = (jumbo ? 100 : 0) + rngu32(rng) % 15;
        let value = [];
        for (let i = 0; i < use_chars; i++) {
            value[i] = rngu32(rng) % 256;
        }
        this.value = new Uint8Array(value);
    }
    toPlainValue(): any {
        return this.value;
    }
    bytes(): Uint8Array {
        return this.value;
    }
    toHexValue(): string {
        return atom_hex(this.bytes());
    }
    shatree(): Uint8Array {
        const hasher = new Sha256();
        hasher.update(new Uint8Array([1]));
        hasher.update(new Uint8Array(this.bytes()));
        return new Uint8Array(hasher.digest());
    }
};

class RandomTuple implements ConvertibleValue {
    left: StructureElement;
    right: StructureElement;
    constructor(rng: MersenneTwister, lv: number) {
        this.left = new StructureElement(rng, lv);
        this.right = new StructureElement(rng, lv);
    }
    toPlainValue(): any {
        return t(this.left.toPlainValue(), this.right.toPlainValue());
    }
    toHexValue(): string {
        return 'ff' + this.left.toHexValue() + this.right.toHexValue();
    }
    shatree(): Uint8Array {
        const hasher = new Sha256();
        hasher.update(new Uint8Array([2]));
        hasher.update(this.left.shatree());
        hasher.update(this.right.shatree());
        return new Uint8Array(hasher.digest());
    }
};

class RandomList implements ConvertibleValue {
    value: Array<StructureElement>;
    constructor(rng: MersenneTwister, lv: number) {
        let list_length = rngu32(rng) % 5;
        let result = [];
        for (let i = 0; i < list_length; i++) {
            result.push(new StructureElement(rng, lv));
        }
        this.value = result;
    }
    toPlainValue(): any {
        let result = this.value.map((v) => v.toPlainValue());
        return result;
    }
    toHexValue(): string {
        let result = '80';
        rev_iter(this.value, (v, i) => {
            result = 'ff' + v.toHexValue() + result;
        });
        return result;
    }
    shatree(): Uint8Array {
        let start_hash = nilValue.shatree();
        rev_iter(this.value, (v, i) => {
            const hasher = new Sha256();
            hasher.update(new Uint8Array([2]));
            hasher.update(v.shatree());
            hasher.update(start_hash);
            start_hash = new Uint8Array(hasher.digest());
        });
        return start_hash;
    }
}

const valueGenerators = [
    (rng: MersenneTwister, lv: number) => new RandomS32Generator(rng),
    (rng: MersenneTwister, lv: number) => new RandomBigIntGenerator(rng),
    (rng: MersenneTwister, lv: number) => new RandomString(rng),
    (rng: MersenneTwister, lv: number) => new RandomBytes(rng),
];

const compositeGenerators = [
    (rng: MersenneTwister, lv: number) => new RandomTuple(rng, lv),
    (rng: MersenneTwister, lv: number) => new RandomList(rng, lv)
];

class StructureElement {
    value: ConvertibleValue;
    constructor(rng: MersenneTwister, lv: number) {
        let kind = rngu32(rng) % generateBasicValues.length + valueGenerators.length + (lv < 3 ? compositeGenerators.length : 0);
        if (kind < generateBasicValues.length) {
            this.value = generateBasicValues[kind](rng, lv + 1);
            return;
        }

        kind -= generateBasicValues.length;

        if (kind < valueGenerators.length) {
            this.value = valueGenerators[kind](rng, lv + 1);
            return;
        }

        this.value = compositeGenerators[kind % compositeGenerators.length](rng, lv);
    }
    toPlainValue(): any {
        return this.value.toPlainValue();
    }
    toHexValue(): string {
        return this.value.toHexValue();
    }
    shatree(): Uint8Array {
        return this.value.shatree();
    }
};

describe('Program', () => {
    it('deals with a range of conversions', () => {
        let rng = new MersenneTwister(1);
        let early_converted = Program.to(t("hi","there"));
        let early_hash = early_converted.sha256tree();

        let test = (element: ConvertibleValue, i: number) => {
            console.log('test index', i);
            console.log(element.toPlainValue());
            console.log(element.toHexValue());
            let from_hex_converted = Program.from_hex(element.toHexValue());
            expect(element.toHexValue()).toEqual(from_hex_converted.toString());
            let from_val_converted = Program.to(element.toPlainValue());
            expect(from_hex_converted.toString()).toEqual(from_val_converted.toString());
            let element_shatree = element.shatree();
            expect(element_shatree).toEqual(from_hex_converted.sha256tree());
            expect(element_shatree).toEqual(from_val_converted.sha256tree());
        };

        for (let i = 0; i < generateBasicValues.length; i++) {
            test(generateBasicValues[i](rng, 0), i);
        }
        for (let i = 0; i < 1000; i++) {
            test(new StructureElement(rng, 0), i);
        }

        let late_hash = early_converted.sha256tree();
        expect(early_hash).toEqual(late_hash);
    });
});
