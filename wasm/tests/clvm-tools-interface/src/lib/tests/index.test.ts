import type { IProgram, ITuple } from '../../../../../pkg/clvm_tools_wasm.js';
import type { G1Element } from 'bls-signatures';

import * as fs from 'fs';
import { resolve } from 'path';
import * as assert from 'assert';
import * as bls_loader from 'bls-signatures';
const {h, t, Program} = require('../../../../../pkg/clvm_tools_wasm.js');

it('Has BLS signatures support', async () => {
    const bls = await bls_loader.default();
    const g1element = new bls.G1Element();
    const converted_g1_element = Program.to(g1element);
    assert.equal('b0c00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000', converted_g1_element.toString());
});

it('Has the "h" function', async () => {
    const unhexed = h('21203031');
    assert.equal([0x21, 0x20, 0x30, 0x31].toString(), unhexed.toString());
});

it('Converts uint8arrays', async () => {
    let ua = new Uint8Array(3);
    ua[0] = 0x30;
    ua[1] = 0x81;
    ua[2] = 0xff;
    let p = Program.to(ua);
    assert.equal([0x30, 0x81, 0xff].toString(), ua.toString());
});

it('Converts uint8arrays from the "h" function', async () => {
    let unhexed = h('8b5fd961cb1a826823c45a796b350d0d6ba2398829f249b1e535fd76f6d150b1');
    let p = Program.to([unhexed]);
    assert.equal("ffa08b5fd961cb1a826823c45a796b350d0d6ba2398829f249b1e535fd76f6d150b180", p.toString());
});

it('Converts a buffer', async () => {
    let unhexed = h('8b5fd961cb1a826823c45a796b350d0d6ba2398829f249b1e535fd76f6d150b1');
    let b = Buffer.from(unhexed);
    let p = Program.to([b]);
    assert.equal("ffa08b5fd961cb1a826823c45a796b350d0d6ba2398829f249b1e535fd76f6d150b180", p.toString());
});

it('Converts to string', async () => {
    const converted_sexp = Program.to([1, 2, 3]);
    assert.equal("ff01ff02ff0380", converted_sexp.toString());
});

it('Accepts already converted objects', async () => {
    const converted_sexp = Program.to([1, 2, 3]);
    const twice_converted = Program.to(converted_sexp);
    assert.equal("ff01ff02ff0380", twice_converted.toString());
});

it('Has as_pair', async () => {
    const converted_sexp = Program.to([1, 2, 3]);
    const as_pair = converted_sexp.as_pair();
    assert.equal("01", as_pair[0].toString());
    assert.equal("ff02ff0380", as_pair[1].toString());
});

it('Has null', async () => {
    assert.equal(Program.null().toString(), '80');
});

it('Has listp', async () => {
    const is_list: IProgram = Program.to([1,2,3]);
    const isnt_list: IProgram = Program.to(456);
    assert.equal(is_list.listp(), true);
    assert.equal(isnt_list.listp(), false);
});

it('Has nullp', async () => {
    const is_null = Program.to([]);
    const is_also_null = Program.to(0);
    const isnt_null = Program.to(7);
    const isnt_also_null = Program.to([99,101]);
    assert.equal(is_null.nullp(), true);
    assert.equal(is_also_null.nullp(), true);
    assert.equal(isnt_null.nullp(), false);
    assert.equal(isnt_also_null.nullp(), false);
});

it('Has as_int', async () => {
    const int_value = Program.to(7).as_int();
    assert.equal(int_value, 7);
    try {
        /*const non_int_value =*/ Program.to([7,13]).as_int();
        assert.fail('was an int but should not be');
    } catch (e: any) {
        assert.equal(e.toString(), "not a number");
    }
});

it('Has as_bigint', async () => {
    const int_value = Program.to(10000000000000000000000n).as_bigint();
    assert.equal(int_value, 10000000000000000000000n);
    try {
        /*const non_int_value =*/ Program.to([7,13]).as_bigint();
        assert.fail('');
    } catch (e: any) {
        assert.equal(e.toString(), "not a number");
    }
});

it('Has first and rest', async () => {
    const test_list = Program.to([7,13,17,23]);
    assert.equal(test_list.first().toString(), '07');
    assert.equal(test_list.rest().toString(), 'ff0dff11ff1780');
    try {
        Program.to([]).first();
        assert.fail("empty list had first");
    } catch (e: any) {
        assert.equal(e.toString(), "not a cons");
    }
    try {
        Program.to([]).rest();
        assert.fail("empty list had rest");
    } catch (e: any) {
        assert.equal(e.toString(), "not a cons");
    }
});

it('Has cons', async () => {
    const test_1 = Program.to(7);
    const test_2 = Program.to([8,9,10]);
    const consed = test_1.cons(test_2);
    const test_3 = Program.to([7,8,9,10]);
    assert.equal(consed.toString(), test_3.toString());
});

it('Has the t function', async () => {
    const p1 = Program.to(7);
    const p2 = Program.to(9);
    const tuple: ITuple = t(p1, p2);
    const consed = p1.cons(p2);
    assert.equal(Program.to(tuple).toString(), consed.toString());
});

it('Has as_bin', async () => {
    const test_data = Program.to([7,8,9,10]);
    const as_bin: Uint8Array = test_data.as_bin();
    assert.equal([255,7,255,8,255,9,255,10,128].toString(), as_bin.toString());
});

it('Has list_len', async () => {
    const list_data = Program.to([7,8,9,10]);
    const list_len = list_data.list_len();
    assert.equal(list_len, 4);
    const not_list = Program.to(16);
    const not_list_len = not_list.list_len();
    assert.equal(not_list_len, 0);
});

it('Has equal_to', async () => {
    const p1 = Program.to([7,8,[9,10],11]);
    const p2 = Program.from_hex('ff07ff08ffff09ff0a80ff0b80');
    const p3 = Program.to([7,8,[9,11],11]);
    assert.ok(p1.equal_to(p2));
    assert.ok(!p1.equal_to(p3));
    assert.ok(!p2.equal_to(p3));
});

it('Has as_javascript', async () => {
    const tuple = t(9,(t(10,11)));
    const original = [7,8,tuple,12];
    const p1 = Program.to(original);
    const p1_as_js = p1.as_javascript();
    assert.equal(original.toString(), p1_as_js.toString());
});

it('Has run', async () => {
    const program = Program.from_hex('ff12ffff10ff02ffff010180ffff11ff02ffff01018080');
    const args = Program.to([13]);
    const [cost, run_result] = program.run(args);
    assert.equal(run_result.toString(), '8200a8');
    assert.equal(cost, 2658);
});

it('Has curry', async () => {
    const program = Program.from_hex('ff12ffff10ff02ffff010180ffff11ff02ffff01018080');
    const program_with_arg = program.curry(Program.to(13));
    const [cost, run_result] = program_with_arg.run(Program.to([]));
    assert.equal(run_result.toString(), '8200a8');
    assert.equal(cost, 2884);
});

export class ChiaExample {
    private MOD: IProgram;

    constructor(MOD: IProgram) {
        this.MOD = MOD;
    }
    public puzzle_for_synthetic_public_key(synthetic_public_key: G1Element): IProgram {
        return this.MOD.curry(Program.to(synthetic_public_key));
    }
}

it('works as expected in context', async () => {
    const bls = await bls_loader.default();
    const program_text = fs.readFileSync(resolve(__dirname, '../../../content/p2_delegated_puzzle_or_hidden_puzzle.clvm.hex'),'utf-8');
    const MOD: IProgram = Program.from_hex(program_text);
    const ce = new ChiaExample(MOD);
    const sk = bls.AugSchemeMPL.key_gen(new Uint8Array([
        0, 50, 6, 244, 24, 199, 1, 25, 52, 88, 192, 19, 18, 12, 89, 6, 220,
        18, 102, 58, 209, 82, 12, 62, 89, 110, 182, 9, 44, 20, 254, 22
    ]));
    const pk = bls.AugSchemeMPL.sk_to_g1(sk);
    // pk bytes 86243290bbcbfd9ae75bdece7981965350208eb5e99b04d5cd24e955ada961f8c0a162dee740be7bdc6c3c0613ba2eb1
    // Expected puzzle hash = 30cdae3d54778db5eba21584c452cfb1a278136b2ec352ba44a52078efea7507
    const target_puzzle = ce.puzzle_for_synthetic_public_key(pk);
    const shatree_result: Uint8Array = target_puzzle.sha256tree();
    assert.equal(shatree_result.toString(), h('30cdae3d54778db5eba21584c452cfb1a278136b2ec352ba44a52078efea7507').toString());
});

const cat2_puzzle = 'ff02ffff01ff02ff5effff04ff02ffff04ffff04ff05ffff04ffff0bff2cff0580ffff04ff0bff80808080ffff04ffff02ff17ff2f80ffff04ff5fffff04ffff02ff2effff04ff02ffff04ff17ff80808080ffff04ffff0bff82027fff82057fff820b7f80ffff04ff81bfffff04ff82017fffff04ff8202ffffff04ff8205ffffff04ff820bffff80808080808080808080808080ffff04ffff01ffffffff81ca3dff46ff0233ffff3c04ff01ff0181cbffffff02ff02ffff03ff05ffff01ff02ff32ffff04ff02ffff04ff0dffff04ffff0bff22ffff0bff2cff3480ffff0bff22ffff0bff22ffff0bff2cff5c80ff0980ffff0bff22ff0bffff0bff2cff8080808080ff8080808080ffff010b80ff0180ffff02ffff03ff0bffff01ff02ffff03ffff09ffff02ff2effff04ff02ffff04ff13ff80808080ff820b9f80ffff01ff02ff26ffff04ff02ffff04ffff02ff13ffff04ff5fffff04ff17ffff04ff2fffff04ff81bfffff04ff82017fffff04ff1bff8080808080808080ffff04ff82017fff8080808080ffff01ff088080ff0180ffff01ff02ffff03ff17ffff01ff02ffff03ffff20ff81bf80ffff0182017fffff01ff088080ff0180ffff01ff088080ff018080ff0180ffff04ffff04ff05ff2780ffff04ffff10ff0bff5780ff778080ff02ffff03ff05ffff01ff02ffff03ffff09ffff02ffff03ffff09ff11ff7880ffff0159ff8080ff0180ffff01818f80ffff01ff02ff7affff04ff02ffff04ff0dffff04ff0bffff04ffff04ff81b9ff82017980ff808080808080ffff01ff02ff5affff04ff02ffff04ffff02ffff03ffff09ff11ff7880ffff01ff04ff78ffff04ffff02ff36ffff04ff02ffff04ff13ffff04ff29ffff04ffff0bff2cff5b80ffff04ff2bff80808080808080ff398080ffff01ff02ffff03ffff09ff11ff2480ffff01ff04ff24ffff04ffff0bff20ff2980ff398080ffff010980ff018080ff0180ffff04ffff02ffff03ffff09ff11ff7880ffff0159ff8080ff0180ffff04ffff02ff7affff04ff02ffff04ff0dffff04ff0bffff04ff17ff808080808080ff80808080808080ff0180ffff01ff04ff80ffff04ff80ff17808080ff0180ffffff02ffff03ff05ffff01ff04ff09ffff02ff26ffff04ff02ffff04ff0dffff04ff0bff808080808080ffff010b80ff0180ff0bff22ffff0bff2cff5880ffff0bff22ffff0bff22ffff0bff2cff5c80ff0580ffff0bff22ffff02ff32ffff04ff02ffff04ff07ffff04ffff0bff2cff2c80ff8080808080ffff0bff2cff8080808080ffff02ffff03ffff07ff0580ffff01ff0bffff0102ffff02ff2effff04ff02ffff04ff09ff80808080ffff02ff2effff04ff02ffff04ff0dff8080808080ffff01ff0bff2cff058080ff0180ffff04ffff04ff28ffff04ff5fff808080ffff02ff7effff04ff02ffff04ffff04ffff04ff2fff0580ffff04ff5fff82017f8080ffff04ffff02ff7affff04ff02ffff04ff0bffff04ff05ffff01ff808080808080ffff04ff17ffff04ff81bfffff04ff82017fffff04ffff0bff8204ffffff02ff36ffff04ff02ffff04ff09ffff04ff820affffff04ffff0bff2cff2d80ffff04ff15ff80808080808080ff8216ff80ffff04ff8205ffffff04ff820bffff808080808080808080808080ff02ff2affff04ff02ffff04ff5fffff04ff3bffff04ffff02ffff03ff17ffff01ff09ff2dffff0bff27ffff02ff36ffff04ff02ffff04ff29ffff04ff57ffff04ffff0bff2cff81b980ffff04ff59ff80808080808080ff81b78080ff8080ff0180ffff04ff17ffff04ff05ffff04ff8202ffffff04ffff04ffff04ff24ffff04ffff0bff7cff2fff82017f80ff808080ffff04ffff04ff30ffff04ffff0bff81bfffff0bff7cff15ffff10ff82017fffff11ff8202dfff2b80ff8202ff808080ff808080ff138080ff80808080808080808080ff018080';

const cat2_curried_program = 'ff02ffff01ff02ffff01ff02ffff03ff0bffff01ff02ffff03ffff09ff05ffff1dff0bffff1effff0bff0bffff02ff06ffff04ff02ffff04ff17ff8080808080808080ffff01ff02ff17ff2f80ffff01ff088080ff0180ffff01ff04ffff04ff04ffff04ff05ffff04ffff02ff06ffff04ff02ffff04ff17ff80808080ff80808080ffff02ff17ff2f808080ff0180ffff04ffff01ff32ff02ffff03ffff07ff0580ffff01ff0bffff0102ffff02ff06ffff04ff02ffff04ff09ff80808080ffff02ff06ffff04ff02ffff04ff0dff8080808080ffff01ff0bffff0101ff058080ff0180ff018080ffff04ffff01b0a7ca4bce10200d073ef10c46e9d27c3b4e31263d4c07fbec447650fcc1b286300e8ecf25c0560f9cb5aa673247fb6a6fff018080';

it('can uncurry an example program', async () => {
    const to_uncurry_text = fs.readFileSync(resolve(__dirname, '../../../content/test_cat2_program.hex'),'utf-8');
    const program = Program.from_hex(to_uncurry_text);
    const uncurried = program.uncurry_error();
    assert.equal(uncurried.length, 2);
    assert.equal(uncurried[1].length, 3);
    assert.equal(uncurried[0].toString(), cat2_puzzle);
    assert.equal(uncurried[1][0].toString(), 'a072dec062874cd4d3aab892a0906688a1ae412b0109982e1797a170add88bdcdc');
    assert.equal(uncurried[1][1].toString(), 'a06d95dae356e32a71db5ddcb42224754a02524c615c5fc35f568c2af04774e589');
    assert.equal(uncurried[1][2].toString(), cat2_curried_program);
});

// Thanks: fcoleman
describe('Program', () => {
    it('preserves the representation across js and hex representations', () => {
        const puzzle1 = Program.to([0]);
        const puzzleHash1 = Buffer.from(puzzle1.sha256tree()).toString('hex');

        const puzzle2 = Program.from_hex(puzzle1.toString());
        const puzzleHash2 = Buffer.from(puzzle2.sha256tree()).toString('hex');

        const puzzle3 = Program.from_hex('ff8080');
        const puzzleHash3 = Buffer.from(puzzle3.sha256tree()).toString('hex');

        expect(puzzle1.toString()).toEqual(puzzle2.toString());
        expect(puzzle1.toString()).toEqual(puzzle3.toString());
        expect(puzzleHash1).toEqual(puzzleHash2);
        expect(puzzleHash1).toEqual(puzzleHash3);
    });
})
