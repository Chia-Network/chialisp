/* tslint:disable */
/* eslint-disable */
export function create_clvm_runner(hex_prog: string, args_js: any, symbols: object, overrides: object): any;
export function final_value(runner: number): any;
export function remove_clvm_runner(runner: number): void;
export function run_step(runner: number): any;
export function compile(input_js: any, filename_js: any, search_paths_js: any[]): any;
export function compose_run_function(hex_prog: string, symbol_table_js: object, function_name: string): any;
export function create_repl(): number;
export function destroy_repl(repl_id: number): void;
export function repl_run_string(repl_id: number, input: string): any;
export function sexp_to_string(v: any): any;
export function h(v: string): Uint8Array;
export function t(a: any, b: any): any;

interface ITuple {
    to_program(): IProgram;
}

interface IProgram {
    toString(): string;
    as_pair(): ITuple;
    listp(): boolean;
    nullp(): boolean;
    as_int(): number;
    as_bigint(): bigint;
    as_bin(): Uint8Array;
    first(): IProgram;
    rest(): IProgram;
    cons(p: IProgram): IProgram;
    run(env: IProgram): [number, IProgram];
    list_len(): number;
    equal_to(other: IProgram): boolean;
    as_javascript(): any;
    curry(args: [IProgram]): IProgram;
    sha256tree(): Uint8Array;
    uncurry_error(): [IProgram, Array<IProgram>];
    uncurry(): [IProgram, Array<IProgram>|null];
}


export class Program {
  private constructor();
/**
** Return copy of self without private attributes.
*/
  toJSON(): Object;
/**
* Return stringified version of self.
*/
  toString(): string;
  free(): void;
  static to_internal(input: any): any;
  static to(input: any): IProgram;
  static from_hex(input: string): IProgram;
  static null(): IProgram;
  static sha256tree_internal(obj: any): Uint8Array;
  static to_string_internal(obj: any): any;
  static as_pair_internal(obj: any): any;
  static listp_internal(obj: any): boolean;
  static nullp_internal(obj: any): boolean;
  static as_int_internal(obj: any): number;
  static as_bigint_internal(obj: any): bigint;
  static first_internal(obj: any): IProgram;
  static rest_internal(obj: any): IProgram;
  static cons_internal(obj: any, other: any): IProgram;
  static run_internal(obj: any, args: any): any;
  static tuple_to_program_internal(obj: any): IProgram;
  static as_bin_internal(obj: any): Uint8Array;
  static list_len_internal(obj: any): number;
  static equal_to_internal(a: any, b: any): boolean;
  static as_javascript_internal(obj: any): any;
  static curry_internal(obj: any, args: any[]): IProgram;
  static uncurry_error_internal(obj: any): IProgram[];
  static uncurry_internal(obj: any): IProgram[];
}
