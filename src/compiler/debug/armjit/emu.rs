// Based on https://github.com/daniel5151/gdbstub/blob/master/examples/armv4t/emu.rs

use num_bigint::ToBigInt;
use std::borrow::Borrow;
use std::collections::HashMap;
use std::fs;
use std::rc::Rc;

use clvmr::Allocator;
use tempfile::NamedTempFile;

use armv4t_emu::reg;
use armv4t_emu::Cpu;
use armv4t_emu::Memory;
use armv4t_emu::Mode;
use gdbstub::arch::Arch;
use gdbstub::common::Pid;
use gdbstub::target::ext::base::singlethread::{
    SingleThreadBase, SingleThreadResume, SingleThreadResumeOps,
};
use gdbstub::target::ext::base::{single_register_access, BaseOps};
use gdbstub::target::ext::breakpoints::{
    Breakpoints, BreakpointsOps, HwBreakpointOps, HwWatchpointOps, SwBreakpoint, SwBreakpointOps,
};
use gdbstub::target::{Target, TargetResult};

use crate::classic::clvm::__type_compatibility__::{bi_zero, Bytes, BytesFromType};
use crate::classic::clvm_tools::stages::stage_0::DefaultProgramRunner;

use crate::compiler::cldb::is_print_request;
use crate::compiler::clvm::{apply_op, sha256tree};
use crate::compiler::compiler::{compile_file, is_apply, DefaultCompilerOpts};
use crate::compiler::comptypes::CompilerOpts;
use crate::compiler::debug::armjit::code::{
    swi_print, Encodable, Instr, Program, Register, NEXT_ALLOC_OFFSET, SWI_DISPATCH_INSTRUCTION,
    SWI_DISPATCH_NEW_CODE, SWI_DONE, SWI_PRINT_EXPR, SWI_THROW, TARGET_ADDR,
};
use crate::compiler::debug::armjit::load::{ElfLoader, EmuSymbolInfo};
use crate::compiler::debug::armjit::memory::{PagedMemory, TargetMemory};
use crate::compiler::debug::build_symbol_table_mut;
use crate::compiler::debug::u8_from_number;
use crate::compiler::dialect::AcceptedDialect;
#[cfg(test)]
use crate::compiler::frontend::frontend;
use crate::compiler::prims::primquote;
#[cfg(test)]
use crate::compiler::sexp::decode_string;
use crate::compiler::sexp::{parse_sexp, SExp};
use crate::compiler::srcloc::Srcloc;
use crate::compiler::TRunProgram;

pub type DynResult<T> = Result<T, Box<dyn std::error::Error>>;

const HLE_RETURN_ADDR: u32 = 0x12345678;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Event {
    DoneStep,
    Halted,
    Output,
    Trap,
    Break,
    WatchWrite(u32),
    WatchRead(u32),
}

pub enum ExecMode {
    Step,
    Continue,
    RangeStep(u32, u32),
}

/// incredibly barebones armv4t-based emulator
pub struct Emu {
    start_addr: u32,
    next_module_addr: u32,

    // example custom register. only read/written to from the GDB client
    pub custom_reg: u32,

    pub exec_mode: ExecMode,

    pub cpu: Cpu,
    pub mem: PagedMemory,

    pub watchpoints: Vec<u32>,
    pub breakpoints: Vec<u32>,

    pub reported_pid: Pid,

    pub clvm_symbols: Rc<HashMap<String, String>>,
    pub jit_symbols: Rc<HashMap<String, EmuSymbolInfo>>,

    pending_gdb_console_output: Vec<String>,
}

impl SingleThreadBase for Emu {
    /// Read the target's registers.
    fn read_registers(
        &mut self,
        regs: &mut <Self::Arch as Arch>::Registers,
    ) -> TargetResult<(), Self> {
        for i in 0..13 {
            regs.r[i] = self.cpu.reg_get(Mode::User, i as u8);
        }
        regs.sp = self.cpu.reg_get(Mode::User, reg::SP);
        regs.lr = self.cpu.reg_get(Mode::User, reg::LR);
        regs.pc = self.cpu.reg_get(Mode::User, reg::PC);
        Ok(())
    }

    /// Write the target's registers.
    fn write_registers(
        &mut self,
        regs: &<Self::Arch as Arch>::Registers,
    ) -> TargetResult<(), Self> {
        todo!();
    }

    /// Support for single-register access.
    /// See [`SingleRegisterAccess`] for more details.
    ///
    /// While this is an optional feature, it is **highly recommended** to
    /// implement it when possible, as it can significantly improve performance
    /// on certain architectures.
    ///
    /// [`SingleRegisterAccess`]:
    /// super::single_register_access::SingleRegisterAccess
    #[inline(always)]
    fn support_single_register_access(
        &mut self,
    ) -> Option<single_register_access::SingleRegisterAccessOps<'_, (), Self>> {
        None
    }

    /// Read bytes from the specified address range and return the number of
    /// bytes that were read.
    ///
    /// Implementations may return a number `n` that is less than `data.len()`
    /// to indicate that memory starting at `start_addr + n` cannot be
    /// accessed.
    ///
    /// Implemenations may also return an appropriate non-fatal error if the
    /// requested address range could not be accessed (e.g: due to MMU
    /// protection, unhanded page fault, etc...).
    ///
    /// Implementations must guarantee that the returned number is less than or
    /// equal `data.len()`.
    fn read_addrs(
        &mut self,
        start_addr: <Self::Arch as Arch>::Usize,
        data: &mut [u8],
    ) -> TargetResult<usize, Self> {
        for (i, d) in data.iter_mut().enumerate() {
            *d = self.mem.r8(start_addr as u32 + i as u32);
        }
        Ok(data.len())
    }

    /// Write bytes to the specified address range.
    ///
    /// If the requested address range could not be accessed (e.g: due to
    /// MMU protection, unhanded page fault, etc...), an appropriate
    /// non-fatal error should be returned.
    fn write_addrs(
        &mut self,
        start_addr: <Self::Arch as Arch>::Usize,
        data: &[u8],
    ) -> TargetResult<(), Self> {
        todo!();
    }

    /// Support for resuming the target (e.g: via `continue` or `step`)
    #[inline(always)]
    fn support_resume(&mut self) -> Option<SingleThreadResumeOps<'_, Self>> {
        Some(self)
    }
}

impl Target for Emu {
    type Error = ();
    type Arch = gdbstub_arch::arm::Armv4t; // as an example

    #[inline(always)]
    fn base_ops(&mut self) -> BaseOps<Self::Arch, Self::Error> {
        BaseOps::SingleThread(self)
    }

    // opt-in to support for setting/removing breakpoints
    #[inline(always)]
    fn support_breakpoints(&mut self) -> Option<BreakpointsOps<Self>> {
        Some(self)
    }
}

impl SwBreakpoint for Emu {
    ///
    /// Return `Ok(false)` if the operation could not be completed.
    fn add_sw_breakpoint(
        &mut self,
        addr: <Self::Arch as Arch>::Usize,
        kind: <Self::Arch as Arch>::BreakpointKind,
    ) -> TargetResult<bool, Self> {
        self.breakpoints.push(addr);
        eprintln!("add breakpoint {kind:?} {addr}");
        Ok(true)
    }

    /// Remove an existing software breakpoint.
    ///
    /// Return `Ok(false)` if the operation could not be completed.
    fn remove_sw_breakpoint(
        &mut self,
        addr: <Self::Arch as Arch>::Usize,
        kind: <Self::Arch as Arch>::BreakpointKind,
    ) -> TargetResult<bool, Self> {
        let found = self
            .breakpoints
            .iter()
            .position(|u| *u == (addr as u32))
            .clone();
        eprintln!("have breakpoint (to delete) {found:?}");
        if let Some(found) = found {
            self.breakpoints.remove(found);
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

impl Breakpoints for Emu {
    /// Support for setting / removing software breakpoints.
    #[inline(always)]
    fn support_sw_breakpoint(&mut self) -> Option<SwBreakpointOps<'_, Self>> {
        Some(self)
    }

    /// Support for setting / removing hardware breakpoints.
    #[inline(always)]
    fn support_hw_breakpoint(&mut self) -> Option<HwBreakpointOps<'_, Self>> {
        None
    }

    /// Support for setting / removing hardware watchpoints.
    #[inline(always)]
    fn support_hw_watchpoint(&mut self) -> Option<HwWatchpointOps<'_, Self>> {
        None
    }
}

impl SingleThreadResume for Emu {
    fn resume(
        &mut self,
        sig: std::option::Option<gdbstub::common::Signal>,
    ) -> Result<(), <Self as gdbstub::target::Target>::Error> {
        return Ok(());
    }
}

fn is_apply_operator(sexp: Rc<SExp>) -> bool {
    if let SExp::Cons(_, h, t) = &*sexp {
        if is_apply(&h) {
            return true;
        }
    }

    false
}

fn is_quote(sexp: &SExp) -> bool {
    if let SExp::Atom(_, a) = sexp {
        return a == &[1];
    }
    false
}

fn is_quote_operator(sexp: Rc<SExp>) -> bool {
    if let SExp::Cons(_, h, t) = &*sexp {
        if is_quote(&h.atomize()) {
            return true;
        }
    }

    false
}

fn match_printing(operator: Rc<SExp>, sexp: Rc<SExp>) -> Option<Rc<SExp>> {
    if let Ok(v) = operator.get_number() {
        if v == 34_u32.to_bigint().unwrap() {
            if let Some((_, outputs)) = is_print_request(&sexp) {
                return Some(outputs);
            }
        }
    }

    None
}

impl Emu {
    fn get_nul_terminated_string(&self, addr: u32) -> Option<String> {
        if addr == 0 {
            return None;
        }

        let mut out = Vec::new();
        for i in 0..1024_u32 {
            let b = self.mem.read_u8(addr + i);
            if b == 0 {
                return Some(String::from_utf8_lossy(&out).to_string());
            }
            out.push(b);
        }

        None
    }

    fn lookup_dispatch_target_by_hash(&self, hash_hex: &str) -> Option<u32> {
        if let Some(lookup) = self.jit_symbols.get(hash_hex) {
            return Some(lookup.address);
        }

        let alias_symbol = format!("_$_{hash_hex}");
        if let Some(alias_info) = self.jit_symbols.get(&alias_symbol) {
            if let Some(alias_name) = self.get_nul_terminated_string(alias_info.address) {
                if let Some(lookup) = self.jit_symbols.get(&alias_name) {
                    return Some(lookup.address);
                }
            }
        }

        None
    }

    pub fn new(
        program_elf: &[u8],
        start_addr: u32,
        clvm_symbols: Rc<HashMap<String, String>>,
    ) -> DynResult<Emu> {
        // set up emulated system
        let mut cpu = Cpu::new();
        let mut mem = PagedMemory::default();

        // copy all in-memory sections from the ELF file into system RAM
        let mut elf_loader = ElfLoader::new(program_elf, start_addr).expect("should load");
        elf_loader.load(&mut mem);

        let jit_symbols = Rc::new(elf_loader.get_symbols());

        // setup execution state
        cpu.reg_set(Mode::User, reg::SP, 0xffffff00);
        cpu.reg_set(Mode::User, reg::LR, HLE_RETURN_ADDR);
        cpu.reg_set(Mode::User, reg::PC, start_addr);
        cpu.reg_set(Mode::User, reg::CPSR, 0x10); // user mode

        Ok(Emu {
            start_addr: start_addr,
            next_module_addr: elf_loader.next_free_addr(),

            custom_reg: 0x12345678,

            exec_mode: ExecMode::Continue,

            cpu,
            mem,

            watchpoints: Vec::new(),
            breakpoints: Vec::new(),

            reported_pid: Pid::new(1).unwrap(),

            jit_symbols,
            clvm_symbols,

            pending_gdb_console_output: Vec::new(),
        })
    }

    pub(crate) fn reset(&mut self) {
        self.cpu.reg_set(Mode::User, reg::SP, 0xffffff00);
        self.cpu.reg_set(Mode::User, reg::LR, HLE_RETURN_ADDR);
        self.cpu.reg_set(Mode::User, reg::PC, self.start_addr);
        self.cpu.reg_set(Mode::User, reg::CPSR, 0x10);
    }

    pub(crate) fn take_pending_gdb_console_output(&mut self) -> Vec<String> {
        std::mem::take(&mut self.pending_gdb_console_output)
    }

    fn allocate_and_write(&mut self, alloc_ptr: u32, sexp: Rc<SExp>) -> u32 {
        let current_addr = self.mem.read_u32(alloc_ptr);
        match &sexp.atomize() {
            SExp::Cons(_, a, b) => {
                self.mem.write_u32(alloc_ptr, current_addr + 8);
                let a_res = self.allocate_and_write(alloc_ptr, a.clone());
                let b_res = self.allocate_and_write(alloc_ptr, b.clone());
                self.mem.write_u32(current_addr, a_res);
                self.mem.write_u32(current_addr + 4, b_res);
            }
            SExp::Atom(_, v) => {
                let length_to_write = ((v.len() + 3) & !3) as u32;
                self.mem
                    .write_u32(alloc_ptr, current_addr + length_to_write + 4);
                self.mem.write_u32(current_addr, v.len() as u32 * 2 + 1);
                self.mem.write_data(v, current_addr + 4);
            }
            SExp::Nil(_) => {
                self.mem.write_u32(alloc_ptr, current_addr + 4);
                self.mem.write_u32(current_addr, 1);
            }
            SExp::Integer(_, i) => {
                if *i == bi_zero() {
                    self.mem.write_u32(alloc_ptr, current_addr + 4);
                    self.mem.write_u32(current_addr, 1);
                    return current_addr;
                }
                let v = u8_from_number(i.clone());
                let length_to_write = ((v.len() + 3) & !3) as u32;
                self.mem
                    .write_u32(alloc_ptr, current_addr + length_to_write + 4);
                self.mem.write_u32(current_addr, v.len() as u32 * 2 + 1);
                self.mem.write_data(&v, current_addr + 4);
            }
            SExp::QuotedString(_, _, v) => {
                let length_to_write = ((v.len() + 3) & !3) as u32;
                self.mem
                    .write_u32(alloc_ptr, current_addr + length_to_write + 4);
                self.mem.write_u32(current_addr, v.len() as u32 * 2 + 1);
                self.mem.write_data(v, current_addr + 4);
            }
        }
        current_addr
    }

    fn do_apply_op(
        &mut self,
        runner: Rc<dyn TRunProgram>,
        srcloc: &Srcloc,
        operator: Rc<SExp>,
        args: Rc<SExp>,
    ) -> Option<Event> {
        let mut allocator = Allocator::new();
        let alloc_ptr = self.cpu.reg_get(Mode::User, 5);
        let mut debug = false;
        if let Some(printing) = match_printing(operator.clone(), args.clone()) {
            self.pending_gdb_console_output
                .push(format!("DEBUG: {printing}"));
            debug = true;
        }
        match apply_op(
            &mut allocator,
            runner.clone(),
            srcloc.clone(),
            operator.clone(),
            args.clone(),
        ) {
            Ok(res) => {
                // Allocate and write back result.
                let write_result = self.allocate_and_write(alloc_ptr, res.clone());
                self.cpu.reg_set(Mode::User, 0, write_result);
                // Increment pc, we handled the operation.
                let pc = self.cpu.reg_get(Mode::User, reg::PC);
                self.cpu.reg_set(Mode::User, reg::PC, pc + 4);
                if debug {
                    Some(Event::Output)
                } else {
                    None
                }
            }
            Err(e) => {
                eprintln!("error simulating instruction: {e:?}");
                Some(Event::Trap)
            }
        }
    }

    fn do_trap(&mut self, pc: u32, value: usize) -> Option<Event> {
        let srcloc = Srcloc::start("*emu*");
        let runner = Rc::new(DefaultProgramRunner::new());

        if value == SWI_DONE {
            Some(Event::Halted)
        } else if value == SWI_THROW {
            Some(Event::Trap)
        } else if value == SWI_DISPATCH_NEW_CODE {
            let r0_value = self.cpu.reg_get(Mode::User, 0);
            let to_run = self.get_sexp(&srcloc, r0_value);

            let env_value = self.cpu.reg_get(Mode::User, 7);
            let env = self.get_sexp(&srcloc, env_value);

            let hash = sha256tree(to_run.clone());
            let string_of_hash = Bytes::new(Some(BytesFromType::Raw(hash))).hex();

            // We have unknown code in to_run.
            //
            // There are two cases:
            //
            // 1) jit_symbols contains a match for the hash of to_run.
            //    In that case, we can transfer control to that function as though
            //    it was a function call.
            //
            // 2) jit_symbols does not contain a symbol for this.  In this case,
            //    we allocate space using the allocation ptr in r5 and generate
            //    code for the first operator in the given clvm with each argument
            //    computed via an SWI_DISPATCH_NEW_CODE instruction.
            //
            //    We will keep reentering the emulator this way until we find
            //    a match or emit a primitive instruction that is freestanding.

            // Setup stack frame in code buffer.

            let current_pc = self.cpu.reg_get(Mode::User, reg::PC);
            if let Some(dispatch_address) = self.lookup_dispatch_target_by_hash(&string_of_hash) {
                // We found it, transfer control.
                self.cpu
                    .reg_set(Mode::User, 0, self.cpu.reg_get(Mode::User, 7));
                self.cpu.reg_set(Mode::User, reg::LR, current_pc + 8);
                self.cpu.reg_set(Mode::User, reg::PC, current_pc + 4);
                self.cpu.reg_set(Mode::User, 1, dispatch_address);
                return None;
            };

            // Quoted is easy.
            if is_quote_operator(to_run.clone()) {
                self.cpu.reg_set(Mode::User, 0, self.mem.r32(r0_value + 4));
                self.cpu.reg_set(Mode::User, 1, current_pc + 8);
                self.cpu.reg_set(Mode::User, reg::PC, current_pc + 4);
                return None;
            }

            if let Ok(number) = to_run.get_number() {
                // Path retrieval.
                let new_expr = Rc::new(SExp::Cons(
                    srcloc.clone(),
                    to_run.clone(),
                    Rc::new(SExp::Cons(
                        srcloc.clone(),
                        env.clone(),
                        Rc::new(SExp::Nil(srcloc.clone())),
                    )),
                ));
                if let Some(error) = self.do_apply_op(
                    runner,
                    &srcloc,
                    Rc::new(SExp::Atom(srcloc.clone(), vec![2])),
                    new_expr,
                ) {
                    return Some(error);
                }
                self.cpu.reg_set(Mode::User, 1, current_pc + 8);
                return None;
            }

            let mut address_list = vec![];
            let mut value_addr = r0_value;
            while value_addr != 0 && self.mem.r32(value_addr).is_multiple_of(2) {
                address_list.push(value_addr);
                value_addr = self.mem.r32(value_addr + 4);
            }
            let apply_operator = is_apply_operator(to_run.clone());
            let arg_addresses: Vec<u32> = address_list
                .iter()
                .skip(1)
                .map(|arg_cons_addr| self.mem.r32(*arg_cons_addr))
                .collect();

            if (value_addr & 1) != 0 && (value_addr >> 1) != 0 {
                // Not a proper list.
                return Some(Event::Trap);
            }

            let alloc_address = self.cpu.reg_get(Mode::User, 5);
            // Structure of data area:
            //                                                offset
            // pointer to next argument                       0
            // pointer to cons                                4
            // operator address                               8
            // code                                           12
            // reverse order argument addresses               after code

            let new_code_address = self.mem.r32(alloc_address);

            // Emit code for each argument in reverse order, accumulating into r0.
            let mut instruction_list = vec![];
            // Values on the stack:
            // Pointer to first argument pointer.  Will be fixed up.
            instruction_list.push(Instr::Long(0));
            // Constructed value for operator evaluation.
            instruction_list.push(Instr::Long(0));
            // Operator sexp.
            instruction_list.push(Instr::Long(self.mem.r32(r0_value) as usize));

            // Push the stack for this.
            instruction_list.push(Instr::Push(vec![Register::FP, Register::LR]));
            instruction_list.push(Instr::Addi(Register::FP, Register::SP, 4));
            instruction_list.push(Instr::Subi(Register::SP, Register::SP, 0x18));
            instruction_list.push(Instr::Str(Register::R(4), Register::SP, 0));
            instruction_list.push(Instr::Str(Register::R(5), Register::SP, 4));
            instruction_list.push(Instr::Str(Register::R(6), Register::SP, 8));
            instruction_list.push(Instr::Str(Register::R(7), Register::SP, 12));
            instruction_list.push(Instr::Subi(
                Register::R(6),
                Register::PC,
                4 * (instruction_list.len() + 2) as i32,
            ));

            // Handle an apply instruction inline.
            if apply_operator {
                // It acts as throw when it doesn't have the right arguments.
                if !matches!(to_run.proper_list().map(|l| l.len()), Some(3)) {
                    return Some(Event::Break);
                }

                // Load new env ptr.
                // It's the second head of this cons chain.
                instruction_list.push(Instr::Ldr(Register::R(0), Register::R(6), 0));
                instruction_list.push(Instr::Ldr(Register::R(0), Register::R(0), 4));
                // Evalute env.
                instruction_list.push(Instr::Swi(SWI_DISPATCH_NEW_CODE));
                instruction_list.push(Instr::Bx(Register::R(1)));
                // Save env ptr in r4.
                instruction_list.push(Instr::Addi(Register::R(4), Register::R(0), 0));
                // Load code.
                instruction_list.push(Instr::Ldr(Register::R(0), Register::R(6), 0));
                instruction_list.push(Instr::Ldr(Register::R(0), Register::R(0), 0));
                // Evaluate code.
                instruction_list.push(Instr::Swi(SWI_DISPATCH_NEW_CODE));
                instruction_list.push(Instr::Bx(Register::R(1)));
                // Load env from r4.
                instruction_list.push(Instr::Addi(Register::R(7), Register::R(4), 0));

                // Dispatch the actual code.
                instruction_list.push(Instr::Swi(SWI_DISPATCH_NEW_CODE));
                instruction_list.push(Instr::Bx(Register::R(1)));
                instruction_list.push(Instr::Ldr(Register::R(7), Register::SP, 12));
            } else {
                // Arguments in env are in a proper list.  Emit code to iterate it from end to start,
                for i in (0..arg_addresses.len()).rev() {
                    instruction_list.push(Instr::Ldr(Register::R(1), Register::R(6), 0));
                    instruction_list.push(Instr::Ldr(
                        Register::R(0),
                        Register::R(1),
                        (i * 4) as i32,
                    ));
                    // Now we have the code to dispatch in r0.
                    instruction_list.push(Instr::Swi(SWI_DISPATCH_NEW_CODE));
                    // Follow dispatcher-selected code target.
                    instruction_list.push(Instr::Bx(Register::R(1)));
                    // Result is in R0.
                    // Allocate a cons and compose it.
                    instruction_list.push(Instr::Ldr(
                        Register::R(2),
                        Register::R(5),
                        NEXT_ALLOC_OFFSET,
                    ));
                    // Bump r2 to point to the next unallocated space.
                    instruction_list.push(Instr::Addi(Register::R(3), Register::R(2), 8));
                    // Update the allocation ptr.
                    instruction_list.push(Instr::Str(
                        Register::R(3),
                        Register::R(5),
                        NEXT_ALLOC_OFFSET,
                    ));
                    // Set the head of the cons to the newly evaluated argument.
                    instruction_list.push(Instr::Str(Register::R(0), Register::R(2), 0));
                    // Tail points at the currently built argument list.
                    instruction_list.push(Instr::Ldr(Register::R(3), Register::R(6), 4));
                    instruction_list.push(Instr::Str(Register::R(3), Register::R(2), 4));
                    // Set the new cons ptr.
                    instruction_list.push(Instr::Str(Register::R(2), Register::R(6), 4));
                }

                // Load the operator address into R0
                instruction_list.push(Instr::Ldr(Register::R(0), Register::R(6), 8));
                // Load the args address into R1
                instruction_list.push(Instr::Ldr(Register::R(1), Register::R(6), 4));
                // Emit dispatch instruction.
                instruction_list.push(Instr::Swi(SWI_DISPATCH_INSTRUCTION));
            }

            instruction_list.push(Instr::Ldr(Register::R(4), Register::SP, 0));
            instruction_list.push(Instr::Ldr(Register::R(5), Register::SP, 4));
            instruction_list.push(Instr::Ldr(Register::R(6), Register::SP, 8));
            instruction_list.push(Instr::Ldr(Register::R(7), Register::SP, 12));
            instruction_list.push(Instr::Subi(Register::SP, Register::FP, 4));
            instruction_list.push(Instr::Pop(vec![Register::FP, Register::LR]));
            instruction_list.push(Instr::Bx(Register::LR));

            instruction_list[0] =
                Instr::Long(new_code_address as usize + 4 * instruction_list.len());
            for arg in arg_addresses.iter() {
                instruction_list.push(Instr::Long(*arg as usize));
            }

            // Allocate space for this thunk.
            self.mem.write_u32(
                alloc_address,
                (new_code_address + instruction_list.len() as u32 * 4) as u32,
            );

            let mut relocations = Vec::new();
            for (i, instr) in instruction_list.iter().enumerate() {
                let mut encoded = Vec::new();
                instr.encode(&mut encoded, &mut relocations, "");
                self.mem.write_u32(
                    new_code_address + (i * 4) as u32,
                    (encoded[0] as u32
                        | (encoded[1] as u32) << 8
                        | (encoded[2] as u32) << 16
                        | (encoded[3] as u32) << 24) as u32,
                );
            }

            self.cpu.reg_set(Mode::User, reg::LR, current_pc + 8);
            self.cpu.reg_set(Mode::User, reg::PC, current_pc + 4);
            self.cpu.reg_set(Mode::User, 1, new_code_address + 12);
            None
        } else if value == SWI_DISPATCH_INSTRUCTION {
            // Grab the sexp for this operation.
            let r0_value = self.cpu.reg_get(Mode::User, 0);
            let operator = self.get_sexp(&srcloc, r0_value);
            let r1_value = self.cpu.reg_get(Mode::User, 1);
            let args = self.get_sexp(&srcloc, r1_value);
            let mut allocator = Allocator::new();
            self.do_apply_op(runner, &srcloc, operator, args)
        } else if (value & 15) == SWI_PRINT_EXPR {
            let register = (value >> 4) & 15;
            let label = (value >> 8);
            let r0_value = self.cpu.reg_get(Mode::User, register as u8);
            let printed_expr = self.get_sexp(&srcloc, r0_value).to_string();
            if label != 0 || register != 0 {
                self.pending_gdb_console_output
                    .push(format!("CLVM({label:x}): r{register} = {printed_expr}"));
            } else {
                self.pending_gdb_console_output
                    .push(format!("CLVM: {printed_expr}"));
            }
            self.cpu.reg_set(Mode::User, reg::PC, pc + 4);
            Some(Event::Output)
        } else {
            self.cpu.reg_set(Mode::User, reg::PC, pc + 4);
            Some(Event::Break)
        }
    }

    /// single-step the interpreter
    pub fn step(&mut self) -> Option<Event> {
        // let mut hit_watchpoint = None;

        let pc = self.cpu.reg_get(Mode::User, reg::PC);
        let snoop_instruction = self.mem.r32(pc);

        if (snoop_instruction & 0x0f000000) == 0x0f000000 {
            // This is a trap instruction, interpret it.
            let cpsr = self.cpu.reg_get(Mode::User, reg::CPSR);
            let match_expression = snoop_instruction >> 28;
            let perform_action = match match_expression {
                0 => ((cpsr >> 30) & 1) != 0,
                10 => ((cpsr >> 31) & 1) == ((cpsr >> 28) & 1),
                14 => true,
                _ => todo!("match arm condition {match_expression}"),
            };
            if perform_action {
                let trap_result = self.do_trap(pc, (snoop_instruction & 0xffffff) as usize);
                if trap_result.is_some() {
                    return trap_result;
                }
            } else {
                self.cpu.reg_set(Mode::User, reg::PC, pc + 4);
            }
        } else {
            self.cpu.step(&mut self.mem);
        }

        let pc = self.cpu.reg_get(Mode::User, reg::PC);

        if self.breakpoints.contains(&pc) {
            return Some(Event::Break);
        }

        // if let Some(access) = hit_watchpoint {
        //     let fixup = if self.cpu.thumb_mode() { 2 } else { 4 };
        //     self.cpu.reg_set(Mode::User, reg::PC, pc - fixup);

        //     return Some(match access.kind {
        //         AccessKind::Read => Event::WatchRead(access.addr),
        //         AccessKind::Write => Event::WatchWrite(access.addr),
        //     });
        // }

        None
    }

    /// run the emulator in accordance with the currently set `ExecutionMode`.
    ///
    /// since the emulator runs in the same thread as the GDB loop, the emulator
    /// will use the provided callback to poll the connection for incoming data
    /// every 1024 steps.
    pub fn run(&mut self, mut poll_incoming_data: impl FnMut() -> bool) -> RunEvent {
        match self.exec_mode {
            ExecMode::Step => RunEvent::Event(self.step().unwrap_or(Event::DoneStep)),
            ExecMode::Continue => {
                let mut cycles = 0;
                loop {
                    if cycles % 1024 == 0 {
                        // poll for incoming data
                        if poll_incoming_data() {
                            break RunEvent::IncomingData;
                        }
                    }
                    cycles += 1;

                    if let Some(event) = self.step() {
                        break RunEvent::Event(event);
                    };
                }
            }
            // just continue, but with an extra PC check
            ExecMode::RangeStep(start, end) => {
                let mut cycles = 0;
                loop {
                    if cycles % 1024 == 0 {
                        // poll for incoming data
                        if poll_incoming_data() {
                            break RunEvent::IncomingData;
                        }
                    }
                    cycles += 1;

                    if let Some(event) = self.step() {
                        break RunEvent::Event(event);
                    };

                    if !(start..end).contains(&self.cpu.reg_get(self.cpu.mode(), reg::PC)) {
                        break RunEvent::Event(Event::DoneStep);
                    }
                }
            }
        }
    }
}

impl Emu {
    /// Get an SExp at a specific address.
    fn get_sexp(&self, srcloc: &Srcloc, addr: u32) -> Rc<SExp> {
        if addr == 0 {
            return Rc::new(SExp::Nil(srcloc.clone()));
        }
        let first = self.mem.read_u32(addr);
        if first == 0 || (first & 1) != 0 {
            // Atom
            let size = first >> 1;
            let result: Vec<u8> = (0..size).map(|i| self.mem.read_u8(addr + 4 + i)).collect();
            Rc::new(SExp::Atom(srcloc.clone(), result))
        } else {
            // Cons
            let rest = self.mem.read_u32(addr + 4);
            let f = self.get_sexp(srcloc, first);
            let r = self.get_sexp(srcloc, rest);
            Rc::new(SExp::Cons(srcloc.clone(), f, r))
        }
    }

    /// Run to completion and return a value by address for tests.
    #[cfg(test)]
    fn run_to_exit(
        program: &[u8],
        start_addr: u32,
        clvm_symbols: Rc<HashMap<String, String>>,
    ) -> DynResult<Option<Rc<SExp>>> {
        let srcloc = Srcloc::start("*emu*");
        let mut emu = Emu::new(program, start_addr, clvm_symbols)?;
        let mut elf_loader = ElfLoader::new(program, start_addr).expect("should load");
        elf_loader.load(&mut emu.mem);

        loop {
            let step_result = emu.step();
            match step_result {
                Some(Event::Halted) => {
                    let r0 = emu.cpu.reg_get(Mode::User, 0);
                    return Ok(Some(emu.get_sexp(&srcloc, r0)));
                }
                Some(Event::Trap) => {
                    return Ok(None);
                }
                _ => {}
            }
        }
    }

    #[cfg(test)]
    fn compile_and_run(filename: &str, program: &str, env: &str) -> DynResult<Option<Rc<SExp>>> {
        let srcloc = Srcloc::start(filename);
        let env_parsed = parse_sexp(srcloc.clone(), env.bytes()).expect("should parse");
        let mut allocator = Allocator::new();
        let mut symbol_table = HashMap::new();
        let runner: Rc<dyn TRunProgram> = Rc::new(DefaultProgramRunner::new());
        let search_paths = vec![];
        let opts = Rc::new(DefaultCompilerOpts::new(filename))
            .set_dialect(AcceptedDialect {
                stepping: Some(23),
                strict: true,
                int_fix: true,
                extra_numeric_constants: false,
            })
            .set_optimize(true)
            .set_search_paths(&search_paths)
            .set_frontend_opt(false);

        let parsed_program = parse_sexp(srcloc.clone(), program.bytes())
            .map_err(|e| format!("failed to parse chialisp program {filename}"))?;
        let fe = frontend(opts.clone(), &parsed_program)
            .map_err(|e| format!("failed to compose frontend program"))?;
        let range_results: HashMap<String, Srcloc> = fe
            .compileform()
            .helpers
            .iter()
            .map(|h| (decode_string(h.name()), h.loc()))
            .collect();

        let compiled = compile_file(&mut allocator, runner, opts, program, &mut symbol_table)
            .expect("should compile").to_sexp();
        build_symbol_table_mut(&mut symbol_table, &compiled);
        let tmpfile = NamedTempFile::new().expect("should be able to make a temp file");
        let tmpname = tmpfile.path().to_str().unwrap().to_string();
        let symbols = Rc::new(symbol_table);
        let generator = Program::new(
            range_results,
            filename,
            &tmpname,
            Rc::new(compiled),
            env_parsed[0].clone(),
            TARGET_ADDR,
            symbols.clone(),
        )
        .expect("should be generatable");
        let elf_data = generator.to_elf(&tmpname).expect("should generate");
        Emu::run_to_exit(&elf_data, TARGET_ADDR, symbols)
    }
}

#[test]
fn test_run_to_exit_and_return_nil() {
    let elf = fs::read("resources/tests/armjit/return_nil.elf").expect("should exist");
    let result = Emu::run_to_exit(&elf, TARGET_ADDR, Rc::new(HashMap::default()))
        .expect("should load")
        .unwrap();
    assert_eq!(result.to_string(), "()");
}

#[test]
fn test_run_to_exit_and_return_pair() {
    let elf = fs::read("resources/tests/armjit/return_cons.elf").expect("should exist");
    let result = Emu::run_to_exit(&elf, TARGET_ADDR, Rc::new(HashMap::default()))
        .expect("should load")
        .unwrap();
    assert_eq!(result.to_string(), "(hi . there)");
}

#[test]
fn test_compile_and_run_simple_quoted_atom() {
    let result = Emu::compile_and_run("test.clsp", "(mod () \"hi there\")", "()")
        .expect("should run")
        .unwrap();
    assert_eq!(
        result,
        Rc::new(SExp::Atom(Srcloc::start("*test*"), b"hi there".to_vec()))
    );
}

#[test]
fn test_compile_and_run_cons() {
    let result = Emu::compile_and_run(
        "test.clsp",
        "(mod () (include *standard-cl-23*) (c \"hi\" \"there\"))",
        "()",
    )
    .expect("should run")
    .unwrap();
    assert_eq!(result.to_string(), "(hi . there)");
}

#[test]
fn test_compile_and_run_apply_simple_1() {
    let result = Emu::compile_and_run(
        "test.clsp",
        "(mod () (include *standard-cl-23*) (a 1 (q . \"toot\")))",
        "()",
    )
    .expect("should run")
    .unwrap();
    assert_eq!(result.to_string(), "toot");
}

#[test]
fn test_compile_and_run_apply_simple_2() {
    let result = Emu::compile_and_run(
        "test.clsp",
        "(mod () (include *standard-cl-23*) (a 1 @))",
        "37777",
    )
    .expect("should run")
    .unwrap();
    assert_eq!(result.to_string(), "37777");
}

#[test]
fn test_compile_and_run_apply_simple_3() {
    let result = Emu::compile_and_run(
        "test.clsp",
        "(mod () (include *standard-cl-23*) (a (q 4 (1 . 1) (1 . 2)) @))",
        "()",
    )
    .expect("should run")
    .unwrap();
    assert_eq!(result.to_string(), "(1 . 2)");
}

#[test]
fn test_compile_and_run_apply_simple_4() {
    let result = Emu::compile_and_run(
        "test.clsp",
        "(mod () (include *standard-cl-23*) (f (q 1 2)))",
        "()",
    )
    .expect("should run")
    .unwrap();
    assert_eq!(result.to_string(), "1");
}

#[test]
fn test_compile_and_run_apply_simple_4_fail() {
    let result = Emu::compile_and_run(
        "test.clsp",
        "(mod () (include *standard-cl-23*) (f 99))",
        "()",
    )
    .expect("should run");
    assert!(result.is_none());
}

#[test]
fn test_compile_and_run_apply_simple_5() {
    let result = Emu::compile_and_run(
        "test.clsp",
        "(mod () (include *standard-cl-23*) (r (q 1 2)))",
        "()",
    )
    .expect("should run")
    .unwrap();
    assert_eq!(result.to_string(), "(2)");
}

#[test]
fn test_compile_and_run_apply_simple_6() {
    let result = Emu::compile_and_run(
        "test.clsp",
        "(mod () (include *standard-cl-23*) (r 99))",
        "()",
    )
    .expect("should run");
    assert!(result.is_none());
}

#[test]
fn test_compile_and_run_apply_at() {
    let result = Emu::compile_and_run(
        "test.clsp",
        "(mod (A) (include *standard-cl-23*) @)",
        "(19)",
    )
    .expect("should run")
    .unwrap();
    assert_eq!(result.to_string(), "(19)");
}

#[test]
fn test_compile_and_run_apply_path() {
    let result = Emu::compile_and_run(
        "test.clsp",
        "(mod (A) (include *standard-cl-23*) A)",
        "(19)",
    )
    .expect("should run")
    .unwrap();
    assert_eq!(result.to_string(), "19");
}

#[test]
fn test_compile_and_run_apply_simple_op() {
    let result = Emu::compile_and_run(
        "test.clsp",
        "(mod (A B) (include *standard-cl-23*) (+ A B))",
        "(99 103)",
    )
    .expect("should run")
    .unwrap();
    assert_eq!(result.to_string(), "202");
}

#[test]
fn test_compile_and_run_apply_simple_op1() {
    let result = Emu::compile_and_run(
        "test.clsp",
        "(mod (A B) (include *standard-cl-23*) (+ 1 A B))",
        "(99 103)",
    )
    .expect("should run")
    .unwrap();
    assert_eq!(result.to_string(), "203");
}

#[test]
fn test_compile_and_run_apply_simple_function_0() {
    let result = Emu::compile_and_run(
        "test.clsp",
        "(mod (A B) (include *standard-cl-23*) (defun F (A B) (+ 1 A B)) (F A B))",
        "(99 103)",
    )
    .expect("should run")
    .unwrap();
    assert_eq!(result.to_string(), "203");
}

#[test]
fn test_compile_and_run_apply_function_1() {
    let result = Emu::compile_and_run(
        "test.clsp",
        "(mod (A) (include *standard-cl-23*) (defun F (A) (+ 1 A)) (F A))",
        "(17)",
    )
    .expect("should run")
    .unwrap();
    assert_eq!(result.to_string(), "18");
}

pub enum RunEvent {
    IncomingData,
    Event(Event),
}
