// Based on https://github.com/daniel5151/gdbstub/blob/master/examples/armv4t/emu.rs

use std::fs;
use std::collections::HashMap;
use std::rc::Rc;

use clvmr::Allocator;
use tempfile::NamedTempFile;

use armv4t_emu::reg;
use armv4t_emu::Cpu;
use armv4t_emu::Memory;
use armv4t_emu::Mode;
use gdbstub::arch::Arch;
use gdbstub::common::Pid;
use gdbstub::target::{Target, TargetResult};
use gdbstub::target::ext::base::{BaseOps, single_register_access};
use gdbstub::target::ext::breakpoints::{Breakpoints, BreakpointsOps, HwBreakpointOps, HwWatchpointOps, SwBreakpoint, SwBreakpointOps};
use gdbstub::target::ext::base::singlethread::{SingleThreadBase, SingleThreadResume, SingleThreadResumeOps};

use crate::classic::clvm_tools::stages::stage_0::DefaultProgramRunner;
use crate::compiler::TRunProgram;
use crate::compiler::compiler::{compile_file, DefaultCompilerOpts};
use crate::compiler::comptypes::CompilerOpts;
use crate::compiler::debug::armjit::code::{Program, SWI_DONE, SWI_THROW, SWI_DISPATCH_NEW_CODE, SWI_DISPATCH_INSTRUCTION, TARGET_ADDR};
use crate::compiler::debug::armjit::load::ElfLoader;
use crate::compiler::debug::armjit::memory::{PagedMemory, TargetMemory};
use crate::compiler::debug::build_symbol_table_mut;
use crate::compiler::dialect::AcceptedDialect;
use crate::compiler::srcloc::Srcloc;
use crate::compiler::sexp::{decode_string, parse_sexp, SExp};

pub type DynResult<T> = Result<T, Box<dyn std::error::Error>>;

const HLE_RETURN_ADDR: u32 = 0x12345678;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Event {
    DoneStep,
    Halted,
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

    // example custom register. only read/written to from the GDB client
    pub custom_reg: u32,

    pub exec_mode: ExecMode,

    pub cpu: Cpu,
    pub mem: PagedMemory,

    pub watchpoints: Vec<u32>,
    pub breakpoints: Vec<u32>,
    pub files: Vec<Option<std::fs::File>>,

    pub reported_pid: Pid,
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
    fn write_registers(&mut self, regs: &<Self::Arch as Arch>::Registers)
        -> TargetResult<(), Self> {
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
        for (i,d) in data.iter_mut().enumerate() {
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
        let found = self.breakpoints.iter().position(|u| *u == (addr as u32)).clone();
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
    fn resume(&mut self, sig: std::option::Option<gdbstub::common::Signal>) -> Result<(), <Self as gdbstub::target::Target>::Error> {
        return Ok(());
    }
}

impl Emu {
    pub fn new(program_elf: &[u8], start_addr: u32) -> DynResult<Emu> {
        // set up emulated system
        let mut cpu = Cpu::new();
        let mut mem = PagedMemory::default();

        // copy all in-memory sections from the ELF file into system RAM
        let mut elf_loader = ElfLoader::new(program_elf, start_addr).expect("should load");
        elf_loader.load(&mut mem);

        // setup execution state
        cpu.reg_set(Mode::User, reg::SP, 0xffffff00);
        cpu.reg_set(Mode::User, reg::LR, HLE_RETURN_ADDR);
        cpu.reg_set(Mode::User, reg::PC, start_addr);
        cpu.reg_set(Mode::User, reg::CPSR, 0x10); // user mode

        Ok(Emu {
            start_addr: start_addr,

            custom_reg: 0x12345678,

            exec_mode: ExecMode::Continue,

            cpu,
            mem,

            watchpoints: Vec::new(),
            breakpoints: Vec::new(),
            files: Vec::new(),

            reported_pid: Pid::new(1).unwrap(),
        })
    }

    pub(crate) fn reset(&mut self) {
        self.cpu.reg_set(Mode::User, reg::SP, 0xffffff00);
        self.cpu.reg_set(Mode::User, reg::LR, HLE_RETURN_ADDR);
        self.cpu.reg_set(Mode::User, reg::PC, self.start_addr);
        self.cpu.reg_set(Mode::User, reg::CPSR, 0x10);
    }

    fn do_trap(&mut self, pc: u32, value: usize) -> Option<Event> {
        eprintln!("trap {value:x}");
        if value == SWI_DONE {
            Some(Event::Halted)
        } else if value == SWI_THROW {
            Some(Event::Trap)
        } else if value == SWI_DISPATCH_NEW_CODE {
            self.cpu.reg_set(Mode::User, reg::PC, pc + 4);
            todo!();
        } else if value == SWI_DISPATCH_INSTRUCTION {
            self.cpu.reg_set(Mode::User, reg::PC, pc + 4);
            todo!();
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

        eprintln!("pc {pc:x} snoop {snoop_instruction:x}");
        if (snoop_instruction & 0x0f000000) == 0x0f000000 {
            // This is a trap instruction, interpret it.
            let cpsr = self.cpu.reg_get(Mode::User, reg::CPSR);
            let match_expression = snoop_instruction >> 28;
            eprintln!("cpsr {cpsr:x} match {match_expression:x}");
            let perform_action =
                match match_expression {
                    0 => ((cpsr >> 29) & 1) != 0,
                    14 => true,
                    _ => todo!()
                };
            if perform_action {
                return self.do_trap(pc, (snoop_instruction & 0xffffff) as usize);
            } else {
                self.cpu.reg_set(Mode::User, reg::PC, pc + 4);
            }
        } else {
            self.cpu.step(&mut self.mem);
            let pc = self.cpu.reg_get(Mode::User, reg::PC);

            if self.breakpoints.contains(&pc) {
                return Some(Event::Break);
            }
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
    #[cfg(test)]
    fn get_sexp(&self, srcloc: &Srcloc, addr: u32) -> Rc<SExp> {
        let first = self.mem.read_u32(addr);
        if first == 0 || (first & 1) != 0 {
            // Atom
            let size = first >> 1;
            let result: Vec<u8> = (0..size).map(|i| {
                self.mem.read_u8(addr + 4 + i)
            }).collect();
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
    fn run_to_exit(program: &[u8], start_addr: u32) -> DynResult<Option<Rc<SExp>>> {
        let srcloc = Srcloc::start("*emu*");
        let mut emu = Emu::new(program, start_addr)?;
        let mut elf_loader = ElfLoader::new(program, start_addr).expect("should load");
        elf_loader.load(&mut emu.mem);

        loop {
            let step_result = emu.step();
            eprintln!("step_result {step_result:?}");
            match step_result {
                Some(Event::Halted) => {
                    let r0 = emu.cpu.reg_get(Mode::User, 0);
                    return Ok(Some(emu.get_sexp(&srcloc, r0)));
                }
                Some(Event::Trap) => {
                    return Ok(None);
                }
                _ => { }
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
            })
            .set_optimize(true)
            .set_search_paths(&search_paths)
            .set_frontend_opt(false);
        let compiled =
            compile_file(
                &mut allocator,
                runner,
                opts,
                program,
                &mut symbol_table,
            ).expect("should compile");
        build_symbol_table_mut(&mut symbol_table, &compiled);
        let generator = Program::new(
            filename,
            Rc::new(compiled),
            env_parsed[0].clone(),
            symbol_table
        ).expect("should be generatable");
        let tmpfile = NamedTempFile::new().expect("should be able to make a temp file");
        let tmpname = tmpfile.path().to_str().unwrap().to_string();
        let elf_data = generator.to_elf(&tmpname).expect("should generate");
        Emu::run_to_exit(&elf_data, TARGET_ADDR)
    }
}

#[test]
fn test_run_to_exit_and_return_nil() {
    let elf = fs::read("resources/tests/armjit/return_nil.elf").expect("should exist");
    let result = Emu::run_to_exit(&elf, TARGET_ADDR).expect("should load").unwrap();
    assert_eq!(result.to_string(), "()");
}

#[test]
fn test_run_to_exit_and_return_pair() {
    let elf = fs::read("resources/tests/armjit/return_cons.elf").expect("should exist");
    let result = Emu::run_to_exit(&elf, TARGET_ADDR).expect("should load").unwrap();
    assert_eq!(result.to_string(), "(hi . there)");
}

#[test]
fn test_compile_and_run_simple_quoted_atom() {
    let result = Emu::compile_and_run(
        "test.clsp",
        "(mod () \"hi there\")",
        "()"
    ).expect("should run").unwrap();
    assert_eq!(result, Rc::new(SExp::Atom(Srcloc::start("*test*"), b"hi there".to_vec())));
}

#[test]
fn test_compile_and_run_cons() {
    let result = Emu::compile_and_run(
        "test.clsp",
        "(mod () (include *standard-cl-23*) (c \"hi\" \"there\"))",
        "()"
    ).expect("should run").unwrap();
    assert_eq!(result.to_string(), "(hi . there)");
}

#[test]
fn test_compile_and_run_apply_simple_1() {
    let result = Emu::compile_and_run(
        "test.clsp",
        "(mod () (include *standard-cl-23*) (a 1 (q . \"toot\")))",
        "()"
    ).expect("should run").unwrap();
    assert_eq!(result.to_string(), "toot");
}

#[test]
fn test_compile_and_run_apply_simple_2() {
    let result = Emu::compile_and_run(
        "test.clsp",
        "(mod () (include *standard-cl-23*) (a 1 @))",
        "37777"
    ).expect("should run").unwrap();
    assert_eq!(result.to_string(), "37777");
}

#[test]
fn test_compile_and_run_apply_simple_3() {
    let result = Emu::compile_and_run(
        "test.clsp",
        "(mod () (include *standard-cl-23*) (a (q 4 (1 . 1) (1 . 2)) @))",
        "()"
    ).expect("should run").unwrap();
    assert_eq!(result.to_string(), "(1 . 2)");
}

#[test]
fn test_compile_and_run_apply_simple_4() {
    let result = Emu::compile_and_run(
        "test.clsp",
        "(mod () (include *standard-cl-23*) (f (q 1 2)))",
        "()"
    ).expect("should run").unwrap();
    assert_eq!(result.to_string(), "1");
}

#[test]
fn test_compile_and_run_apply_simple_4_fail() {
    let result = Emu::compile_and_run(
        "test.clsp",
        "(mod () (include *standard-cl-23*) (f 99))",
        "()"
    ).expect("should run");
    assert!(result.is_none());
}

#[test]
fn test_compile_and_run_apply_simple_5() {
    let result = Emu::compile_and_run(
        "test.clsp",
        "(mod () (include *standard-cl-23*) (r (q 1 2)))",
        "()"
    ).expect("should run").unwrap();
    assert_eq!(result.to_string(), "(2)");
}

#[test]
fn test_compile_and_run_apply_simple_6() {
    let result = Emu::compile_and_run(
        "test.clsp",
        "(mod () (include *standard-cl-23*) (r 99))",
        "()"
    ).expect("should run");
    assert!(result.is_none());
}

pub enum RunEvent {
    IncomingData,
    Event(Event),
}

