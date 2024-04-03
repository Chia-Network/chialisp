// Based on https://github.com/daniel5151/gdbstub/blob/master/examples/armv4t/emu.rs
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

use crate::compiler::debug::armjit::load::ElfLoader;
use crate::compiler::debug::armjit::memory::PagedMemory;

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
        eprintln!("Setting PC to {:#010x?}", start_addr);
        cpu.reg_set(Mode::User, reg::SP, 0x80000000);
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
        self.cpu.reg_set(Mode::User, reg::SP, 0x10000000);
        self.cpu.reg_set(Mode::User, reg::LR, HLE_RETURN_ADDR);
        self.cpu.reg_set(Mode::User, reg::PC, self.start_addr);
        self.cpu.reg_set(Mode::User, reg::CPSR, 0x10);
    }

    fn do_trap(&mut self, pc: u32, value: u32) -> Option<Event> {
        match value {
            SWI_THROW => {
                return Some(Event::Trap);
            }
            SWI_DISPATCH_NEW_CODE => {
                self.cpu.reg_set(Mode::User, reg::PC, pc + 4);
                todo!();
            }
            SWI_DISPATCH_INSTRUCTION => {
                self.cpu.reg_set(Mode::User, reg::PC, pc + 4);
                todo!();
            }
            _ => {
                self.cpu.reg_set(Mode::User, reg::PC, pc + 4);
                return Some(Event::Break);
            }
        }
    }

    /// single-step the interpreter
    pub fn step(&mut self) -> Option<Event> {
        // let mut hit_watchpoint = None;

        self.cpu.step(&mut self.mem);
        let pc = self.cpu.reg_get(Mode::User, reg::PC);

        let snoop_instruction = self.mem.r32(pc);
        if (snoop_instruction & 0x0f000000) == 0x0f000000 {
            // This is a trap instruction, interpret it.
            let cpsr = self.cpu.reg_get(Mode::User, reg::CPSR);
            let match_expression = snoop_instruction >> 28;
            let perform_action =
                match match_expression {
                    0 => ((cpsr >> 29) & 1) != 0,
                    14 => true,
                    _ => todo!()
                };
            if perform_action {
                return self.do_trap(pc, snoop_instruction & 0xffffff);
            } else {
                self.cpu.reg_set(Mode::User, reg::PC, pc + 4);
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

        if self.breakpoints.contains(&pc) {
            return Some(Event::Break);
        }

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

pub enum RunEvent {
    IncomingData,
    Event(Event),
}

