// Based on https://github.com/daniel5151/gdbstub/blob/master/examples/armv4t/main.rs

use gdbstub::common::Signal;
use gdbstub::conn::Connection;
use gdbstub::conn::ConnectionExt;
use gdbstub::stub::run_blocking;
use gdbstub::stub::DisconnectReason;
use gdbstub::stub::GdbStub;
use gdbstub::stub::SingleThreadStopReason;
use gdbstub::target::Target;
use std::net::TcpListener;
use std::net::TcpStream;

use crate::compiler::debug::armjit::emu::{DynResult, Emu, Event, RunEvent};

fn wait_for_tcp(port: u16) -> DynResult<TcpStream> {
    let sockaddr = format!("127.0.0.1:{}", port);
    eprintln!("Waiting for a GDB connection on {:?}...", sockaddr);

    let sock = TcpListener::bind(sockaddr)?;
    let (stream, addr) = sock.accept()?;
    eprintln!("Debugger connected from {}", addr);

    Ok(stream)
}

enum EmuGdbEventLoop {}

fn print_run_event(event: &RunEvent) -> String {
    match event {
        RunEvent::IncomingData => "IncomingData".to_string(),
        RunEvent::Event(event) => format!("Event({event:?})")
    }
}

impl run_blocking::BlockingEventLoop for EmuGdbEventLoop {
    type Target = Emu;
    type Connection = Box<dyn ConnectionExt<Error = std::io::Error>>;
    type StopReason = SingleThreadStopReason<u32>;

    #[allow(clippy::type_complexity)]
    fn wait_for_stop_reason(
        target: &mut Emu,
        conn: &mut Self::Connection,
    ) -> Result<
        run_blocking::Event<SingleThreadStopReason<u32>>,
        run_blocking::WaitForStopReasonError<
            <Self::Target as Target>::Error,
            <Self::Connection as Connection>::Error,
        >,
    > {
        // The `armv4t` example runs the emulator in the same thread as the GDB state
        // machine loop. As such, it uses a simple poll-based model to check for
        // interrupt events, whereby the emulator will check if there is any incoming
        // data over the connection, and pause execution with a synthetic
        // `RunEvent::IncomingData` event.
        //
        // In more complex integrations, the target will probably be running in a
        // separate thread, and instead of using a poll-based model to check for
        // incoming data, you'll want to use some kind of "select" based model to
        // simultaneously wait for incoming GDB data coming over the connection, along
        // with any target-reported stop events.
        //
        // The specifics of how this "select" mechanism work + how the target reports
        // stop events will entirely depend on your project's architecture.
        //
        // Some ideas on how to implement this `select` mechanism:
        //
        // - A mpsc channel
        // - epoll/kqueue
        // - Running the target + stopping every so often to peek the connection
        // - Driving `GdbStub` from various interrupt handlers

        let poll_incoming_data = || {
            // gdbstub takes ownership of the underlying connection, so the `borrow_conn`
            // method is used to borrow the underlying connection back from the stub to
            // check for incoming data.
            conn.peek().map(|b| b.is_some()).unwrap_or(true)
        };

        let run_res = target.run(poll_incoming_data);
        eprintln!("GDB {}", print_run_event(&run_res));
        match run_res {
            RunEvent::IncomingData => {
                let byte = conn
                    .read()
                    .map_err(run_blocking::WaitForStopReasonError::Connection)?;
                Ok(run_blocking::Event::IncomingData(byte))
            }
            RunEvent::Event(event) => {
                use gdbstub::target::ext::breakpoints::WatchKind;

                // translate emulator stop reason into GDB stop reason
                let stop_reason = match event {
                    Event::Trap => SingleThreadStopReason::Signal(Signal::SIGABRT),
                    Event::DoneStep => SingleThreadStopReason::DoneStep,
                    Event::Halted => SingleThreadStopReason::Terminated(Signal::SIGSTOP),
                    Event::Break => SingleThreadStopReason::SwBreak(()),
                    Event::WatchWrite(addr) => SingleThreadStopReason::Watch {
                        tid: (),
                        kind: WatchKind::Write,
                        addr,
                    },
                    Event::WatchRead(addr) => SingleThreadStopReason::Watch {
                        tid: (),
                        kind: WatchKind::Read,
                        addr,
                    },
                };

                Ok(run_blocking::Event::TargetStopped(stop_reason))
            }
        }
    }

    fn on_interrupt(
        _target: &mut Emu,
    ) -> Result<Option<SingleThreadStopReason<u32>>, <Emu as Target>::Error> {
        // Because this emulator runs as part of the GDB stub loop, there isn't any
        // special action that needs to be taken to interrupt the underlying target. It
        // is implicitly paused whenever the stub isn't within the
        // `wait_for_stop_reason` callback.
        Ok(Some(SingleThreadStopReason::Signal(Signal::SIGINT)))
    }
}

pub fn start_stub() -> Result<Box<dyn ConnectionExt<Error = std::io::Error>>, ()> {
    Ok(Box::new(wait_for_tcp(9001).map_err(|_| ())?))
}

pub fn run_stub(connection: Box<dyn ConnectionExt<Error = std::io::Error>>, emu: &mut Emu) -> DynResult<()> {

    let gdb = GdbStub::new(connection);

    match gdb.run_blocking::<EmuGdbEventLoop>(emu) {
        Ok(disconnect_reason) => match disconnect_reason {
            DisconnectReason::Disconnect => {
                println!("GDB client has disconnected. Running to completion...");
                while emu.step() != Some(Event::Halted) {}
            }
            DisconnectReason::TargetExited(code) => {
                println!("Target exited with code {}!", code)
            }
            DisconnectReason::TargetTerminated(sig) => {
                println!("Target terminated with signal {}!", sig)
            }
            DisconnectReason::Kill => println!("GDB sent a kill command!"),
        },
        Err(e) => {
            if e.is_target_error() {
                println!(
                    "target encountered a fatal error: {:?}",
                    e.into_target_error().unwrap()
                )
            } else if e.is_connection_error() {
                let (e, kind) = e.into_connection_error().unwrap();
                println!("connection error: {:?} - {}", kind, e,)
            } else {
                println!("gdbstub encountered a fatal error: {:?}", e)
            }
        }
    }

    let ret = emu.cpu.reg_get(armv4t_emu::Mode::User, 0);

    Ok(())
}
