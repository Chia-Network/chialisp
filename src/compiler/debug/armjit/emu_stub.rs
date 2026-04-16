// Based on https://github.com/daniel5151/gdbstub/blob/master/examples/armv4t/main.rs

use gdbstub::common::Signal;
use gdbstub::conn::Connection;
use gdbstub::conn::ConnectionExt;
use gdbstub::stub::run_blocking;
use gdbstub::stub::DisconnectReason;
use gdbstub::stub::GdbStub;
use gdbstub::stub::SingleThreadStopReason;
use gdbstub::target::Target;
use std::net::{SocketAddr, TcpListener, TcpStream};

use crate::compiler::debug::armjit::emu::{DynResult, Emu, Event, RunEvent};

fn wait_for_tcp(port: Option<u16>) -> DynResult<TcpStream> {
    let sockaddr = format!("127.0.0.1:{}", port.unwrap_or(0));

    let sock = TcpListener::bind(sockaddr)?;
    let local_addr = sock.local_addr();
    eprintln!("Waiting for a GDB connection on {:?}...", local_addr);
    let (stream, addr) = sock.accept()?;
    eprintln!("Debugger connected from {}", addr);

    Ok(stream)
}

enum EmuGdbEventLoop {}

fn hex_ascii(nibble: u8) -> u8 {
    match nibble {
        0..=9 => b'0' + nibble,
        10..=15 => b'a' + (nibble - 10),
        _ => b'0',
    }
}

fn send_gdb_console_packet<C: Connection<Error = std::io::Error>>(
    conn: &mut C,
    message: &str,
) -> Result<(), std::io::Error> {
    let payload = message.as_bytes();
    let mut checksum = b'O';
    conn.write(b'$')?;
    conn.write(b'O')?;
    for byte in payload {
        let hi = hex_ascii(*byte >> 4);
        let lo = hex_ascii(*byte & 0x0f);
        checksum = checksum.wrapping_add(hi);
        checksum = checksum.wrapping_add(lo);
        conn.write(hi)?;
        conn.write(lo)?;
    }
    let newline = b'\n';
    let newline_hi = hex_ascii(newline >> 4);
    let newline_lo = hex_ascii(newline & 0x0f);
    checksum = checksum.wrapping_add(newline_hi);
    checksum = checksum.wrapping_add(newline_lo);
    conn.write(newline_hi)?;
    conn.write(newline_lo)?;
    conn.write(b'#')?;
    conn.write(hex_ascii(checksum >> 4))?;
    conn.write(hex_ascii(checksum & 0x0f))?;
    conn.flush()?;
    Ok(())
}

fn flush_pending_gdb_console_output<C: Connection<Error = std::io::Error>>(
    target: &mut Emu,
    conn: &mut C,
) -> Result<(), std::io::Error> {
    for message in target.take_pending_gdb_console_output() {
        send_gdb_console_packet(conn, &message)?;
    }
    Ok(())
}

fn print_run_event(event: &RunEvent) -> String {
    match event {
        RunEvent::IncomingData => "IncomingData".to_string(),
        RunEvent::Event(event) => format!("Event({event:?})"),
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
        flush_pending_gdb_console_output(target, conn)
            .map_err(run_blocking::WaitForStopReasonError::Connection)?;
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
                    Event::Halted => SingleThreadStopReason::Signal(Signal::SIGSTOP),
                    Event::Output => SingleThreadStopReason::Signal(Signal::SIGUSR1),
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

pub fn start_stub(
    port: Option<u16>,
) -> Result<(SocketAddr, Box<dyn ConnectionExt<Error = std::io::Error>>), ()> {
    let connection = wait_for_tcp(port).map_err(|_| ())?;
    Ok((
        connection.local_addr().map_err(|_| ())?,
        Box::new(connection),
    ))
}

pub fn run_stub(
    connection: Box<dyn ConnectionExt<Error = std::io::Error>>,
    emu: &mut Emu,
) -> Result<(), String> {
    let gdb = GdbStub::new(connection);

    emu.cpu.reg_get(armv4t_emu::Mode::User, 0);

    gdb.run_blocking::<EmuGdbEventLoop>(emu).map_err(|e| format!("Error: {e:?}"))?;

    Ok(())
}
