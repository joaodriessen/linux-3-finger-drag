//! The I/O shell around the gesture state machine.
//!
//! Owns the real touchpad (exclusively grabbed for the program's whole
//! life) and a synthetic uinput clone of it. Raw events are read here,
//! split into frames, and fed to [`GestureMachine`]; the machine's
//! [`Output`]s are applied back to the clone and the virtual mouse.
//! All *decisions* live in `gesture.rs` -- this file only moves bytes.
//!
//! Why a lifetime-long grab + clone (vs. grabbing mid-gesture, this
//! project's first approach): a mid-gesture EVIOCGRAB leaves the
//! compositor's touch tracking permanently corrupted -- it never sees
//! the closing lift-off frames for fingers grabbed away mid-flight, so
//! it's stuck believing they're still down. By owning the real device
//! outright and re-emitting a clean copy, we control every frame the
//! compositor ever sees: either an accurate mirror of the real pad, or
//! an explicit "nothing is touching" state -- never a silent gap.

use std::fs::{File, OpenOptions};
use std::io;
use std::os::fd::{AsRawFd, RawFd};
use std::os::unix::fs::OpenOptionsExt;
use std::time::Instant;

use libc::O_NONBLOCK;
use tracing::{debug, info, warn};

use input_linux::{sys, AbsoluteAxis, AbsoluteInfoSetup, EvdevHandle, EventKind, UInputHandle};

use super::gesture::{Ev, GestureMachine, Output, EV_SYN, MAX_SLOTS, SYN_DROPPED, SYN_REPORT};

const READ_BATCH: usize = 64;

const EV_KEY: u16 = 0x01;
const EV_ABS: u16 = 0x03;
const BTN_LEFT: u16 = 0x110;
const BTN_TOUCH: u16 = 0x14a;
const BTN_TOOL_FINGER: u16 = 0x145;
const ABS_MT_SLOT: u16 = 0x2f;
const ABS_MT_TRACKING_ID: u16 = 0x39;
const ABS_MT_POSITION_X: u16 = 0x35;
const ABS_MT_POSITION_Y: u16 = 0x36;
const ABS_MT_TOUCH_MAJOR: u16 = 0x30;

/// The synthetic finger that carries out 3-finger drags ON THE CLONE:
/// one touch with BTN_LEFT held (the clone is a clickpad, so libinput
/// reads button + moving finger as a drag). Because this motion goes
/// through the same libinput touchpad pipeline as ordinary cursor
/// movement -- same device, same acceleration curve, same speed
/// setting -- drags and the cursor feel identical by construction.
/// (This replaced a separate virtual REL-mouse device, whose different
/// libinput acceleration curve could never be made to match.)
struct DragFinger {
    active: bool,
    /// Sub-unit position accumulator (acceleration multiplier applied).
    pos: (f64, f64),
    emitted: (i32, i32),
    next_id: i32,
}

/// The `phys` marker stamped on our synthetic clone so device discovery
/// can never mistake our own clone for a real touchpad (it impersonates
/// the real device's name, vendor and capabilities *exactly*, so this
/// marker is the only reliable way to tell them apart -- which matters
/// when re-discovering after a hotplug).
pub const CLONE_PHYS_MARKER: &str = "linux-3-finger-drag/proxy";

fn zero_event() -> sys::input_event {
    unsafe { std::mem::zeroed() }
}

/// The kernel's legacy `UI_SET_PHYS` is `_IOW('U', 108, char*)` -- ioctl
/// size = sizeof(char*), argument = pointer to a NUL-terminated string.
/// (input-linux 0.7's binding encodes size 1, which the kernel rejects
/// with EINVAL, so we issue the ioctl ourselves.)
fn ui_set_phys(fd: RawFd, phys: &std::ffi::CStr) -> io::Result<()> {
    const IOC_WRITE: libc::c_ulong = 1;
    let cmd: libc::c_ulong = (IOC_WRITE << 30)
        | ((std::mem::size_of::<*const libc::c_char>() as libc::c_ulong) << 16)
        | ((b'U' as libc::c_ulong) << 8)
        | 108;
    let rc = unsafe { libc::ioctl(fd, cmd as _, phys.as_ptr()) };
    if rc < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

fn to_raw(ev: &Ev) -> sys::input_event {
    let mut raw = zero_event();
    raw.type_ = ev.type_;
    raw.code = ev.code;
    raw.value = ev.value;
    raw
}

pub struct MtProxy {
    real: EvdevHandle<File>,
    synth: UInputHandle<File>,
    raw_fd: RawFd,
    slot_count: usize,
    /// Axis ranges of the pad, for centering/wrapping the drag finger.
    x_range: (i32, i32),
    y_range: (i32, i32),
    /// Plausible contact size for the synthetic finger (libinput's
    /// touch-size quirks discard touches that never report one).
    synth_touch_major: Option<i32>,
    /// Drag speed multiplier relative to cursor speed (config
    /// `acceleration`; 1.0 = drags feel exactly like cursor movement).
    accel: f64,
    drag: DragFinger,
    frame: Vec<Ev>,
    /// True between a SYN_DROPPED and the SYN_REPORT that closes it:
    /// per the evdev protocol, everything in that window is garbage and
    /// must be discarded, with state re-read from the kernel afterward.
    dropping: bool,
    read_buf: [sys::input_event; READ_BATCH],
}

impl AsRawFd for MtProxy {
    fn as_raw_fd(&self) -> RawFd {
        self.raw_fd
    }
}

impl MtProxy {
    /// Opens the real touchpad at `path`, grabs it exclusively for the
    /// rest of the program's life, and creates a synthetic clone with
    /// identical capabilities for the compositor to read instead.
    pub fn new(path: &str) -> io::Result<Self> {
        let real_file = OpenOptions::new()
            .read(true)
            .custom_flags(O_NONBLOCK)
            .open(path)?;
        let raw_fd = real_file.as_raw_fd();
        let real = EvdevHandle::new(real_file);

        // held for the entire program lifetime -- see module doc for why
        // this must never be released mid-gesture
        real.grab(true)?;
        info!("Exclusively grabbed the real trackpad at {}.", path);

        let synth = Self::clone_device(&real)?;

        // The device's real slot range: snapshot ioctls sized past it
        // return zeroed entries whose tracking_id 0 reads as "finger
        // down" -- the phantom-touch bug. Ask the device, don't assume.
        let slot_count = real
            .absolute_info(AbsoluteAxis::MultitouchSlot)
            .map(|i| (i.maximum as usize + 1).clamp(1, MAX_SLOTS))
            .unwrap_or(MAX_SLOTS);
        let x_range = real
            .absolute_info(AbsoluteAxis::MultitouchPositionX)
            .map(|i| (i.minimum, i.maximum))
            .unwrap_or((0, 1000));
        let y_range = real
            .absolute_info(AbsoluteAxis::MultitouchPositionY)
            .map(|i| (i.minimum, i.maximum))
            .unwrap_or((0, 1000));
        // a mid-scale fingertip: big enough to pass libinput's touch-size
        // quirks, small enough never to read as a palm/thumb
        let synth_touch_major = real
            .absolute_info(AbsoluteAxis::MultitouchTouchMajor)
            .ok()
            .map(|i| i.minimum + (i.maximum - i.minimum) / 4);

        Ok(MtProxy {
            real,
            synth,
            raw_fd,
            slot_count,
            x_range,
            y_range,
            synth_touch_major,
            accel: 1.0,
            drag: DragFinger {
                active: false,
                pos: (0.0, 0.0),
                emitted: (0, 0),
                next_id: 61000,
            },
            frame: Vec::with_capacity(READ_BATCH),
            dropping: false,
            read_buf: [zero_event(); READ_BATCH],
        })
    }

    /// Drag speed relative to cursor speed (hot-reloadable).
    pub fn set_accel(&mut self, accel: f64) {
        self.accel = accel;
    }

    pub fn slot_count(&self) -> usize {
        self.slot_count
    }
    pub fn x_extent(&self) -> f64 {
        f64::from(self.x_range.1 - self.x_range.0)
    }
    pub fn y_extent(&self) -> f64 {
        f64::from(self.y_range.1 - self.y_range.0)
    }

    /// Builds a synthetic uinput device with the same EV_KEY/EV_ABS/
    /// INPUT_PROP capabilities as the real device, so the compositor's
    /// libinput sees something functionally identical to the hardware.
    fn clone_device(real: &EvdevHandle<File>) -> io::Result<UInputHandle<File>> {
        let uinput_file = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(O_NONBLOCK)
            .open("/dev/uinput")?;
        let uinput_fd = uinput_file.as_raw_fd();
        let synth = UInputHandle::new(uinput_file);

        synth.set_evbit(EventKind::Key)?;
        synth.set_evbit(EventKind::Absolute)?;

        for key in real.key_bits()?.iter() {
            synth.set_keybit(key)?;
        }

        let mut abs_setups = Vec::new();
        for axis in real.absolute_bits()?.iter() {
            synth.set_absbit(axis)?;
            let info = real.absolute_info(axis)?;
            abs_setups.push(AbsoluteInfoSetup { axis, info });
        }

        for prop in real.device_properties()?.iter() {
            synth.set_propbit(prop)?;
        }

        // Impersonate the real device's identity (vendor/product/name),
        // not just its capabilities. KDE keys its per-device libinput
        // settings (natural scroll, accel profile, tap-to-click...) in
        // kcminputrc by exactly this triple -- a clone with a made-up
        // identity is "new" to KDE and silently falls back to defaults,
        // which is what caused scrolling to come back reversed when this
        // proxy first replaced the real device as KWin's input source.
        // Matching identity means the user's saved preferences apply
        // automatically, with nothing to keep in sync.
        let real_id = real.device_id()?;
        let mut real_name = real.device_name()?;
        while real_name.last() == Some(&0) {
            real_name.pop();
        }
        // ...but stamp our marker into `phys` (which KDE ignores) so
        // device discovery can always tell the clone from the original.
        let phys = std::ffi::CString::new(CLONE_PHYS_MARKER).expect("no NUL in marker");
        ui_set_phys(uinput_fd, &phys)
            .map_err(|e| io::Error::new(e.kind(), format!("set_phys: {e}")))?;
        synth
            .create(&real_id, &real_name, 0, &abs_setups)
            .map_err(|e| io::Error::new(e.kind(), format!("uinput create: {e}")))?;
        debug!(
            "Synthetic touchpad clone created, impersonating \"{}\".",
            String::from_utf8_lossy(&real_name)
        );

        // give udev/the compositor a beat to pick the new device up
        // before events start flowing through it
        std::thread::sleep(std::time::Duration::from_millis(500));

        Ok(synth)
    }

    /// Drains every event currently readable, feeding complete frames to
    /// the machine and applying its outputs. Returns when the fd would
    /// block. An `ENODEV` error means the device was unplugged /
    /// re-enumerated; the caller handles re-discovery.
    pub fn drain(&mut self, machine: &mut GestureMachine) -> io::Result<()> {
        loop {
            let n = match self.real.read(&mut self.read_buf) {
                Ok(n) => n,
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => return Ok(()),
                Err(e) => return Err(e),
            };
            if n == 0 {
                return Ok(());
            }

            for i in 0..n {
                let raw = self.read_buf[i];
                if self.dropping {
                    // evdev protocol: after SYN_DROPPED, everything up to
                    // and including the next SYN_REPORT is unreliable --
                    // discard it, then re-read authoritative state.
                    if raw.type_ == EV_SYN && raw.code == SYN_REPORT {
                        self.dropping = false;
                        let snapshot = self.slot_snapshot()?;
                        let outs = machine.on_resync(&snapshot, Instant::now());
                        self.apply(&outs)?;
                    }
                    continue;
                }

                if raw.type_ == EV_SYN && raw.code == SYN_DROPPED {
                    warn!("Kernel reported dropped events; resyncing slot state.");
                    self.frame.clear();
                    self.dropping = true;
                    continue;
                }

                self.frame.push(Ev::new(raw.type_, raw.code, raw.value));

                if raw.type_ == EV_SYN && raw.code == SYN_REPORT {
                    let outs = machine.on_frame(&self.frame, Instant::now());
                    self.frame.clear();
                    self.apply(&outs)?;
                }
            }
        }
    }

    /// Applies the machine's outputs to the clone, in order.
    pub fn apply(&mut self, outputs: &[Output]) -> io::Result<()> {
        for output in outputs {
            match output {
                Output::EmitSynth(evs) => {
                    let raw: Vec<sys::input_event> = evs.iter().map(to_raw).collect();
                    self.synth.write(&raw)?;
                }
                Output::MouseDown => self.drag_down()?,
                Output::MouseUp => self.drag_up()?,
                Output::MouseMove { dx, dy } => self.drag_move(*dx, *dy)?,
            }
        }
        Ok(())
    }

    fn write_evs(&self, evs: &[(u16, u16, i32)]) -> io::Result<()> {
        let mut raw: Vec<sys::input_event> = evs
            .iter()
            .map(|&(t, c, v)| {
                let mut e = zero_event();
                e.type_ = t;
                e.code = c;
                e.value = v;
                e
            })
            .collect();
        let mut syn = zero_event();
        syn.type_ = EV_SYN;
        syn.code = SYN_REPORT;
        raw.push(syn);
        self.synth.write(&raw)?;
        Ok(())
    }

    fn center(&self) -> (i32, i32) {
        (
            (self.x_range.0 + self.x_range.1) / 2,
            (self.y_range.0 + self.y_range.1) / 2,
        )
    }

    fn touch_frame(&mut self, at: (i32, i32), with_button: bool) -> Vec<(u16, u16, i32)> {
        let id = self.drag.next_id;
        self.drag.next_id = if id >= 65000 { 61000 } else { id + 1 };
        let mut evs = vec![
            (EV_ABS, ABS_MT_SLOT, 0),
            (EV_ABS, ABS_MT_TRACKING_ID, id),
            (EV_ABS, ABS_MT_POSITION_X, at.0),
            (EV_ABS, ABS_MT_POSITION_Y, at.1),
        ];
        if let Some(major) = self.synth_touch_major {
            evs.push((EV_ABS, ABS_MT_TOUCH_MAJOR, major));
        }
        evs.push((EV_KEY, BTN_TOUCH, 1));
        evs.push((EV_KEY, BTN_TOOL_FINGER, 1));
        if with_button {
            evs.push((EV_KEY, BTN_LEFT, 1));
        }
        evs
    }

    /// Begin the drag: a synthetic finger lands at pad center with
    /// BTN_LEFT pressed.
    fn drag_down(&mut self) -> io::Result<()> {
        let c = self.center();
        self.drag.pos = (f64::from(c.0), f64::from(c.1));
        self.drag.emitted = c;
        self.drag.active = true;
        let frame = self.touch_frame(c, true);
        self.write_evs(&frame)
    }

    /// Move the synthetic finger by the reference finger's delta (in pad
    /// units, scaled by the relative `acceleration`). Approaching a pad
    /// edge lifts and re-lands the finger at center -- BTN_LEFT stays
    /// held, so libinput keeps the drag alive through the hop.
    fn drag_move(&mut self, dx: i32, dy: i32) -> io::Result<()> {
        if !self.drag.active {
            return Ok(());
        }
        self.drag.pos.0 += f64::from(dx) * self.accel;
        self.drag.pos.1 += f64::from(dy) * self.accel;

        let margin_x = (self.x_range.1 - self.x_range.0) / 10;
        let margin_y = (self.y_range.1 - self.y_range.0) / 10;
        let near_edge = self.drag.pos.0 < f64::from(self.x_range.0 + margin_x)
            || self.drag.pos.0 > f64::from(self.x_range.1 - margin_x)
            || self.drag.pos.1 < f64::from(self.y_range.0 + margin_y)
            || self.drag.pos.1 > f64::from(self.y_range.1 - margin_y);
        if near_edge {
            // hop: lift (button stays), re-land centered
            self.write_evs(&[
                (EV_ABS, ABS_MT_SLOT, 0),
                (EV_ABS, ABS_MT_TRACKING_ID, -1),
                (EV_KEY, BTN_TOUCH, 0),
                (EV_KEY, BTN_TOOL_FINGER, 0),
            ])?;
            let c = self.center();
            self.drag.pos = (f64::from(c.0), f64::from(c.1));
            self.drag.emitted = c;
            let frame = self.touch_frame(c, false); // button already held
            return self.write_evs(&frame);
        }

        let v = (
            self.drag.pos.0.round() as i32,
            self.drag.pos.1.round() as i32,
        );
        if v != self.drag.emitted {
            let mut evs = vec![(EV_ABS, ABS_MT_SLOT, 0)];
            if v.0 != self.drag.emitted.0 {
                evs.push((EV_ABS, ABS_MT_POSITION_X, v.0));
            }
            if v.1 != self.drag.emitted.1 {
                evs.push((EV_ABS, ABS_MT_POSITION_Y, v.1));
            }
            self.drag.emitted = v;
            self.write_evs(&evs)?;
        }
        Ok(())
    }

    /// End the drag: release the finger and the button.
    fn drag_up(&mut self) -> io::Result<()> {
        if !self.drag.active {
            return Ok(());
        }
        self.drag.active = false;
        self.write_evs(&[
            (EV_ABS, ABS_MT_SLOT, 0),
            (EV_ABS, ABS_MT_TRACKING_ID, -1),
            (EV_KEY, BTN_TOUCH, 0),
            (EV_KEY, BTN_TOOL_FINGER, 0),
            (EV_KEY, BTN_LEFT, 0),
        ])
    }

    /// Defensive: release the drag finger/button if a drag was live
    /// (shutdown, device loss).
    pub fn release_drag(&mut self) -> io::Result<()> {
        self.drag_up()
    }

    /// Authoritative per-slot state straight from the kernel
    /// (EVIOCGMTSLOTS), sized to the device's true slot range.
    fn slot_snapshot(&self) -> io::Result<Vec<(i32, i32, i32)>> {
        let mut ids = vec![0i32; self.slot_count];
        self.real
            .multi_touch_slots(AbsoluteAxis::MultitouchTrackingId, &mut ids)?;
        let mut xs = vec![0i32; self.slot_count];
        self.real
            .multi_touch_slots(AbsoluteAxis::MultitouchPositionX, &mut xs)?;
        let mut ys = vec![0i32; self.slot_count];
        self.real
            .multi_touch_slots(AbsoluteAxis::MultitouchPositionY, &mut ys)?;

        Ok((0..self.slot_count)
            .map(|s| (ids[s], xs[s], ys[s]))
            .collect())
    }

    /// Destroys the synthetic clone (the grab on the real device is
    /// released automatically when the fd closes on drop).
    pub fn destruct(self) -> io::Result<()> {
        self.synth.dev_destroy()
    }
}
